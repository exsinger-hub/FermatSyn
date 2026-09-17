"""Multi-path Fermat scanning (MP-Fermat) and a pluggable scan-order zoo.

This module is the runnable core of the journal-extension contribution
**Multi-path Fermat scanning (MP-Fermat)**.  It replaces the original
hard-coded 64x64 ``*.npy`` permutation matrices (see
``models/fermat_path_generate.py`` and ``models/spiral_path.py``) with:

1. A *dynamic* Fermat / multi-path-Fermat path generator that works for an
   arbitrary ``H x W`` feature map (no pre-baked ``.npy`` files needed).
2. A *pluggable* scan-order interface (``SCAN_REGISTRY`` + ``MultiPathScan``)
   so a scan-only controlled ablation can be run by swapping a single option:
   ``raster / snake / rect_spiral / hilbert / zorder / random / fermat /
   mp_fermat``.

--------------------------------------------------------------------------
Serialization mechanism (ported from ``models/modules.py``)
--------------------------------------------------------------------------
In the original ``MambaLayerOnlyspiral.forward`` (modules.py ~L661-698) a
feature map ``x`` of shape ``(B, C, H, W)`` is serialized to a token sequence
``(B, N, C)`` (``N = H*W``) and back using dense ``N x N`` permutation
matrices loaded from disk::

    x1 = x.view(B, C, -1)                                  # (B, C, N)
    x1 = einsum('ij,klj->kli', spiral_eye, x1).permute(0,2,1)   # (B, N, C)
    ...                                                    # run Mamba
    out1 = einsum('ij,klj->kli', despiral_eye,
                  mamba_out1.permute(0,2,1)).view(B, C, H, W)

``spiral_eye`` is the one-hot matrix with ``spiral_eye[k, idx[k]] = 1`` (built
in ``fermat_path_generate.py`` via ``matrix[arange(N), spiral_indices] = 1``).
The einsum therefore computes ``out_seq[b,c,k] = x_flat[b,c, idx[k]]`` -- i.e.
a plain **gather** along the spatial axis with the permutation ``idx``.
``despiral_eye = spiral_eye.T`` performs the **inverse** gather, which is just
an ``index_select`` with the inverse permutation ``argsort(idx)``.
The ``despiral_r_eye`` matrix is the same trick for the flipped (reverse-scan)
ordering ``idx.flip(0)``.

This module reproduces that mapping with ``torch.index_select`` instead of the
wasteful dense ``N x N`` (4096x4096 for 64x64) matmul.  ``index_select`` is
differentiable, so gradients flow exactly as before.

--------------------------------------------------------------------------
How ``MambaLayerOnlyspiral`` would consume this (integration note, STEP 4)
--------------------------------------------------------------------------
The drop-in replacement only touches ``MambaLayerOnlyspiral`` in modules.py:

* ``__init__`` (L656-659): delete the three ``np.load(...)`` lines and the
  ``self.*_eye`` tensors; instead store ``self.scan = MultiPathScan(scan=opt,
  K=K, bidirectional=True)``.  Keep ``self.conv1d``, but make its channel count
  ``2*K*dim -> dim`` (currently 576->288 assumes 2 scans of 288 channels) so it
  works for arbitrary K and dim.
* ``forward`` (L661-695): replace the ``view + einsum`` serialization with::

      seqs = self.scan(x)            # list of (B, N, C) sequences
  run each through Mamba, then::
      outs = [self.scan.deserialize(o, i) for i, o in enumerate(...)]
  ``cat`` the ``outs`` on the channel dim and apply ``self.conv1d``.
* Remove the hard-coded ``output.view(-1, 288, 64, 64)`` at L695 -- use the
  ``H, W`` captured at the top of ``forward`` instead (the whole point of this
  module is to stop hard-coding 64x64).

With ``scan="fermat"`` (or ``mp_fermat`` with ``K=1``) and
``bidirectional=True`` this reproduces the original two-scan behavior exactly.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ===========================================================================
# Single-curve generators
# ===========================================================================
def generate_fermat_indices(H, W, phase=0.0, alpha=None, lambda_c=0.0,
                            n_candidates=None):
    """Single golden-angle Fermat (sunflower) scan order for an ``H x W`` grid.

    Ports the greedy nearest-neighbour grid-matching logic from
    ``models/fermat_path_generate.py`` but parameterized by ``H, W`` and an
    angular ``phase`` offset (radians) so that several decorrelated paths can
    be produced (see :func:`generate_multipath_fermat_indices`).

    Continuity-constrained matching (paper Eq.7).  For spiral step ``k`` and an
    unused candidate grid cell ``u`` the assignment score is::

        Score_u = (1 - lambda_c) * d_Fermat(u, p_k) / eta_f
                  +    lambda_c  * d_contin(u, pi_{k-1}) / eta_c

    where ``d_Fermat`` is the Euclidean distance from cell ``u`` to the
    *continuous* Fermat point ``p_k`` (global isotropy / coverage term) and
    ``d_contin`` is the distance from ``u`` to the *previously assigned* grid
    cell ``pi_{k-1}`` (local path-continuity term).  The greedy assignment
    picks the unused cell minimizing ``Score_u``.

    Normalizers: both ``eta_f`` and ``eta_c`` are set to the grid diagonal
    ``sqrt((H-1)^2 + (W-1)^2)`` -- a fixed, scale-stable choice so the two
    terms live on a comparable ``[0, ~1]`` range and ``lambda_c`` is a clean
    convex weight (median-candidate-distance was the documented alternative;
    grid-diagonal is preferred here because it is deterministic and does not
    drift as the unused set shrinks).

    Limiting cases:
        * ``lambda_c == 0`` -> pure nearest-Fermat-point greedy = the original
          ``fermat_path_generate.py`` behaviour (used for the K=1 equivalence).
        * ``lambda_c == 1`` -> pure continuity-chasing (greedy nearest-unused
          walk seeded from the central first spiral point).

    Efficiency / approximation: with ``n_candidates`` set, each step only
    scores the ``n_candidates`` unused cells closest to ``p_k`` (a local
    neighbourhood around the continuous Fermat point) instead of all ``N``
    cells; this is an approximation of the global argmin that is exact when the
    chosen cell lies within that neighbourhood.  Default ``n_candidates=None``
    keeps the exact full-grid argmin (the safe fallback, used for the small
    feature-map grids in this codebase, e.g. 64x64).

    Args:
        H, W (int): feature map height / width.
        phase (float): angular offset added to every spiral sample (radians).
        alpha (float, optional): radial scale.  Defaults to the same value used
            by the original code (``r_max / sqrt(N-1)``).
        lambda_c (float): continuity weight in ``[0, 1]`` (paper Eq.7).
        n_candidates (int, optional): restrict candidates per step (see above).

    Returns:
        LongTensor of length ``H*W``: a permutation of ``0..H*W-1`` (row-major
        flat indices ``r*W + c``) giving the visiting order.
    """
    c_x = (W - 1) / 2.0
    c_y = (H - 1) / 2.0
    golden_angle_rad = math.radians(137.508)
    N = H * W

    # Corner radius, identical convention to the original implementation.
    r_max = math.sqrt((W - 1 - c_x) ** 2 + (H - 1 - c_y) ** 2)
    if alpha is None:
        alpha = r_max / math.sqrt(max(N - 1, 1))

    indices = np.arange(N)
    theta = indices * golden_angle_rad + phase
    r = alpha * np.sqrt(indices)
    x_spiral = r * np.cos(theta)
    y_spiral = r * np.sin(theta)

    # Grid coordinates.  NOTE: the original code uses (i - c_x, j - c_y) with
    # i over rows and j over cols, and flat index i*W + j (row-major).  We keep
    # that exact convention so K=1 matches the existing .npy ordering.
    rows = np.repeat(np.arange(H), W)            # i (row) for each flat index
    cols = np.tile(np.arange(W), H)              # j (col) for each flat index
    grid_x = rows - c_x
    grid_y = cols - c_y

    # Eq.7 normalizers (grid diagonal; >0 for any 1x2+ grid, guard tiny grids).
    eta = math.sqrt((H - 1) ** 2 + (W - 1) ** 2) or 1.0
    eta_f = eta_c = eta

    result_indices = []
    used = np.zeros(N, dtype=bool)
    prev = -1  # pi_{k-1}; -1 means "no previous cell yet" (k == 0)

    use_cand = (n_candidates is not None) and (n_candidates < N)

    for k in range(N):
        # d_Fermat term: distance from every grid cell to continuous point p_k.
        dxf = grid_x - x_spiral[k]
        dyf = grid_y - y_spiral[k]
        d_fermat = np.sqrt(dxf * dxf + dyf * dyf)

        if lambda_c == 0.0 or prev < 0:
            # Pure-Fermat scoring (also the k==0 case: no previous cell).
            score = d_fermat
        else:
            dxc = grid_x - grid_x[prev]
            dyc = grid_y - grid_y[prev]
            d_contin = np.sqrt(dxc * dxc + dyc * dyc)
            score = ((1.0 - lambda_c) * d_fermat / eta_f
                     + lambda_c * d_contin / eta_c)

        if use_cand:
            # Restrict to the n_candidates unused cells nearest to p_k.
            d_masked = np.where(used, np.inf, d_fermat)
            cand = np.argpartition(d_masked, n_candidates)[:n_candidates]
            sc = score[cand].copy()
            sc[used[cand]] = np.inf          # never pick an already-used cell
            g = int(cand[int(np.argmin(sc))])
        else:
            score = np.where(used, np.inf, score)  # mask used cells
            g = int(np.argmin(score))

        result_indices.append(g)
        used[g] = True
        prev = g

    # (greedy with masking always assigns exactly N distinct points)
    assert len(result_indices) == N
    assert len(set(result_indices)) == N, "Fermat indices contain duplicates"
    return torch.tensor(result_indices, dtype=torch.long)


def generate_multipath_fermat_indices(H, W, K=2, lambda_c=0.0,
                                      n_candidates=None):
    """K decorrelated Fermat paths; path ``i`` uses ``phase = 2*pi*i/K``.

    ``K=1`` (with ``lambda_c=0``) is exactly
    ``[generate_fermat_indices(H, W, phase=0.0)]``.  ``lambda_c`` (paper Eq.7
    continuity weight) is applied identically to every path.

    Returns:
        list of ``K`` LongTensors, each a permutation of ``0..H*W-1``.
    """
    assert K >= 1
    paths = []
    for i in range(K):
        phase = 2.0 * math.pi * i / K
        paths.append(generate_fermat_indices(
            H, W, phase=phase, lambda_c=lambda_c, n_candidates=n_candidates))
    return paths


# Default direction-balance weight for the registered "dir_balanced" scan.
# mu=0.03 is the max-locality Pareto-dominating point of the geometric sweep
# (A_hat 0.32->0.24, nbr 0.94->0.96 at lambda_c=0.7); mu in [0.01, 0.3] is the
# safe regime (mu>=0.5 degrades locality). See
# Miccai2026-paper3777/figures/exp_dirbalanced_matching.py for the sweep.
DIRBALANCED_MU = 0.03


def generate_dirbalanced_indices(H, W, phase=0.0, alpha=None, lambda_c=0.7,
                                 mu=DIRBALANCED_MU, n_dir_bins=8,
                                 n_candidates=None):
    """Direction-balanced (harmonic-aware) Fermat grid-matching.

    Identical to :func:`generate_fermat_indices` but augments the Eq.7 score
    with a step-direction-balancing penalty::

        Score(u) = (1-lambda_c)*d_Fermat(u,p_k)/eta
                 +    lambda_c *d_contin(u,prev)/eta
                 +        mu   *freq_dir( bin(beta_u) )

    where ``beta_u = atan2(row_u-row_prev, col_u-col_prev)`` is the candidate
    step orientation and ``freq_dir`` the running normalized frequency of that
    direction bin among placed steps.  Penalizing over-used directions breaks
    the 4-fold grid-locked anisotropy that the multi-path phase offsets cannot
    touch (the bias lives in the matching, not the global phase rotation).

    With ``mu == 0`` this is byte-identical to :func:`generate_fermat_indices`
    (verified by ``torch.equal``).  At ``lambda_c=0.7, mu=0.03`` it
    Pareto-dominates the plain Fermat operating point on BOTH isotropy and
    locality (A_hat 0.32->0.24, R4 0.71->0.24, nbr 0.94->0.96).
    """
    c_x = (W - 1) / 2.0
    c_y = (H - 1) / 2.0
    golden_angle_rad = math.radians(137.508)
    N = H * W

    r_max = math.sqrt((W - 1 - c_x) ** 2 + (H - 1 - c_y) ** 2)
    if alpha is None:
        alpha = r_max / math.sqrt(max(N - 1, 1))

    indices = np.arange(N)
    theta = indices * golden_angle_rad + phase
    r = alpha * np.sqrt(indices)
    x_spiral = r * np.cos(theta)
    y_spiral = r * np.sin(theta)

    rows = np.repeat(np.arange(H), W)            # row i of each flat index
    cols = np.tile(np.arange(W), H)              # col j of each flat index
    grid_x = rows - c_x
    grid_y = cols - c_y
    rows_f = rows.astype(np.float64)
    cols_f = cols.astype(np.float64)

    eta = math.sqrt((H - 1) ** 2 + (W - 1) ** 2) or 1.0
    eta_f = eta_c = eta

    binw = 2.0 * np.pi / n_dir_bins
    dir_counts = np.zeros(n_dir_bins, dtype=np.float64)
    n_steps = 0

    result_indices = []
    used = np.zeros(N, dtype=bool)
    prev = -1
    use_cand = (n_candidates is not None) and (n_candidates < N)

    for k in range(N):
        dxf = grid_x - x_spiral[k]
        dyf = grid_y - y_spiral[k]
        d_fermat = np.sqrt(dxf * dxf + dyf * dyf)

        if lambda_c == 0.0 or prev < 0:
            score = d_fermat.copy()
        else:
            dxc = grid_x - grid_x[prev]
            dyc = grid_y - grid_y[prev]
            d_contin = np.sqrt(dxc * dxc + dyc * dyc)
            score = ((1.0 - lambda_c) * d_fermat / eta_f
                     + lambda_c * d_contin / eta_c)

        # direction-balancing penalty (inactive at mu==0 or the first step)
        if mu > 0.0 and prev >= 0:
            beta = np.arctan2(rows_f - rows_f[prev], cols_f - cols_f[prev])
            binidx = np.mod(np.round(beta / binw).astype(np.int64), n_dir_bins)
            if n_steps > 0:
                score = score + mu * (dir_counts / n_steps)[binidx]

        if use_cand:
            d_masked = np.where(used, np.inf, d_fermat)
            cand = np.argpartition(d_masked, n_candidates)[:n_candidates]
            sc = score[cand].copy()
            sc[used[cand]] = np.inf
            g = int(cand[int(np.argmin(sc))])
        else:
            score = np.where(used, np.inf, score)
            g = int(np.argmin(score))

        # record the realized step's direction bin before advancing prev
        if prev >= 0:
            beta_g = math.atan2(rows_f[g] - rows_f[prev],
                                cols_f[g] - cols_f[prev])
            dir_counts[int(round(beta_g / binw)) % n_dir_bins] += 1.0
            n_steps += 1

        result_indices.append(g)
        used[g] = True
        prev = g

    assert len(result_indices) == N
    assert len(set(result_indices)) == N, "dir-balanced indices have duplicates"
    return torch.tensor(result_indices, dtype=torch.long)


def generate_raster_indices(H, W):
    """Row-major raster order (the trivial identity permutation)."""
    return torch.arange(H * W, dtype=torch.long)


def generate_snake_indices(H, W):
    """Boustrophedon / snake order: even rows L->R, odd rows R->L."""
    idx = []
    for r in range(H):
        cols = range(W) if (r % 2 == 0) else range(W - 1, -1, -1)
        for c in cols:
            idx.append(r * W + c)
    return torch.tensor(idx, dtype=torch.long)


def generate_rect_spiral_indices(H, W):
    """Rectangular (clockwise, inward) spiral order.

    Ported verbatim from ``models/spiral_path.generate_spiral_indices``.
    """
    indices = []
    left, right, top, bottom = 0, W - 1, 0, H - 1
    while left <= right and top <= bottom:
        for i in range(left, right + 1):
            indices.append(top * W + i)
        top += 1
        for i in range(top, bottom + 1):
            indices.append(i * W + right)
        right -= 1
        if top <= bottom:
            for i in range(right, left - 1, -1):
                indices.append(bottom * W + i)
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                indices.append(i * W + left)
            left += 1
    return torch.tensor(indices, dtype=torch.long)


def _morton_key(x, y, bits):
    """Interleave the bits of ``x`` (low) and ``y`` (high) -> Morton code."""
    key = 0
    for i in range(bits):
        key |= ((x >> i) & 1) << (2 * i)
        key |= ((y >> i) & 1) << (2 * i + 1)
    return key


def generate_zorder_indices(H, W):
    """Z-order (Morton) curve.

    Non-power-of-two sizes are handled implicitly: every in-range cell still
    gets a unique Morton key, and we simply sort the in-range flat indices by
    that key (out-of-range cells of the bounding power-of-two square never
    appear), yielding a valid permutation.
    """
    bits = max(1, math.ceil(math.log2(max(H, W, 2))))
    flat = np.arange(H * W)
    rows = flat // W
    cols = flat % W
    keys = np.array([_morton_key(int(c), int(r), bits)
                     for r, c in zip(rows, cols)])
    order = np.argsort(keys, kind="stable")
    return torch.tensor(flat[order], dtype=torch.long)


def _hilbert_d2xy(n, d):
    """Map Hilbert distance ``d`` to ``(x, y)`` on an ``n x n`` curve (n=2^k)."""
    rx = ry = 0
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # rotate quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def generate_hilbert_indices(H, W):
    """Hilbert space-filling curve order.

    Non-power-of-two handling: we pad the grid up to the next power of two
    ``p = 2**ceil(log2(max(H, W)))``, walk the full ``p x p`` Hilbert curve,
    and keep only the visited cells that fall inside the real ``H x W`` region
    (``x < W and y < H``), preserving their Hilbert order.  The retained cells
    form a valid permutation of ``0..H*W-1`` whose locality is the Hilbert
    curve's, minus the padding.
    """
    p = 1
    while p < max(H, W):
        p *= 2
    p = max(p, 1)
    idx = []
    for d in range(p * p):
        x, y = _hilbert_d2xy(p, d)  # x -> column, y -> row
        if x < W and y < H:
            idx.append(y * W + x)
    assert len(idx) == H * W
    return torch.tensor(idx, dtype=torch.long)


def generate_random_indices(H, W, seed=0):
    """A fixed random permutation determined by ``seed`` (reproducible)."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randperm(H * W, generator=g)


# ===========================================================================
# Registry
# ===========================================================================
# Convention:
#   * "mp_fermat" -> callable(H, W, K=...) returning a *list* of K LongTensors.
#   * every other entry -> callable(H, W) returning a *single* LongTensor.
# Use get_scan_indices() / MultiPathScan to abstract over this distinction.
SCAN_REGISTRY = {
    "raster": generate_raster_indices,
    "snake": generate_snake_indices,
    "rect_spiral": generate_rect_spiral_indices,
    "zorder": generate_zorder_indices,
    "hilbert": generate_hilbert_indices,
    "random": generate_random_indices,
    "fermat": generate_fermat_indices,
    "mp_fermat": generate_multipath_fermat_indices,
    "dir_balanced": generate_dirbalanced_indices,
}


# ===========================================================================
# (De)serialization helpers
# ===========================================================================
def inverse_indices(idx):
    """Inverse permutation of ``idx`` (``inv[idx[k]] = k``), via argsort."""
    return torch.argsort(idx)


def serialize(x, idx):
    """Serialize a feature map to a token sequence along a scan order.

    Args:
        x (Tensor): ``(B, C, H, W)``.
        idx (LongTensor): permutation of length ``H*W``.

    Returns:
        Tensor ``(B, N, C)`` with ``seq[b, k, c] = x_flat[b, c, idx[k]]`` --
        identical to the original ``einsum('ij,klj->kli', spiral_eye, x_flat)``
        followed by ``.permute(0, 2, 1)``.
    """
    B, C, H, W = x.shape
    idx = idx.to(x.device)
    x_flat = x.reshape(B, C, H * W)
    seq = x_flat.index_select(2, idx)          # (B, C, N) gathered
    return seq.permute(0, 2, 1).contiguous()   # (B, N, C)


def deserialize(x_seq, idx, H, W):
    """Inverse of :func:`serialize`: token sequence -> feature map.

    Args:
        x_seq (Tensor): ``(B, N, C)``.
        idx (LongTensor): the *same* permutation used by :func:`serialize`.
        H, W (int): target spatial size.

    Returns:
        Tensor ``(B, C, H, W)``.  Differentiable (pure ``index_select``).
    """
    B, N, C = x_seq.shape
    assert N == H * W, f"sequence length {N} != H*W {H * W}"
    inv = inverse_indices(idx).to(x_seq.device)
    seq = x_seq.permute(0, 2, 1).contiguous()  # (B, C, N)
    out = seq.index_select(2, inv)             # scatter back via inverse perm
    return out.reshape(B, C, H, W)


# ===========================================================================
# Caching layer
# ===========================================================================
def get_scan_indices(name, H, W, K=2, lambda_c=0.0, mu=DIRBALANCED_MU,
                     device="cpu", cache={}):
    """Memoized scan-index lookup.

    Returns a *list* of LongTensors regardless of scan type (single-path scans
    return a one-element list) so callers can iterate uniformly.  Paths for a
    given ``(name, H, W, K, lambda_c, mu)`` are computed only once.  ``lambda_c``
    (paper Eq.7 continuity weight) affects the Fermat scans; ``mu`` is the
    direction-balance weight of the ``dir_balanced`` scan (ignored otherwise).

    Note: the default mutable ``cache={}`` argument is intentional -- it acts
    as a module-level memo shared across calls.
    """
    key = (name, H, W, K, lambda_c, mu)
    if key not in cache:
        if name not in SCAN_REGISTRY:
            raise KeyError(f"unknown scan '{name}'; choose from "
                           f"{sorted(SCAN_REGISTRY)}")
        fn = SCAN_REGISTRY[name]
        if name == "mp_fermat":
            paths = fn(H, W, K=K, lambda_c=lambda_c)
        elif name == "fermat":
            paths = [fn(H, W, lambda_c=lambda_c)]
        elif name == "dir_balanced":
            paths = [fn(H, W, lambda_c=lambda_c, mu=mu)]
        else:
            paths = [fn(H, W)]
        cache[key] = [p.long() for p in paths]
    return [p.to(device) for p in cache[key]]


# ===========================================================================
# Pluggable nn.Module
# ===========================================================================
class MultiPathScan(nn.Module):
    """Framework-light pluggable scan front-end.

    Produces the list of serialized token sequences for a feature map and
    remembers the indices so the matching :meth:`deserialize` inverts them.
    It does **not** call Mamba -- it only handles the (de)serialization so it
    can be dropped into ``MambaLayerOnlyspiral`` (see module docstring).

    Args:
        scan (str): a key of :data:`SCAN_REGISTRY`.
        K (int): number of paths (only meaningful for ``mp_fermat``).
        bidirectional (bool): if True, every path also yields its reverse scan
            (matching the original forward+flip two-scan design).
        lambda_c (float): Eq.7 continuity weight for the Fermat scans.
    """

    def __init__(self, scan="mp_fermat", K=2, bidirectional=True, lambda_c=0.0,
                 mu=DIRBALANCED_MU):
        super().__init__()
        if scan not in SCAN_REGISTRY:
            raise KeyError(f"unknown scan '{scan}'")
        self.scan = scan
        self.K = K
        self.bidirectional = bidirectional
        self.lambda_c = lambda_c
        self.mu = mu  # direction-balance weight (only used by "dir_balanced")
        # Populated on each forward(); each entry is (idx, reversed_flag).
        self._plan = []

    def num_sequences(self, H=None, W=None):
        """Number of sequences forward() will emit (paths * directions)."""
        n_paths = self.K if self.scan == "mp_fermat" else 1
        return n_paths * (2 if self.bidirectional else 1)

    def forward(self, x):
        """Serialize ``x`` ``(B, C, H, W)`` into a list of ``(B, N, C)`` seqs."""
        B, C, H, W = x.shape
        idx_list = get_scan_indices(self.scan, H, W, K=self.K,
                                    lambda_c=self.lambda_c, mu=self.mu,
                                    device=x.device)
        self._plan = []
        self._hw = (H, W)
        seqs = []
        for idx in idx_list:
            seq = serialize(x, idx)
            seqs.append(seq)
            self._plan.append((idx, False))
            if self.bidirectional:
                seqs.append(torch.flip(seq, dims=[1]))  # reverse-scan
                self._plan.append((idx, True))
        return seqs

    def deserialize(self, x_seq, i):
        """Invert the ``i``-th sequence produced by the last :meth:`forward`."""
        idx, reversed_flag = self._plan[i]
        H, W = self._hw
        if reversed_flag:
            x_seq = torch.flip(x_seq, dims=[1])
        return deserialize(x_seq, idx, H, W)


# ===========================================================================
# Anisotropy diagnostics (paper evidence) + self-test
# ===========================================================================
def _path_step_stats(idx, H, W, nbins=16):
    """Diagnostics for one scan path.

    Returns ``(mean_step, dir_anisotropy)`` where:

    * ``mean_step`` -- mean Euclidean spacing between consecutive visited cells.
    * ``dir_anisotropy`` in ``[0, 1]`` -- directional anisotropy of the scan,
      defined as ``1 - H(theta) / log2(nbins)`` where ``H`` is the Shannon
      entropy of the histogram of consecutive step *directions* (angles).
      Axis-aligned scans (raster / snake / rect_spiral / hilbert) move along
      only a few discrete directions, so their direction histogram is peaky
      -> low entropy -> high anisotropy (near 1).  The golden-angle Fermat /
      MP-Fermat scans spread step directions almost uniformly over the circle
      -> high entropy -> low anisotropy (near 0).  Lower is more isotropic.
    """
    idx_np = idx.numpy()
    r = idx_np // W
    c = idx_np % W
    dr = np.diff(r.astype(np.float64))
    dc = np.diff(c.astype(np.float64))
    steps = np.sqrt(dr * dr + dc * dc)
    ang = np.arctan2(dr, dc)                          # step direction
    hist, _ = np.histogram(ang, bins=nbins, range=(-np.pi, np.pi))
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    entropy = -np.sum(p * np.log2(p))
    anisotropy = 1.0 - entropy / np.log2(nbins)
    return float(steps.mean()), float(anisotropy)


def _pearson(a, b):
    """Pearson correlation of two 1-D arrays (no scipy dependency)."""
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def compute_scan_metrics(idx, H, W, nbins=16, n_pairs=20000, seed=0):
    """Isotropy<->locality trade-off quantifier for one scan permutation.

    Args:
        idx (LongTensor): permutation of ``0..H*W-1`` (the scan order).
        H, W (int): grid size.
        nbins (int): direction-histogram resolution for anisotropy.
        n_pairs (int): random cell pairs sampled for ``locality_corr``.
        seed (int): RNG seed for the pair sampling (reproducible).

    Returns:
        dict with:
          * ``directional_anisotropy`` -- ``1 - entropy(step-direction hist)
            / log2(nbins)``; lower = more isotropic step directions.
          * ``nn_spacing_var`` -- variance of consecutive-step (jump) distances
            along the scan, a coverage/spacing-uniformity proxy (for the
            continuous golden-angle spiral this mirrors the near-uniform
            sunflower point-set spacing; on the grid it captures how erratic
            the realized jumps are).
          * ``locality_corr`` -- Pearson correlation between 2-D Euclidean
            distance and 1-D sequence-index distance over sampled cell pairs;
            HIGHER = better locality preservation.
          * ``mean_step`` / ``p95_step`` -- mean and 95th-percentile jump
            distance between consecutive visited cells.
          * ``true_neighbor_frac`` -- fraction of consecutive sequence pairs
            that are true 8-grid-neighbours (step distance <= sqrt(2)); a clean
            locality proxy, HIGHER = better.
    """
    idx_np = idx.numpy()
    r = (idx_np // W).astype(np.float64)
    c = (idx_np % W).astype(np.float64)

    # Consecutive-step jump distances.
    dr = np.diff(r)
    dc = np.diff(c)
    steps = np.sqrt(dr * dr + dc * dc)

    # Directional anisotropy (same definition as the 64x64 table).
    ang = np.arctan2(dr, dc)
    hist, _ = np.histogram(ang, bins=nbins, range=(-np.pi, np.pi))
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    entropy = -np.sum(p * np.log2(p))
    anisotropy = 1.0 - entropy / np.log2(nbins)

    # true 8-neighbour fraction (4-neighbours included).
    true_neighbor_frac = float(np.mean(steps <= math.sqrt(2.0) + 1e-9))

    # Locality correlation: 2-D spatial distance vs 1-D sequence-position
    # distance over random pairs.  seqpos[g] = step at which grid cell g is
    # visited (the inverse permutation).
    seqpos = np.argsort(idx_np).astype(np.float64)  # seqpos[idx[k]] = k
    N = H * W
    rng = np.random.default_rng(seed)
    m = min(n_pairs, N * (N - 1) // 2)
    a = rng.integers(0, N, size=m)
    b = rng.integers(0, N, size=m)
    keep = a != b
    a, b = a[keep], b[keep]
    ra, ca = a // W, a % W
    rb, cb = b // W, b % W
    d2d = np.sqrt((ra - rb) ** 2 + (ca - cb) ** 2).astype(np.float64)
    d1d = np.abs(seqpos[a] - seqpos[b])
    locality_corr = _pearson(d2d, d1d)

    return {
        "directional_anisotropy": float(anisotropy),
        "nn_spacing_var": float(steps.var()),
        "locality_corr": float(locality_corr),
        "mean_step": float(steps.mean()),
        "p95_step": float(np.percentile(steps, 95)),
        "true_neighbor_frac": true_neighbor_frac,
    }


def sweep_lambda_c(H, W, K=2, lambdas=(0.0, 0.3, 0.5, 0.7, 0.9, 1.0),
                   verbose=True):
    """Print/return the isotropy<->locality trade-off vs Eq.7 ``lambda_c``.

    For each ``lambda_c`` the K mp_fermat paths are generated and their
    :func:`compute_scan_metrics` are averaged over the K paths.  This is the
    raw material for the paper's section 3.3 trade-off curve / figure.

    Returns:
        dict mapping ``lambda_c -> {metric: averaged value}``.
    """
    metric_keys = ["directional_anisotropy", "locality_corr",
                   "true_neighbor_frac", "mean_step", "p95_step",
                   "nn_spacing_var"]
    results = {}
    if verbose:
        print(f"\nmp_fermat lambda_c sweep @ {H}x{W} (K={K}, avg over K paths):")
        header = f"  {'lambda_c':>8}" + "".join(f"{k:>14}" for k in [
            "anisotropy", "locality_r", "nbr_frac",
            "mean_step", "p95_step", "spacing_var"])
        print(header)
    for lam in lambdas:
        paths = get_scan_indices("mp_fermat", H, W, K=K, lambda_c=lam)
        agg = {k: [] for k in metric_keys}
        for idx in paths:
            mt = compute_scan_metrics(idx, H, W)
            for k in metric_keys:
                agg[k].append(mt[k])
        row = {k: float(np.mean(v)) for k, v in agg.items()}
        results[lam] = row
        if verbose:
            print(f"  {lam:>8.2f}"
                  f"{row['directional_anisotropy']:>14.4f}"
                  f"{row['locality_corr']:>14.4f}"
                  f"{row['true_neighbor_frac']:>14.4f}"
                  f"{row['mean_step']:>14.4f}"
                  f"{row['p95_step']:>14.4f}"
                  f"{row['nn_spacing_var']:>14.4f}")
    return results


def _run_self_test(verbose=True):
    sizes = [(32, 32), (64, 64), (17, 29)]
    single_scans = ["raster", "snake", "rect_spiral", "zorder",
                    "hilbert", "random", "fermat"]

    for (H, W) in sizes:
        N = H * W
        # ---- validity of every generator (permutation check) ----
        for name in single_scans:
            idx = get_scan_indices(name, H, W)[0]
            assert idx.numel() == N, f"{name} {H}x{W}: wrong length"
            assert torch.equal(torch.unique(idx), torch.arange(N)), \
                f"{name} {H}x{W}: not a valid permutation"
        # mp_fermat with K=2 and K=1
        mp2 = get_scan_indices("mp_fermat", H, W, K=2)
        assert len(mp2) == 2
        for p, idx in enumerate(mp2):
            assert torch.equal(torch.unique(idx), torch.arange(N)), \
                f"mp_fermat path {p} {H}x{W}: not a permutation"
        # K=1 mp_fermat == fermat(phase=0)
        mp1 = get_scan_indices("mp_fermat", H, W, K=1)[0]
        fer = get_scan_indices("fermat", H, W)[0]
        assert torch.equal(mp1, fer), \
            f"mp_fermat(K=1) != fermat at {H}x{W}"

        # ---- continuity-constrained Fermat (Eq.7) stays a valid permutation
        #      for several lambda_c, and lambda_c=0 reproduces pure greedy ----
        for lam in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for p, idx in enumerate(
                    get_scan_indices("mp_fermat", H, W, K=2, lambda_c=lam)):
                assert idx.numel() == N, \
                    f"mp_fermat lam={lam} path {p} {H}x{W}: wrong length"
                assert torch.equal(torch.unique(idx), torch.arange(N)), \
                    f"mp_fermat lam={lam} path {p} {H}x{W}: not a permutation"
        lam0 = get_scan_indices("mp_fermat", H, W, K=2, lambda_c=0.0)
        for p in range(2):
            assert torch.equal(lam0[p], mp2[p]), \
                f"lambda_c=0 must equal pure greedy (path {p}) at {H}x{W}"

        # ---- exact serialize -> deserialize round-trip ----
        x = torch.randn(2, 5, H, W)
        for name in single_scans:
            idx = get_scan_indices(name, H, W)[0]
            rt = deserialize(serialize(x, idx), idx, H, W)
            assert torch.allclose(rt, x), f"round-trip failed for {name} {H}x{W}"
        for p, idx in enumerate(mp2):  # each of the K Fermat paths
            rt = deserialize(serialize(x, idx), idx, H, W)
            assert torch.allclose(rt, x), \
                f"round-trip failed for mp_fermat path {p} {H}x{W}"

        # ---- MultiPathScan module round-trip (incl. bidirectional flips) ----
        mod = MultiPathScan(scan="mp_fermat", K=2, bidirectional=True)
        seqs = mod(x)
        assert len(seqs) == mod.num_sequences()
        for i, s in enumerate(seqs):
            rt = mod.deserialize(s, i)
            assert torch.allclose(rt, x), \
                f"MultiPathScan round-trip failed seq {i} at {H}x{W}"

        if verbose:
            print(f"[OK] all generators valid + exact round-trip at {H}x{W} "
                  f"(N={N})")

    # ---- anisotropy table for 64x64 (paper evidence) ----
    if verbose:
        H, W = 64, 64
        print("\nDirectional-anisotropy statistics @ 64x64 "
              "(lower anisotropy => more isotropic step directions):")
        print(f"  {'scan':<16}{'mean_step':>12}{'anisotropy':>12}")
        for name in single_scans + ["mp_fermat"]:
            paths = get_scan_indices(name, H, W, K=2)
            means, anis = [], []
            for idx in paths:
                m, a = _path_step_stats(idx, H, W)
                means.append(m)
                anis.append(a)
            tag = name + (" (avg K)" if name == "mp_fermat" else "")
            print(f"  {tag:<16}{np.mean(means):>12.4f}{np.mean(anis):>12.4f}")

    # ---- lambda_c isotropy<->locality trade-off sweep (paper section 3.3) ----
    if verbose:
        sweep_lambda_c(64, 64, K=2)

    if verbose:
        print("\nAll self-tests passed.")


if __name__ == "__main__":
    _run_self_test(verbose=True)
