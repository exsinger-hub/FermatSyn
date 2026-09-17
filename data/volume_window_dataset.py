"""Ordered SynthRAD volume windows for inter-slice A2 experiments."""

import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SLICE_NAME_RE = re.compile(
    r"^(?P<patient>.+)_(?P<slice_index>\d+)_concat\.npy$"
)


def parse_patient_slice(path):
    match = SLICE_NAME_RE.match(os.path.basename(path))
    if match is None:
        raise ValueError("unrecognized SynthRAD slice name: %s" % path)
    return match.group("patient"), int(match.group("slice_index"))


def _contiguous_runs(indexed_paths):
    runs = []
    current = []
    previous = None
    for slice_index, path in indexed_paths:
        if previous is None or slice_index == previous + 1:
            current.append((slice_index, path))
        else:
            if current:
                runs.append(current)
            current = [(slice_index, path)]
        previous = slice_index
    if current:
        runs.append(current)
    return runs


def _split_pair(array, load_size, direction):
    tensor = torch.from_numpy(array).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError("paired array must have 2 or 3 dimensions")

    width = tensor.shape[-1]
    if width < 512 or width % 256 != 0:
        raise ValueError("expected side-by-side 256-pixel images, got width=%d" % width)
    num_sources = width // 256 - 1
    source = tensor[:, :, : 256 * num_sources]
    if num_sources > 1:
        source = torch.cat(
            [source[:, :, i * 256 : (i + 1) * 256] for i in range(num_sources)],
            dim=0,
        )
    target = tensor[:, :, 256 * num_sources : 256 * (num_sources + 1)]

    source = F.interpolate(
        source.unsqueeze(0), size=(load_size, load_size), mode="bicubic", align_corners=True
    ).squeeze(0)
    target = F.interpolate(
        target.unsqueeze(0), size=(load_size, load_size), mode="bicubic", align_corners=True
    ).squeeze(0)
    if direction == "BtoA":
        source, target = target, source
    return source[:1], target[:1]


class SynthRADVolumeWindowDataset(Dataset):
    """Return fixed-length, consecutive, patient-local slice windows."""

    def __init__(
        self,
        dataroot,
        phase="train",
        window_size=5,
        stride=5,
        load_size=256,
        fine_size=256,
        direction="AtoB",
        augment=False,
        max_windows=0,
    ):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if stride < 1:
            raise ValueError("stride must be positive")
        if fine_size > load_size:
            raise ValueError("fine_size cannot exceed load_size")
        if direction not in {"AtoB", "BtoA"}:
            raise ValueError("direction must be AtoB or BtoA")

        self.phase_dir = os.path.join(os.path.abspath(dataroot), phase)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.load_size = int(load_size)
        self.fine_size = int(fine_size)
        self.direction = direction
        self.augment = bool(augment)
        self.max_windows = int(max_windows or 0)

        paths = [
            os.path.join(self.phase_dir, name)
            for name in sorted(os.listdir(self.phase_dir))
            if name.endswith("_concat.npy")
        ]
        if not paths:
            raise RuntimeError("no paired .npy slices found in %s" % self.phase_dir)

        grouped = defaultdict(list)
        for path in paths:
            patient, slice_index = parse_patient_slice(path)
            grouped[patient].append((slice_index, path))

        all_windows = []
        for patient in sorted(grouped):
            indexed = sorted(grouped[patient])
            indices = [item[0] for item in indexed]
            if len(indices) != len(set(indices)):
                raise RuntimeError("duplicate slice index for patient %s" % patient)
            for run in _contiguous_runs(indexed):
                if len(run) < self.window_size:
                    continue
                starts = range(0, len(run) - self.window_size + 1)
                for start in starts:
                    chunk = run[start : start + self.window_size]
                    all_windows.append(
                        {
                            "patient_id": patient,
                            "slice_indices": [item[0] for item in chunk],
                            "paths": [item[1] for item in chunk],
                            "stride_offset": start % self.stride,
                        }
                    )
        if not all_windows:
            raise RuntimeError("no consecutive windows could be constructed")
        self.all_windows = all_windows
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Rotate the supervised centre-slice subset when stride is greater than one."""

        offset = int(epoch) % self.stride
        windows = [
            record
            for record in self.all_windows
            if record["stride_offset"] == offset
        ]
        if self.max_windows > 0:
            windows = windows[: self.max_windows]
        if not windows:
            raise RuntimeError("epoch offset produced no windows")
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        record = self.windows[index]
        sources, targets = [], []
        for path in record["paths"]:
            source, target = _split_pair(np.load(path), self.load_size, self.direction)
            sources.append(source)
            targets.append(target)
        source = torch.stack(sources, dim=0)
        target = torch.stack(targets, dim=0)

        if self.fine_size < self.load_size:
            max_offset = self.load_size - self.fine_size
            top = random.randint(0, max_offset)
            left = random.randint(0, max_offset)
            source = source[:, :, top : top + self.fine_size, left : left + self.fine_size]
            target = target[:, :, top : top + self.fine_size, left : left + self.fine_size]
        if self.augment and random.random() < 0.5:
            source = torch.flip(source, dims=[-1])
            target = torch.flip(target, dims=[-1])

        return {
            "A": source,
            "B": target,
            "A_center": source[self.window_size // 2],
            "B_center": target[self.window_size // 2],
            "patient_id": record["patient_id"],
            "slice_indices": torch.tensor(record["slice_indices"], dtype=torch.long),
            "center_slice_index": int(record["slice_indices"][self.window_size // 2]),
            "paths": record["paths"],
        }
