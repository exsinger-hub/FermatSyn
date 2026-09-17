# FermatSyn

Official minimal release for the MICCAI version of FermatSyn: paired 2-D slice synthesis from 3-D medical volumes with SAM2-enhanced bidirectional Mamba and Fermat spiral scanning.

## Scope

The repository contains the runnable model, data-window loader, scan-path utilities, training/evaluation entry points, tests, dependency declaration, and the MICCAI paper source under `paper/`. Dataset files, checkpoints, generated results, and cluster launch scripts are intentionally excluded.

The current run identity is the dynamic full-candidate serializer with:

```text
--scan_use_multipath
--scan_name fermat
--scan_K 1
--scan_lambda_c 0.7
```

The executable graph includes reverse flip-back, exact inverse permutation, and position-wise aligned-mean fusion.

The paper source is `paper/paper-3777.tex`; compile it from that directory with a standard LaTeX environment.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The SAM2 checkpoint is not redistributed. Download it separately when the selected configuration requires it:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

## Run

Use `train_a2.py` for training and `evaluate_frozen_3tap.py` for evaluation. Both scripts expose `--help`; dataset paths and output directories must be supplied for a local dataset.

```bash
python train_a2.py --help
python evaluate_frozen_3tap.py --help
```

## Smoke check

The minimal import smoke check is:

```bash
python -m compileall -q .
python -c "from models.a2_context import InterSliceContextModulator; from models.a2_spatial import SUPPORTED_INPUT_MODES; print(sorted(SUPPORTED_INPUT_MODES))"
```

The checked-in tests include historical dataset/checkpoint cases and therefore require local medical data, SAM2 weights, and the optional dependencies listed in `requirements.txt`.

## License

The repository is released under the MIT License. See `LICENSE`.
