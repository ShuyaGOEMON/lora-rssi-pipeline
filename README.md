# lora-rssi-pipeline

Scripts that rebuild the files of the LoRa RSSI 3D indoor positioning fingerprint dataset from its raw session logs and verify the result. It also contains the scripts of the baseline evaluation reported in the dataset paper (`evaluation/`).

The dataset itself is published separately on Zenodo: https://doi.org/10.5281/zenodo.22932995. This repository contains code only; no data are included.

## Repository layout

```
lora-rssi-pipeline/
├── evaluation/                  # baseline evaluation of the paper (Table 4, Fig. 3)
│   ├── known_rp_rf_floor_xy.py
│   ├── known_rp_mlp_floor_xy.py
│   ├── known_rp_rf_rp_classification.py
│   ├── known_rp_mlp_rp_classification.py
│   ├── unseen_rp_rf_floor_xy.py
│   ├── unseen_rp_mlp_floor_xy.py
│   └── requirements.txt         # requirements of the evaluation scripts
├── raw_logs/                    # place the ten session logs here (not tracked)
├── regenerated/                 # output of the pipeline (not tracked)
├── src/
│   ├── config.py                # constants and session definitions (not run directly)
│   ├── step1_build_dataset.py   # raw logs -> raw/rssi_samples.csv
│   ├── step2_build_meta.py      # -> meta/tx_config.csv, meta/data_dictionary.csv
│   ├── step3_verify_dataset.py  # checks against the confirmed statistics and the published file
│   └── requirements.txt         # requirements of the pipeline
├── LICENSE
└── README.md
```

## Requirements

- The pipeline (`src/`) needs Python 3 with pandas 2.0 or later (`pandas.to_datetime(..., format="ISO8601")` is used).
- Tested only with Python 3.13.14 and pandas 3.0.6.
- The evaluation scripts (`evaluation/`) have their own requirements file, `evaluation/requirements.txt`; see Baseline evaluation below.

```
pip install -r src/requirements.txt
```

## Getting the data

1. Download the zip archive from the Zenodo record of the dataset (https://doi.org/10.5281/zenodo.22932995) and extract it. Its top-level folder is `LoRaRSSIFingerprint/`.
2. Copy the ten CSV files in `LoRaRSSIFingerprint/raw_logs/` directly into `raw_logs/` of this repository:

```
20260515-2.csv  20260518-1.csv  20260518-2.csv  20260520-1.csv  20260522-2.csv
20260522-3.csv  20260522-4.csv  20260625-1.csv  20260625-2.csv  20260625-3.csv
```

The pipeline reads exactly these ten files by name and ignores any other file in `raw_logs/`.

## Usage

Run every command from the repository root; all paths are relative to the working directory.

```
python src/step1_build_dataset.py
python src/step2_build_meta.py
python src/step3_verify_dataset.py path/to/LoRaRSSIFingerprint/raw/rssi_samples.csv
```

The argument of step 3 is optional. Without it, only the confirmed statistics are checked; with it, the regenerated file is also compared with the published file you downloaded. Step 3 prints one line per check and exits with status 1 if any check fails.

The outputs are written to `regenerated/`, never to the downloaded dataset:

```
regenerated/
├── raw/rssi_samples.csv
└── meta/
    ├── tx_config.csv
    └── data_dictionary.csv
```

## What the pipeline does

| Step | Input | Output |
|---|---|---|
| 1 | 10 session logs, 55,263 records (one per received packet) | `rssi_samples.csv`: 13,900 rows x 12 columns (278 reference points x 50 samples) |
| 2 | constants of the dataset specification | `tx_config.csv`: 4 rows x 14 columns; `data_dictionary.csv`: 12 rows |
| 3 | regenerated files (and optionally the published file) | check report |

Step 1, in order:

1. reads each session log and checks its record count;
2. pivots the log from long (one row per packet) to wide (one row per sample, one column per transmitter);
3. applies the transmitter-to-column assignment of the session (in sessions 8-10 the four transmitter units were installed at each other's positions by operator error, which reversed the assignment in the log; see `config.py`);
4. derives `elapsed_s` from the earliest record of the same session in `raw_logs/` and drops the raw timestamp, whose absolute date is unusable (no RTC, no NTP);
5. asserts that every reference point comes from exactly one session and checks the expected row and reference point counts;
6. writes the file sorted by floor, x, y and sample_index, with integer values (a missing RSSI is an empty cell) and CRLF line endings.

The values in `config.py` are taken from the specification of the published dataset. If a run fails, investigate the mismatch instead of editing the constants.

## Baseline evaluation (`evaluation/`)

The six scripts in `evaluation/` reproduce the baseline results of the dataset paper (Table 4 and Fig. 3). They read the published `raw/rssi_samples.csv` (MD5 4ede8e1a4cab309fa01470e3225f8b39) directly and do not use the output of the pipeline. All use 5-fold cross-validation.

| Script | Split | Model | Target |
|---|---|---|---|
| `known_rp_rf_floor_xy.py` | known RPs: StratifiedKFold by floor | RF | floor and (x, y) |
| `known_rp_mlp_floor_xy.py` | known RPs: StratifiedKFold by floor | MLP | floor and (x, y) |
| `known_rp_rf_rp_classification.py` | known RPs: StratifiedKFold by RP | RF | RP |
| `known_rp_mlp_rp_classification.py` | known RPs: StratifiedKFold by RP | MLP | RP |
| `unseen_rp_rf_floor_xy.py` | unseen RPs: GroupKFold, groups = RP key (floor, x, y) | RF | floor and (x, y) |
| `unseen_rp_mlp_floor_xy.py` | unseen RPs: GroupKFold, groups = RP key (floor, x, y) | MLP | floor and (x, y) |

Requirements: numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 (the versions used for the paper) and matplotlib. Install them from the repository root:

```
pip install -r evaluation/requirements.txt
```

The versions are pinned because the GroupKFold fold assignment and the random forest results may change with the scikit-learn version. The MLP scripts also need PyTorch (install the build for your platform and CUDA version) and use a CUDA GPU if available. Tested only with Python 3.13.15, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 and torch 2.11.0+cu130.

Usage (example):

```
python evaluation/known_rp_rf_floor_xy.py --csv path/to/LoRaRSSIFingerprint/raw/rssi_samples.csv --seed 42
```

`--csv` is required. `--seed` defaults to 42, the value reported in the paper; for the unseen-RP scripts, the paper also reports the range over the seeds 42, 0, 1, 2 and 3.

The scores are printed to standard output. The files are written to the working directory:

- `<script>_folds.csv`: fold assignment (folds 1-5);
- `heatmap_<script>*.png`: heatmaps of the results;
- `unseen_rp_rf_floor_xy.py` only: `unseen_rp_rf_floor_xy_rp_metrics.csv` (per-RP metrics) and `figure3.png` / `figure3.pdf` (Fig. 3).

Notes on reproducibility:

- In the known-RP scripts, `--seed` also changes the folds. In the unseen-RP scripts, the folds do not depend on `--seed` but depend on the group strings and the scikit-learn version.
- GPU results may not be bit-identical across hardware and library versions.

## License

The code is released under the MIT License, Copyright (c) 2026 Shuya Ishikawa; see `LICENSE`. The dataset is distributed separately under CC BY 4.0.

## Citation

If you use the dataset, please cite:

Dataset paper: [TBD]
Dataset: Zenodo, https://doi.org/10.5281/zenodo.22932995