# lora-rssi-pipeline

Scripts that rebuild the files of the LoRa RSSI 3D indoor positioning fingerprint dataset from its raw session logs and verify the result.

The dataset itself is published separately on Zenodo: https://doi.org/10.5281/zenodo.22932995. This repository contains code only; no data are included.

## Repository layout

```
lora-rssi-pipeline/
├── raw_logs/                    # place the ten session logs here (not tracked)
├── regenerated/                 # output of the pipeline (not tracked)
├── src/
│   ├── config.py                # constants and session definitions (not run directly)
│   ├── step1_build_dataset.py   # raw logs -> raw/rssi_samples.csv
│   ├── step2_build_meta.py      # -> meta/tx_config.csv, meta/data_dictionary.csv
│   └── step3_verify_dataset.py  # checks against the confirmed statistics and the published file
├── LICENSE.txt
├── README.md
└── requirements.txt
```

## Requirements

- Python 3 with pandas 2.0 or later (`pandas.to_datetime(..., format="ISO8601")` is used).
- Tested only with Python 3.13.14 and pandas 3.0.6.

```
pip install -r requirements.txt
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
3. applies the transmitter-to-column assignment of the session (the logger recorded the four transmitters in reverse order in sessions 8-10; see `config.py`);
4. derives `elapsed_s` from the earliest record of the same session in `raw_logs/` and drops the raw timestamp, whose absolute date is unusable (no RTC, no NTP);
5. asserts that every reference point comes from exactly one session and checks the expected row and reference point counts;
6. writes the file sorted by floor, x, y and sample_index, with integer values (a missing RSSI is an empty cell) and CRLF line endings.

The values in `config.py` are taken from the specification of the published dataset. If a run fails, investigate the mismatch instead of editing the constants.

## License

The code is released under the MIT License, Copyright (c) 2026 Shuya Ishikawa; see `LICENSE.txt`. The dataset is distributed separately under CC BY 4.0.

## Citation

If you use the dataset, please cite:

Dataset paper: (submitted)
Dataset: Zenodo, https://doi.org/10.5281/zenodo.22932995