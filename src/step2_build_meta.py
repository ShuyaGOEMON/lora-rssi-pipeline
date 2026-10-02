"""Generate regenerated/meta/tx_config.csv and regenerated/meta/data_dictionary.csv.

Both files are small and fully determined by the confirmed specification, so
they are written from constants rather than derived from the logs. The row
count and the column list of the fingerprint file are cross-checked against
the generated raw/rssi_samples.csv.

Run from the repository root after step1_build_dataset.py:
    python src/step2_build_meta.py
"""
from __future__ import annotations

import pandas as pd

from config import OUTPUT_COLUMNS, OUT_DIR

# tx_config.csv : 4 rows x 14 columns
TX_CONFIG = [
    dict(tx_id=1, floor=3, x=36, y=-4, placement="open_space"),
    dict(tx_id=2, floor=4, x=0, y=10, placement="elevator_hall"),
    dict(tx_id=3, floor=7, x=36, y=2, placement="open_space"),
    dict(tx_id=4, floor=8, x=0, y=0, placement="elevator_hall"),
]
RADIO = dict(
    model="E220-900T22S(JP)",
    frequency_mhz=920.6,
    channel="ch0",
    tx_power_dbm=13,
    spreading_factor=7,
    bandwidth_khz=125,
    payload_bytes=200,
    tx_interval_s=2,
    antenna="DTH-RPLR-R1",
)
TX_CONFIG_COLUMNS = [
    "tx_id", "floor", "x", "y", "placement", "model", "frequency_mhz",
    "channel", "tx_power_dbm", "spreading_factor", "bandwidth_khz",
    "payload_bytes", "tx_interval_s", "antenna",
]

# data_dictionary.csv : one row per published column (12 rows)
DATA_DICTIONARY = [
    ("x", "integer", "m", "Horizontal coordinate of the reference point, common to all floors", ""),
    ("y", "integer", "m", "Horizontal coordinate of the reference point, common to all floors", ""),
    ("floor", "integer", "", "Floor label, 3 to 8", ""),
    ("sample_index", "integer", "", "Sequential sample number within a reference point, 1 to 50", ""),
    ("measurement_date", "string", "YYYY-MM-DD", "Date of measurement, taken from the name of the source log file", ""),
    ("session_id", "integer", "", "Measurement session identifier, 1 to 10; one source log file is one session", ""),
    ("elapsed_s", "integer", "s", "Time of the first packet received for this row, measured from the earliest record of the same session in raw_logs/; relative time only (no RTC, no NTP)", ""),
    ("rssi_tx1", "integer", "dBm", "Received power of the transmitter installed on the 3F", "empty cell"),
    ("rssi_tx2", "integer", "dBm", "Received power of the transmitter installed on the 4F", "empty cell"),
    ("rssi_tx3", "integer", "dBm", "Received power of the transmitter installed on the 7F", "empty cell"),
    ("rssi_tx4", "integer", "dBm", "Received power of the transmitter installed on the 8F", "empty cell"),
    ("receiver_config", "string", "", "Receiving antenna configuration: A = DTH-RPLR-R1 (sessions 1-7), B = DTH-RPLR-R2 (sessions 8-10)", ""),
]
DATA_DICTIONARY_COLUMNS = ["column", "type", "unit", "definition", "missing_representation"]


def build_tx_config() -> pd.DataFrame:
    df = pd.DataFrame([{**row, **RADIO} for row in TX_CONFIG])[TX_CONFIG_COLUMNS]
    assert df.shape == (4, 14), df.shape
    return df


def build_data_dictionary() -> pd.DataFrame:
    df = pd.DataFrame(DATA_DICTIONARY, columns=DATA_DICTIONARY_COLUMNS)
    assert list(df["column"]) == OUTPUT_COLUMNS, list(df["column"])
    return df


def main() -> None:
    meta_dir = OUT_DIR / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tx = build_tx_config()
    tx.to_csv(meta_dir / "tx_config.csv", index=False)

    dd = build_data_dictionary()
    dd.to_csv(meta_dir / "data_dictionary.csv", index=False)

    # the dictionary must describe exactly the columns of the published file
    fingerprints = OUT_DIR / "raw" / "rssi_samples.csv"
    if fingerprints.exists():
        header = list(pd.read_csv(fingerprints, nrows=0).columns)
        assert header == list(dd["column"]), header

    print(f"wrote {meta_dir/'tx_config.csv'}: {tx.shape[0]} rows x {tx.shape[1]} columns")
    print(f"wrote {meta_dir/'data_dictionary.csv'}: {dd.shape[0]} rows")


if __name__ == "__main__":
    main()