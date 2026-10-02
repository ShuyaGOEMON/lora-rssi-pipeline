"""raw_logs/*.csv (long) -> regenerated/raw/rssi_samples.csv (wide, 13,900 x 12).

Processing steps, in order:
  1. read each distributed session log, check its record count and attach
     session_id / measurement_date
  2. pivot long -> wide (one row per sample, one column per transmitter)
  3. apply the transmitter-to-column assignment of the session
  4. derive elapsed_s and drop the unreliable raw timestamp
  5. assert that every RP comes from exactly one session (the superseded
     first measurements of the six re-measured RPs are not in raw_logs/)
  6. assert the confirmed row / RP counts, then write (integers, CRLF)

Run from the repository root (paths are relative to the working directory):
    python src/step1_build_dataset.py
"""
from __future__ import annotations

import pandas as pd

from config import (
    COLUMN_ASSIGNMENT, COORD_COLUMNS, EXPECTED_LOG_RECORDS, EXPECTED_ROWS,
    EXPECTED_RP_COUNT, EXPECTED_TOTAL_ROWS, OUTPUT_COLUMNS, OUT_DIR,
    RAW_COLUMNS, RAW_DIR, RECEIVER_CONFIG, SAMPLES_PER_RP, SESSIONS,
    TX_ID_BASE,
)

RP = ["x", "y", "floor"]
RSSI_COLUMNS = [f"rssi_tx{i}" for i in (1, 2, 3, 4)]
KEY = ["measurement_date", "session_id", "x", "y", "floor", "sample_index"]


def load_session(session_id: int, stem: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{stem}.csv")
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{stem}: missing raw columns {missing}")
    if len(df) != EXPECTED_LOG_RECORDS[session_id]:
        raise ValueError(
            f"{stem}: {len(df)} records, expected {EXPECTED_LOG_RECORDS[session_id]}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["tx_id"] = df["tx_id"].astype(int) - TX_ID_BASE + 1  # normalise to 1..4
    # 20260515-2 writes the coordinates as floats ("0.0"); cast them so that
    # the RP keys of every session are comparable.
    for col in COORD_COLUMNS:
        df[col] = df[col].astype(float).round().astype(int)
    df["floor"] = df["z"]
    df["session_id"] = session_id
    df["measurement_date"] = f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}"
    return df


def to_wide(df: pd.DataFrame, session_id: int) -> pd.DataFrame:
    dup = int(df.duplicated(subset=KEY + ["tx_id"]).sum())
    if dup:
        raise ValueError(f"session {session_id}: {dup} duplicated (RP, sample_index, tx_id)")

    rssi = df.pivot(index=KEY, columns="tx_id", values="rssi")

    # Transmitter-to-column assignment of this session. For sessions 8-10 the
    # four columns were logged in reverse order and are reassigned here.
    assignment = COLUMN_ASSIGNMENT[session_id]
    rssi = rssi.rename(columns={logger_id: assignment[logger_id - 1] for logger_id in (1, 2, 3, 4)})
    rssi = rssi.reindex(columns=[1, 2, 3, 4])
    rssi.columns = [f"rssi_tx{i}" for i in (1, 2, 3, 4)]

    # elapsed_s: time of the first packet received for that row, measured from
    # the earliest record of the same session in raw_logs/, so that it can be
    # recomputed from the distributed logs alone. In session 1 the first two
    # RPs visited were superseded, so its reference is the start of the third
    # RP, not the start of the session. The absolute date of the logger is
    # unusable (no RTC, no NTP), so only the relative time is kept.
    first = df.groupby(KEY)["timestamp"].min()
    elapsed = (first - first.min()).dt.total_seconds().round(3).rename("elapsed_s")

    out = rssi.join(elapsed).reset_index()
    out["receiver_config"] = RECEIVER_CONFIG[session_id]
    return out


def check(df: pd.DataFrame) -> None:
    assert list(df.columns) == OUTPUT_COLUMNS, list(df.columns)
    assert len(df) == EXPECTED_TOTAL_ROWS, len(df)
    counts = df.groupby(RP).size()
    assert len(counts) == EXPECTED_RP_COUNT, len(counts)
    assert (counts == SAMPLES_PER_RP).all(), counts[counts != SAMPLES_PER_RP]
    per_session = df.groupby("session_id").size().to_dict()
    assert per_session == EXPECTED_ROWS, per_session
    idx = df.groupby(RP)["sample_index"].agg(["min", "max"])
    assert (idx["min"] == 1).all() and (idx["max"] == SAMPLES_PER_RP).all()


def main() -> None:
    frames = [to_wide(load_session(sid, stem), sid) for sid, stem in SESSIONS.items()]
    df = pd.concat(frames, ignore_index=True)

    # No repeat resolution is needed or applied: raw_logs/ holds only the
    # later measurement of the six re-measured RPs.
    repeats = df.groupby(RP)["session_id"].nunique()
    assert (repeats > 1).sum() == 0, repeats[repeats > 1]

    df = df.sort_values(["floor", "x", "y", "sample_index"]).reset_index(drop=True)
    df = df[OUTPUT_COLUMNS]
    check(df)

    # Output format of the published file: the logger timestamps have a
    # resolution of 1 s and RSSI is reported in whole dBm, so both are written
    # as integers (nullable Int64, a missing RSSI stays an empty cell, no
    # sentinel); lines end in CRLF, as in raw_logs/.
    assert (df["elapsed_s"] % 1 == 0).all(), "elapsed_s is not integral"
    df["elapsed_s"] = df["elapsed_s"].astype("int64")
    df[RSSI_COLUMNS] = df[RSSI_COLUMNS].astype("Int64")

    path = OUT_DIR / "raw" / "rssi_samples.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\r\n")
    print(f"wrote {path}: {len(df)} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()