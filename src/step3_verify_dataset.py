"""Verify the regenerated dataset against the published file and the confirmed statistics.

Run from the repository root after step2_build_meta.py:
    python src/step3_verify_dataset.py [path/to/published/rssi_samples.csv]

Without an argument, only the confirmed statistics are checked.

The comparison with the published file is done on the row multiset, not on the
file bytes, so a different row order does not count as a difference.
elapsed_s is compared separately, row by row on (floor, x, y, sample_index).
"""
from __future__ import annotations

import sys

import pandas as pd

from config import (
    EXPECTED_RP_COUNT, EXPECTED_TOTAL_ROWS, OUTPUT_COLUMNS, OUT_DIR,
    RSSI_VALID_RANGE,
)

TOL = 0.01  # dBm, statistics are published to two decimals

EXPECTED_MEAN = {"rssi_tx1": -82.68, "rssi_tx2": -77.05, "rssi_tx3": -73.65, "rssi_tx4": -76.84}
EXPECTED_SD = {"rssi_tx1": 19.58, "rssi_tx2": 16.36, "rssi_tx3": 18.98, "rssi_tx4": 19.80}
EXPECTED_MIN = {"rssi_tx1": -126, "rssi_tx2": -133, "rssi_tx3": -128, "rssi_tx4": -127}
EXPECTED_MAX = {"rssi_tx1": -4, "rssi_tx2": -16, "rssi_tx3": -15, "rssi_tx4": -22}
EXPECTED_MISSING_PCT = {"rssi_tx1": 1.76, "rssi_tx2": 0.67, "rssi_tx3": 0.00, "rssi_tx4": 0.00}
EXPECTED_DETECTED = {4: 13_563, 3: 337}
EXPECTED_FLOOR_MEAN = {
    3: (-52.9, -71.8, -78.5, -84.1),
    4: (-82.4, -62.4, -98.8, -97.6),
    5: (-82.5, -73.3, -78.7, -84.8),
    6: (-91.5, -80.6, -59.1, -77.4),
    7: (-94.7, -86.5, -51.2, -66.8),
    8: (-97.5, -91.3, -68.3, -47.1),
}
EXPECTED_MISSING_COUNT = {"rssi_tx1": 244, "rssi_tx2": 93, "rssi_tx3": 0, "rssi_tx4": 0}
# Standard deviation within a reference point, over its 50 samples.
EXPECTED_WITHIN_RP_SD = {
    "median": (1.27, 1.33, 0.97, 1.18),
    "mean": (2.01, 1.94, 1.64, 1.82),
    "p95": (6.03, 5.57, 4.59, 5.05),
    "max": (15.20, 8.54, 8.42, 7.84),
}
RSSI = ["rssi_tx1", "rssi_tx2", "rssi_tx3", "rssi_tx4"]

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok ' if ok else 'NG '}] {label}{'' if ok else '  ' + detail}")
    if not ok:
        failures.append(label)


def check_structure(df: pd.DataFrame) -> None:
    check("columns", list(df.columns) == OUTPUT_COLUMNS, str(list(df.columns)))
    check("row count", len(df) == EXPECTED_TOTAL_ROWS, str(len(df)))
    rp = df.groupby(["x", "y", "floor"]).size()
    check("reference points", len(rp) == EXPECTED_RP_COUNT, str(len(rp)))
    check("50 samples per RP", bool((rp == 50).all()))
    check("coordinate range",
          bool(df.x.between(0, 36).all() and df.y.between(-4, 10).all()
               and df.floor.between(3, 8).all()))
    check("one session per RP",
          int(df.groupby(["x", "y", "floor"])["session_id"].nunique().max()) == 1)
    # elapsed_s is measured from the earliest record of each session in raw_logs/
    check("elapsed_s starts at 0 in every session",
          bool((df.groupby("session_id")["elapsed_s"].min() == 0).all()))


def check_statistics(df: pd.DataFrame) -> None:
    for col in RSSI:
        s = df[col]
        check(f"{col} mean", abs(s.mean() - EXPECTED_MEAN[col]) < TOL, f"{s.mean():.2f}")
        check(f"{col} sd", abs(s.std() - EXPECTED_SD[col]) < TOL, f"{s.std():.2f}")
        check(f"{col} min/max",
              s.min() == EXPECTED_MIN[col] and s.max() == EXPECTED_MAX[col],
              f"{s.min()} / {s.max()}")
        pct = round(100 * s.isna().mean(), 2)
        check(f"{col} missing %", abs(pct - EXPECTED_MISSING_PCT[col]) < TOL, f"{pct}")
        check(f"{col} missing count",
              int(s.isna().sum()) == EXPECTED_MISSING_COUNT[col], str(int(s.isna().sum())))
        lo, hi = RSSI_VALID_RANGE
        vals = s.dropna()
        bad = vals[(vals < lo) | (vals > hi)]
        check(f"{col} within {lo}..{hi} dBm", bad.empty, f"{len(bad)} value(s) outside")

    check("missing values are on the 8F only",
          bool((df.loc[df[RSSI].isna().any(axis=1), "floor"] == 8).all()))

    detected = df[RSSI].notna().sum(axis=1).value_counts().to_dict()
    check("detected transmitters per row",
          {k: int(v) for k, v in detected.items()} == EXPECTED_DETECTED, str(detected))

    floor_mean = df.groupby("floor")[RSSI].mean().round(1)
    for floor, expected in EXPECTED_FLOOR_MEAN.items():
        got = tuple(floor_mean.loc[floor])
        check(f"floor {floor} mean RSSI",
              all(abs(a - b) < 0.05 for a, b in zip(got, expected)), str(got))

    sd = df.groupby(["x", "y", "floor"])[RSSI].std()
    got = {
        "median": tuple(sd.median()), "mean": tuple(sd.mean()),
        "p95": tuple(sd.quantile(0.95)), "max": tuple(sd.max()),
    }
    for stat, expected in EXPECTED_WITHIN_RP_SD.items():
        check(f"within-RP sd ({stat})",
              all(abs(a - b) < 0.01 for a, b in zip(got[stat], expected)),
              str(tuple(round(v, 2) for v in got[stat])))

    # every column must peak on the floor of its own transmitter
    peaks = floor_mean.idxmax().to_dict()
    check("peak floor equals installation floor",
          peaks == {"rssi_tx1": 3, "rssi_tx2": 4, "rssi_tx3": 7, "rssi_tx4": 8}, str(peaks))


def check_against_published(df: pd.DataFrame, published_path: str) -> None:
    pub = pd.read_csv(published_path)
    cols = [c for c in OUTPUT_COLUMNS if c != "elapsed_s"]  # compared separately below
    a = df[cols].sort_values(cols).reset_index(drop=True)
    b = pub[cols].sort_values(cols).reset_index(drop=True)
    check("row multiset equals published file", a.equals(b))
    if not a.equals(b):
        only_new = pd.concat([a, b, b]).drop_duplicates(keep=False)
        print(f"       rows only in the regenerated file: {len(only_new)}")

    key = ["floor", "x", "y", "sample_index"]
    m = df[key + ["elapsed_s"]].merge(
        pub[key + ["elapsed_s"]], on=key, how="inner", suffixes=("", "_pub"))
    n_diff = int((m["elapsed_s"] != m["elapsed_s_pub"]).sum())
    check("elapsed_s equals published file", len(m) == len(df) and n_diff == 0,
          f"matched rows {len(m)}, different elapsed_s {n_diff}")


def main(published_path: str | None = None) -> None:
    failures.clear()  # so that repeated calls of main() start clean
    df = pd.read_csv(OUT_DIR / "raw" / "rssi_samples.csv")
    check_structure(df)
    check_statistics(df)
    if published_path is not None:
        check_against_published(df, published_path)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)