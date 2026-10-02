"""LoRa RSSI fingerprint dataset (Phase 2): pipeline configuration.

All values are taken from the confirmed dataset specification.
Do not edit them to make a run pass; investigate the mismatch instead.
"""
from __future__ import annotations

from pathlib import Path

# --- paths -----------------------------------------------------------------
RAW_DIR = Path("raw_logs")          # the ten retained session logs, as distributed
OUT_DIR = Path("regenerated")  # regenerated files: raw/rssi_samples.csv, meta/; never the downloaded dataset

# --- sessions --------------------------------------------------------------
# The order of this mapping defines session_id. One raw file = one session.
SESSIONS: dict[int, str] = {
    1: "20260515-2",
    2: "20260518-1",
    3: "20260518-2",
    4: "20260520-1",
    5: "20260522-2",
    6: "20260522-3",
    7: "20260522-4",
    8: "20260625-1",
    9: "20260625-2",
    10: "20260625-3",
}

# Only the ten logs listed in SESSIONS are part of the release; the pipeline
# reads them by name and uses no other file in raw_logs/.

# Rows contributed to the release per session. The superseded first
# measurements of the re-measured RPs are not in raw_logs/, so the wide table
# of each distributed log has exactly this many rows.
# Verified against raw_logs/ 2026-10-02.
EXPECTED_ROWS: dict[int, int] = {
    1: 1400, 2: 2300, 3: 700, 4: 2150, 5: 2650,
    6: 100, 7: 1850, 8: 800, 9: 450, 10: 1500,
}

# Records (one line per received packet) in each distributed log.
# Total 55,263 = 13,900 x 4 - 337 undetected. Verified 2026-10-02.
EXPECTED_LOG_RECORDS: dict[int, int] = {
    1: 5263, 2: 9200, 3: 2800, 4: 8600, 5: 10600,
    6: 400, 7: 7400, 8: 3200, 9: 1800, 10: 6000,
}
EXPECTED_TOTAL_ROWS = 13_900
EXPECTED_RP_COUNT = 278
SAMPLES_PER_RP = 50

# --- transmitter-to-column assignment --------------------------------------
# assignment[logger_tx_id - 1] = published column index.
# Sessions 8-10 (25 June 2026) were logged in reverse order: logger tx_id 1
# carried the transmitter installed on the 8F, 2 -> 7F, 3 -> 4F, 4 -> 3F.
# The cause was not determined; both a logging error and a physical exchange
# of the units remain possible (the transmitters were reinstalled daily).
# Confirmed against the raw logs 2026-09-25: at RP (0, 10, 5), directly above
# the 4F transmitter, the logger reports tx_id 3 at -54.4 dBm on average, and
# at the 8F RPs beside the 8F transmitter it reports tx_id 1 at -38.9 dBm.
# The same check leaves sessions 1-7 unchanged.
COLUMN_ASSIGNMENT: dict[int, tuple[int, int, int, int]] = {
    **{s: (1, 2, 3, 4) for s in range(1, 8)},
    8: (4, 3, 2, 1),
    9: (4, 3, 2, 1),
    10: (4, 3, 2, 1),
}

# Receiving antenna configuration: A = DTH-RPLR-R1, B = DTH-RPLR-R2.
RECEIVER_CONFIG: dict[int, str] = {s: ("A" if s <= 7 else "B") for s in SESSIONS}

# --- reference point curation ----------------------------------------------
# RPs measured twice in the original logs; all six were re-measured in
# session 10, which replaced the first measurement. The superseded records
# (998: 801 in 20260515-2, 197 in 20260520-1) were removed before the logs
# were distributed, so raw_logs/ holds only the later measurement and
# step1_build_dataset.py asserts that no RP occurs in more than one session.
# Listed for documentation only.
OVERWRITTEN_RPS = [
    (0, 10, 5), (0, 0, 8), (0, 2, 8), (0, 8, 8), (0, 10, 8), (2, 2, 8),
]

# There is no drop list, and none must be applied here. Verified 2026-09-25:
# across the ten original (pre-release) logs the only coordinates measured in
# more than one session are the six listed above, and the union of all
# measured RPs is exactly the 278 RPs of the release (49 per floor on 3F-6F
# and 8F, 33 on 7F).
# The "deleted coordinates" of the provenance notes therefore belong to the
# two excluded files, not to the retained sessions.

# --- raw log format --------------------------------------------------------
# Columns required from each log. The logger also writes rx_id (always 0) and
# originally building_id, a constant site code that is removed before the logs
# are distributed, so neither is required here.
RAW_COLUMNS = ["timestamp", "x", "y", "z", "tx_id", "rssi", "sample_index"]

# Verified 2026-09-25: tx_id takes the values 1..4 in every retained log.
TX_ID_BASE = 1

# x, y and z are written as integers in nine logs and as floats ("0.0") in
# 20260515-2, so they must be cast before they are used as RP keys.
COORD_COLUMNS = ["x", "y", "z"]

# Plausible RSSI range for the published columns, checked by step3_verify_dataset.py.
# The original 20260515-2 recorded rssi = -256 twice at RP (0, 0, 8) for the
# 8F transmitter. Those records belonged to the superseded first measurement
# of that RP and are not in raw_logs/ (verified 2026-10-02: no -256 in any
# distributed log). The check is kept as a guard.
RSSI_VALID_RANGE = (-140, 0)

# --- published schema ------------------------------------------------------
OUTPUT_COLUMNS = [
    "x", "y", "floor", "sample_index", "measurement_date", "session_id",
    "elapsed_s", "rssi_tx1", "rssi_tx2", "rssi_tx3", "rssi_tx4",
    "receiver_config",
]