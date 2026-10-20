"""Unseen RPs: random-forest floor classification and horizontal (x, y) regression.

This script also draws Fig. 3 of the paper.

Baseline evaluation for Table 4 of the paper.

Split : GroupKFold (5 folds, no shuffling); groups = RP key (floor, x, y).
        Same folds as unseen_rp_mlp_floor_xy.py.
Model : RandomForestClassifier (floor) + RandomForestRegressor (x, y), n_estimators=100,
        other parameters at the scikit-learn defaults.
Input : raw/rssi_samples.csv of the dataset v1.0.1 (MD5 4ede8e1a4cab309fa01470e3225f8b39).
        Features rssi_tx1 to rssi_tx4; missing values (no packet received) are filled with -140.

Usage:
    python unseen_rp_rf_floor_xy.py --csv path/to/raw/rssi_samples.csv [--seed 42]

Outputs (current directory; the metrics are printed to standard output):
    unseen_rp_rf_floor_xy_folds.csv  test fold (1-5) of every RP (columns: floor, x, y, fold)
    figure3.png, figure3.pdf             Fig. 3 (per-RP floor accuracy); the PNG is used in the paper
    unseen_rp_rf_floor_xy_rp_metrics.csv per-RP values plotted in Fig. 3
    heatmap_unseen_rp_rf_floor_xy_*.png  per-RP maps of the three metrics

The folds do not depend on --seed. --seed sets random_state of both random
forests. The paper reports --seed 42 and the range over the seeds 42, 0, 1, 2, 3.
The fold assignment depends on the group strings and the scikit-learn version.

Environment used for the paper: Python 3.13.15, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 0. Settings
# ============================================================
# Command-line arguments
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--csv', required=True, help='path to raw/rssi_samples.csv of the dataset (v1.0.1)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42, used in the paper)')
args = parser.parse_args()
CSV_PATH = args.csv
EXPECTED_MD5 = '4ede8e1a4cab309fa01470e3225f8b39'  # raw/rssi_samples.csv of dataset v1.0.1
FLOOR_COL = 'floor'  # floor column of the published CSV
FEATURE_COLS = ['rssi_tx1', 'rssi_tx2', 'rssi_tx3', 'rssi_tx4']  # features
N_SPLITS = 5
RANDOM_STATE = args.seed
FLOOR_HEIGHT_M = 3.8  # floor height (m, measured)

# ============================================================
# 1. Load and check the data
# ============================================================
# Record the execution environment
import sys, sklearn
print(f"Python {sys.version.split()[0]} / numpy {np.__version__} / pandas {pd.__version__}"
      f" / scikit-learn {sklearn.__version__}")

df = pd.read_csv(CSV_PATH)

# Check the input file (MD5 and required columns)
import hashlib
with open(CSV_PATH, 'rb') as f:
    csv_md5 = hashlib.md5(f.read()).hexdigest()
print(f"Input CSV: {CSV_PATH}  MD5: {csv_md5}"
      + ("" if csv_md5 == EXPECTED_MD5 else "  WARNING: does not match the MD5 of v1.0.1"))
missing_cols = [c for c in [FLOOR_COL, 'x', 'y'] + FEATURE_COLS if c not in df.columns]
assert not missing_cols, f"Missing required columns: {missing_cols}"

feature_cols = FEATURE_COLS
X = df[feature_cols]

# --- Missing values ---
print("=" * 60)
print("Missing values")
print("=" * 60)
null_counts = X.isnull().sum()
if null_counts.sum() == 0:
    print("No missing values - SimpleImputer has no effect")
else:
    print("Missing values found - packets not received (NaN) are filled with -140")  # NaN = not received
    print(null_counts[null_counts > 0])
    print(f"Missing rate: {X.isnull().sum().sum() / X.size * 100:.2f}%")
print()

# --- Targets ---
y_z = df[FLOOR_COL].astype(int)           # floor (classification)
y_xy = df[['x', 'y']].astype(float) # horizontal coordinates (regression)

# --- Group label = RP key (floor, x, y) ---
# All samples of one RP are placed in the same fold
groups = df[FLOOR_COL].astype(str) + '_' + df['x'].astype(str) + '_' + df['y'].astype(str)

floors = sorted(y_z.unique())
n_rps = groups.nunique()
print(f"Samples          : {len(X)}")
print(f"Features         : {X.shape[1]}")
print(f"Floors           : {len(floors)}  {floors}")
print(f"RPs              : {n_rps}")
print(f"Samples per RP   : {len(X) // n_rps}")
print()

# ============================================================
# 2. Build the pipelines
# ============================================================
# Floor (z): random forest classifier
clf_z = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-140)),
    ('model', RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ))
])

# Coordinates (x, y): random forest regressor
reg_xy = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-140)),
    ('model', RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ))
])

# ============================================================
# 3. Cross-validation (GroupKFold: split by RP)
# ============================================================
# No shuffling. The fold assignment is determined by the group strings and the scikit-learn version.
# unseen_rp_rf_floor_xy.py and unseen_rp_mlp_floor_xy.py build the groups in the same way and use the same folds.
gkf = GroupKFold(n_splits=N_SPLITS)

# --- Results ---
fold_z_acc = []
fold_xy_err_all = []
fold_xy_err_correct = []
all_z_true = []
all_z_pred = []
all_distances_correct = []
all_distances_wrong = []
all_floor_errors = []
all_3d_errors = []

# --- Per-sample values for the per-RP maps ---
rp_true_x = []
rp_true_y = []
rp_true_z = []
rp_correct = []
rp_horiz = []
rp_3d = []

print("=" * 60)
print("Cross-validation (GroupKFold: split by RP)")
print("=" * 60)

fold_test_idx = []  # (fold, test row indices) for the fold assignment file
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_z, groups=groups), 1):
    fold_test_idx.append((fold, test_idx))
    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_z_train, y_z_test = y_z.iloc[train_idx], y_z.iloc[test_idx]
    y_xy_train, y_xy_test = y_xy.iloc[train_idx], y_xy.iloc[test_idx]

    # Number of RPs in the fold
    train_rps = groups.iloc[train_idx].nunique()
    test_rps = groups.iloc[test_idx].nunique()

    # ---- (1) Floor classification ----
    clf_z.fit(X_train, y_z_train)
    z_pred = clf_z.predict(X_test)

    z_acc = accuracy_score(y_z_test, z_pred)
    fold_z_acc.append(z_acc)
    all_z_true.extend(y_z_test.tolist())
    all_z_pred.extend(z_pred.tolist())

    # Absolute floor difference
    floor_err = np.abs(y_z_test.values - z_pred)
    all_floor_errors.extend(floor_err.tolist())

    # ---- (2) Horizontal regression ----
    reg_xy.fit(X_train, y_xy_train)
    xy_pred = reg_xy.predict(X_test)

    # Horizontal error (Euclidean distance)
    distances = np.sqrt(np.sum((xy_pred - y_xy_test.values) ** 2, axis=1))

    # Split by correct / wrong floor
    correct_mask = (y_z_test.values == z_pred)
    wrong_mask = ~correct_mask

    dist_correct = distances[correct_mask]
    dist_wrong = distances[wrong_mask]

    all_distances_correct.extend(dist_correct.tolist())
    all_distances_wrong.extend(dist_wrong.tolist())

    # 3D error (auxiliary metric)
    z_diff_m = np.abs(y_z_test.values - z_pred) * FLOOR_HEIGHT_M
    dist_3d = np.sqrt(distances ** 2 + z_diff_m ** 2)
    all_3d_errors.extend(dist_3d.tolist())

    # ---- Store values for the per-RP maps ----
    rp_true_x.extend(y_xy_test.values[:, 0].tolist())
    rp_true_y.extend(y_xy_test.values[:, 1].tolist())
    rp_true_z.extend(y_z_test.values.tolist())
    rp_correct.extend((y_z_test.values == z_pred).astype(int).tolist())
    rp_horiz.extend(distances.tolist())
    rp_3d.extend(dist_3d.tolist())

    # Fold results
    mean_xy_all = np.mean(distances)
    mean_xy_correct = np.mean(dist_correct) if len(dist_correct) > 0 else float('nan')
    fold_xy_err_all.append(mean_xy_all)
    fold_xy_err_correct.append(mean_xy_correct)

    print(f"Fold {fold} | train RPs: {train_rps} test RPs: {test_rps}"
          f" | floor accuracy: {z_acc:.4f}"
          f" | horizontal error (all): {mean_xy_all:.2f}m"
          f" | horizontal error (correct floor): {mean_xy_correct:.2f}m"
          f" | wrong floor: {wrong_mask.sum()}/{len(z_pred)}")

# ============================================================
# Fold assignment (test fold of every RP)
# ============================================================
FOLD_CSV = 'unseen_rp_rf_floor_xy_folds.csv'
fold_df = pd.concat(
    [df.iloc[test_idx][[FLOOR_COL, 'x', 'y']].drop_duplicates().assign(fold=fold)
     for fold, test_idx in fold_test_idx],
    ignore_index=True,
).sort_values(['fold', FLOOR_COL, 'x', 'y'])
assert len(fold_df) == len(fold_df[[FLOOR_COL, 'x', 'y']].drop_duplicates()) == len(np.unique(groups)), \
    "Each RP must be in exactly one test fold"
fold_df.to_csv(FOLD_CSV, index=False)
print(f'Saved: {FOLD_CSV}')

# ============================================================
# 4. Overall results
# ============================================================
print()
print("=" * 60)
print("Overall results")
print("=" * 60)

# ---- Floor classification ----
print("\n[Floor classification]")
print(f"  Mean accuracy : {np.mean(fold_z_acc):.4f} (SD: {np.std(fold_z_acc):.4f})")

all_z_true = np.array(all_z_true)
all_z_pred = np.array(all_z_pred)
all_floor_errors = np.array(all_floor_errors)

print(f"\n  Floor error distribution:")
for diff_val in sorted(np.unique(all_floor_errors)):
    count = np.sum(all_floor_errors == diff_val)
    pct = count / len(all_floor_errors) * 100
    label = "correct" if diff_val == 0 else f"off by {int(diff_val)} floor(s)"
    print(f"    {label}: {count}/{len(all_floor_errors)} ({pct:.1f}%)")

print(f"\n  Confusion matrix (rows = true, columns = predicted):")
cm = confusion_matrix(all_z_true, all_z_pred, labels=floors)
cm_df = pd.DataFrame(cm, index=[f"{f}F" for f in floors], columns=[f"{f}F" for f in floors])
print(cm_df.to_string())

print(f"\n  Classification report by floor:")
print(classification_report(
    all_z_true, all_z_pred,
    labels=floors,
    target_names=[f"{f}F" for f in floors]
))

# ---- Horizontal error (correct floor only) ----
print("[Horizontal error (correct floor only)]")
dc = np.array(all_distances_correct)
print(f"  Samples    : {len(dc)}")
print(f"  Mean       : {np.mean(dc):.2f} m")
print(f"  Median     : {np.median(dc):.2f} m")
print(f"  75%ile     : {np.percentile(dc, 75):.2f} m")
print(f"  90%ile     : {np.percentile(dc, 90):.2f} m")
print(f"  Max        : {np.max(dc):.2f} m")

# ---- Horizontal error (wrong floor, for reference) ----
if len(all_distances_wrong) > 0:
    print("\n[Horizontal error (wrong floor, for reference)]")
    dw = np.array(all_distances_wrong)
    print(f"  Samples    : {len(dw)}")
    print(f"  Mean       : {np.mean(dw):.2f} m")
    print(f"  Median     : {np.median(dw):.2f} m")

# ---- 3D error (auxiliary metric) ----
print(f"\n[3D error (auxiliary metric, floor height = {FLOOR_HEIGHT_M} m)]")
d3 = np.array(all_3d_errors)
print(f"  Mean       : {np.mean(d3):.2f} m")
print(f"  Median     : {np.median(d3):.2f} m")
print(f"  75%ile     : {np.percentile(d3, 75):.2f} m")
print(f"  90%ile     : {np.percentile(d3, 90):.2f} m")

# ============================================================
# 5. Per-RP maps
# ============================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D  # legend of the transmitter marker

mpl.rcParams.update({  # sans-serif font; embed TrueType fonts in the PDF
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

FILE_PREFIX = 'heatmap_unseen_rp_rf_floor_xy'  # prefix of the per-metric PNG files
EVAL_LABEL = 'unseen RPs, RF, GroupKFold'  # title of the per-metric PNG files
PAPER_FIG = 'figure3.png'           # Fig. 3 of the paper
PAPER_FIG_PDF = 'figure3.pdf'       # vector version of Fig. 3 (not used in the paper)
RP_CSV = 'unseen_rp_rf_floor_xy_rp_metrics.csv'  # per-RP values plotted in Fig. 3
# Transmitter positions (x, y, floor); same values as meta/tx_config.csv
TX_POS = {'Tx1': (36, -4, 3), 'Tx2': (0, 10, 4), 'Tx3': (36, 2, 7), 'Tx4': (0, 0, 8)}
CELL = 1.8  # side of one cell (m); smaller than the 2 m RP spacing so that cell borders are visible
# Transmitter marker style (shared by the plot and the legend) and legend text
TX_STYLE = dict(marker='^', ms=4, mfc='none', mec='red', mew=0.9, linestyle='none')
TX_LABEL = 'Transmitter position (shown on its installation floor)'
tx_handle = Line2D([], [], label=TX_LABEL, **TX_STYLE)

# Aggregate the test samples by (z, x, y) and average per RP
rp_df = pd.DataFrame({
    'x': rp_true_x, 'y': rp_true_y, 'z': rp_true_z,
    'correct': rp_correct, 'horiz': rp_horiz, 'err3d': rp_3d,
})

rp_agg = rp_df.groupby(['z', 'x', 'y']).agg(
    accuracy=('correct', 'mean'),   # floor accuracy per RP
    horiz_err=('horiz', 'mean'),    # mean horizontal error per RP
    err_3d=('err3d', 'mean'),       # mean 3D error per RP
    n=('correct', 'size'),
).reset_index()
assert len(rp_agg) == 278 and (rp_agg['n'] == 50).all(), "Unexpected number of RPs or samples per RP"
rp_agg.rename(columns={'z': 'floor'}).to_csv(RP_CSV, index=False)
print(f'Saved: {RP_CSV}')

def draw_floor(ax, sub, col, cmap, norm, fl):
    """Draw the RPs of one floor as cells on the 2 m grid (coordinates in m)"""
    cells = [Rectangle((x - CELL / 2, y - CELL / 2), CELL, CELL)
             for x, y in zip(sub['x'], sub['y'])]
    pc = PatchCollection(cells, cmap=cmap, norm=norm, edgecolor='0.6', linewidth=0.3)
    pc.set_array(sub[col].to_numpy())
    ax.add_collection(pc)
    for name, (tx, ty, tf) in TX_POS.items():  # transmitters only on their installation floor (open red triangles)
        if tf == fl:
            ax.plot(tx, ty, **TX_STYLE)
    ax.set_xlim(-2, 38)
    ax.set_ylim(-6, 13)
    ax.set_aspect('equal')
    return pc

# (metric column, label, colormap, vmin, vmax); None = automatic scale
metrics = [
    ('accuracy', 'Floor classification accuracy', 'viridis', 0.0, 1.0),
    ('horiz_err', 'Horizontal error (m)', 'hot_r', None, None),
    ('err_3d', '3D error (m)', 'hot_r', None, None),
]

floors_sorted = sorted(rp_agg['z'].unique())

# ---- Per-metric PNG files (3 metrics, 6 floors in one row) ----
for metric_col, metric_label, cmap, vmin, vmax in metrics:
    vmin_use = rp_agg[metric_col].min() if vmin is None else vmin
    vmax_use = rp_agg[metric_col].max() if vmax is None else vmax
    norm = Normalize(vmin=vmin_use, vmax=vmax_use)

    n_fl = len(floors_sorted)
    fig, axes = plt.subplots(1, n_fl, figsize=(3.2 * n_fl, 2.4), squeeze=False)
    pc = None
    for ax, fl in zip(axes[0], floors_sorted):
        sub = rp_agg[rp_agg['z'] == fl]
        pc = draw_floor(ax, sub, metric_col, cmap, norm, fl)
        ax.set_title(f'Floor {fl}')
        ax.set_xlabel('x (m)')
    axes[0][0].set_ylabel('y (m)')  # y label on the leftmost panel only
    fig.colorbar(pc, ax=list(axes[0]), shrink=0.8, label=metric_label)
    axes[0][0].legend(handles=[tx_handle], loc='lower left', bbox_to_anchor=(0, 1.32),
                      frameon=False, borderaxespad=0)  # transmitter legend above the leftmost panel
    fig.suptitle(f'Per-RP {metric_label.lower()} ({EVAL_LABEL})')
    out_path = f'{FILE_PREFIX}_{metric_col}.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out_path}  (RPs: {len(rp_agg)})')

# ---- Fig. 3 of the paper (floor accuracy only, full two-column width, 3 columns x 2 rows, no title) ----
with mpl.rc_context({'font.size': 7}):
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 2.2), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0.06, 'hspace': 0.30})
    pc = None
    for ax, fl in zip(axes.flat, floors_sorted):
        sub = rp_agg[rp_agg['z'] == fl]
        pc = draw_floor(ax, sub, 'accuracy', 'viridis', norm, fl)
        ax.set_title(f'Floor {fl}', pad=2)
        ax.set_xticks(range(0, 37, 6))
        ax.set_yticks([-4, 0, 4, 8])
        ax.tick_params(length=2, pad=1.5)
    for ax in axes[1]:
        ax.set_xlabel('x (m)', labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel('y (m)', labelpad=1)
    cb = fig.colorbar(pc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.015, aspect=25)
    cb.set_label('Floor classification accuracy')
    cb.ax.tick_params(length=2, pad=1.5)
    fig.legend(handles=[tx_handle], loc='lower center', bbox_to_anchor=(0.45, 0.93),
               frameon=False, borderaxespad=0, handletextpad=0.4)  # transmitter legend above the panels
    fig.savefig(PAPER_FIG, dpi=600, bbox_inches='tight', pad_inches=0.02)  # Fig. 3 (PNG, 600 dpi)
    fig.savefig(PAPER_FIG_PDF, bbox_inches='tight', pad_inches=0.02)  # vector version (reference)
    plt.show()
    print(f'Saved: {PAPER_FIG}, {PAPER_FIG_PDF}')