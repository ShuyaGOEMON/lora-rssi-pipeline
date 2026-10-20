"""Known RPs: random-forest RP classification (one class per RP, 278 classes).

Baseline evaluation for Table 4 of the paper.

Split : StratifiedKFold (5 folds, shuffle=True, random_state=--seed), stratified by RP class.
        Same folds as known_rp_mlp_rp_classification.py.
Model : RandomForestClassifier, n_estimators=100, other parameters at the scikit-learn
        defaults. Floor and (x, y) are taken from the predicted RP.
Input : raw/rssi_samples.csv of the dataset v1.0.1 (MD5 4ede8e1a4cab309fa01470e3225f8b39).
        Features rssi_tx1 to rssi_tx4; missing values (no packet received) are filled with -140.

Usage:
    python known_rp_rf_rp_classification.py --csv path/to/raw/rssi_samples.csv [--seed 42]

Outputs (current directory; the metrics are printed to standard output):
    known_rp_rf_rp_classification_folds.csv  test fold (1-5) of every row of the input CSV
        (columns: row = 0-based data row, floor, x, y, sample_index, fold)
    heatmap_known_rp_rf_rp_classification.png   per-RP maps

--seed sets random_state of StratifiedKFold and of the random forest, so
changing it also changes the folds.

Environment used for the paper: Python 3.13.15, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
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

# --- Target: RP class ---
# Each RP key (z, x, y) is one class (278 classes)
rp_labels = (df[FLOOR_COL].astype(str) + '_' + df['x'].astype(str)
             + '_' + df['y'].astype(str)).values

le = LabelEncoder()
y_rp = le.fit_transform(rp_labels)
n_classes = len(le.classes_)

# Class -> (z, x, y)
class_to_coords = {}
for cls_idx, cls_label in enumerate(le.classes_):
    parts = cls_label.split('_')
    class_to_coords[cls_idx] = (int(parts[0]), float(parts[1]), float(parts[2]))

floors = sorted(set(c[0] for c in class_to_coords.values()))

print(f"Samples          : {len(X)}")
print(f"Features         : {X.shape[1]}")
print(f"Floors           : {len(floors)}  {floors}")
print(f"RPs (classes)    : {n_classes}")
print(f"Samples per RP   : {len(X) // n_classes}")
print()

# ============================================================
# 2. Build the pipeline (278-class RP classifier)
# ============================================================
clf_rp = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-140)),
    ('model', RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ))
])

# ============================================================
# 3. Cross-validation (StratifiedKFold: stratified by RP class)
# ============================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_rp_acc = []
fold_z_acc = []
all_rp_true = []
all_rp_pred = []
all_z_true = []
all_z_pred = []
all_distances_correct = []   # horizontal error when the floor is correct
all_distances_wrong = []     # horizontal error when the floor is wrong
all_floor_errors = []
all_3d_errors = []

# --- Per-sample values for the per-RP maps ---
rp_true_x = []
rp_true_y = []
rp_true_z = []
rp_correct = []
rp_horiz = []

print("=" * 60)
print("Cross-validation (StratifiedKFold: stratified by RP class)")
print("=" * 60)

fold_test_idx = []  # (fold, test row indices) for the fold assignment file
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_rp), 1):
    fold_test_idx.append((fold, test_idx))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_rp_train, y_rp_test = y_rp[train_idx], y_rp[test_idx]

    # ---- Train and predict the RP classifier ----
    clf_rp.fit(X_train, y_rp_train)
    rp_pred = clf_rp.predict(X_test)

    rp_acc = accuracy_score(y_rp_test, rp_pred)
    fold_rp_acc.append(rp_acc)
    all_rp_true.extend(y_rp_test.tolist())
    all_rp_pred.extend(rp_pred.tolist())

    # ---- Recover (z, x, y) from the predicted class ----
    pred_coords = np.array([class_to_coords[c] for c in rp_pred])
    true_coords = np.array([class_to_coords[c] for c in y_rp_test])

    z_pred = pred_coords[:, 0].astype(int)
    z_true = true_coords[:, 0].astype(int)
    xy_pred = pred_coords[:, 1:3]
    xy_true = true_coords[:, 1:3]

    # ---- Floor (derived from z of the predicted RP) ----
    z_acc = accuracy_score(z_true, z_pred)
    fold_z_acc.append(z_acc)
    all_z_true.extend(z_true.tolist())
    all_z_pred.extend(z_pred.tolist())
    all_floor_errors.extend(np.abs(z_true - z_pred).tolist())

    # ---- Horizontal error (derived from x, y of the predicted RP) ----
    distances = np.sqrt(np.sum((xy_pred - xy_true) ** 2, axis=1))
    correct_mask = (z_true == z_pred)
    all_distances_correct.extend(distances[correct_mask].tolist())
    all_distances_wrong.extend(distances[~correct_mask].tolist())

    z_diff_m = np.abs(z_true - z_pred) * FLOOR_HEIGHT_M
    dist_3d = np.sqrt(distances ** 2 + z_diff_m ** 2)
    all_3d_errors.extend(dist_3d.tolist())

    # ---- Store values for the per-RP maps ----
    rp_true_x.extend(xy_true[:, 0].tolist())
    rp_true_y.extend(xy_true[:, 1].tolist())
    rp_true_z.extend(z_true.tolist())
    rp_correct.extend(correct_mask.astype(int).tolist())
    rp_horiz.extend(distances.tolist())

    mean_xy_correct = np.mean(distances[correct_mask]) if correct_mask.sum() > 0 else float('nan')
    print(f"Fold {fold} | RP accuracy: {rp_acc:.4f}"
          f" | floor accuracy: {z_acc:.4f}"
          f" | horizontal error (all): {np.mean(distances):.2f}m"
          f" | horizontal error (correct floor): {mean_xy_correct:.2f}m"
          f" | wrong RP: {(y_rp_test != rp_pred).sum()}/{len(rp_pred)}")

# ============================================================
# Fold assignment (test fold of every row of the input CSV)
# ============================================================
FOLD_CSV = 'known_rp_rf_rp_classification_folds.csv'
fold_df = df[[c for c in [FLOOR_COL, 'x', 'y', 'sample_index'] if c in df.columns]].copy()
fold_df.insert(0, 'row', np.arange(len(df)))
fold_df['fold'] = 0
for fold, test_idx in fold_test_idx:
    fold_df.iloc[test_idx, fold_df.columns.get_loc('fold')] = fold
assert (fold_df['fold'] > 0).all(), "Some rows are not in any test fold"
fold_df.to_csv(FOLD_CSV, index=False)
print(f'Saved: {FOLD_CSV}')

# ============================================================
# 4. Overall results
# ============================================================
print()
print("=" * 60)
print("Overall results (known RPs, RF, RP classification)")
print("=" * 60)

# ---- RP classification ----
print(f"\n[RP classification ({n_classes} classes)]")
print(f"  Mean accuracy : {np.mean(fold_rp_acc):.4f} (SD: {np.std(fold_rp_acc):.4f})")

# ---- Floor classification ----
print(f"\n[Floor classification (derived from z of the predicted RP)]")
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

if len(all_distances_wrong) > 0:
    print("\n[Horizontal error (wrong floor, for reference)]")
    dw = np.array(all_distances_wrong)
    print(f"  Samples    : {len(dw)}")
    print(f"  Mean       : {np.mean(dw):.2f} m")
    print(f"  Median     : {np.median(dw):.2f} m")

print(f"\n[3D error (auxiliary metric, floor height = {FLOOR_HEIGHT_M} m)]")
d3 = np.array(all_3d_errors)
print(f"  Mean       : {np.mean(d3):.2f} m")
print(f"  Median     : {np.median(d3):.2f} m")
print(f"  75%ile     : {np.percentile(d3, 75):.2f} m")
print(f"  90%ile     : {np.percentile(d3, 90):.2f} m")

# ---- Summary ----
print(f"\n[Summary]")
print(f"  RP accuracy           : {np.mean(fold_rp_acc):.4f}")
print(f"  Floor accuracy        : {np.mean(fold_z_acc):.4f}")
print(f"  Horizontal median     : {np.median(dc):.2f} m")

# ============================================================
# 5. Per-RP maps (one image)
# ============================================================
import matplotlib.pyplot as plt

# Aggregate the test samples by (z, x, y) and average per RP
rp_df = pd.DataFrame({
    'x': rp_true_x, 'y': rp_true_y, 'z': rp_true_z,
    'correct': rp_correct, 'horiz': rp_horiz,
})

rp_agg = rp_df.groupby(['z', 'x', 'y']).agg(
    accuracy=('correct', 'mean'),   # floor accuracy per RP
    horiz_err=('horiz', 'mean'),    # mean horizontal error per RP
    n=('correct', 'size'),
).reset_index()

floors_sorted = sorted(rp_agg['z'].unique())
n_fl = len(floors_sorted)

# Top row: floor accuracy, bottom row: horizontal error (2 rows x floors)
fig, axes = plt.subplots(2, n_fl, figsize=(5 * n_fl, 9), squeeze=False)

rows = [
    (0, 'accuracy', 'Floor Accuracy', 'viridis', 0.0, 1.0),
    (1, 'horiz_err', 'Horizontal Error [m]', 'hot_r', 0, 12),
]

for row_idx, metric_col, metric_label, cmap, vmin, vmax in rows:
    vmin_use = rp_agg[metric_col].min() if vmin is None else vmin
    vmax_use = rp_agg[metric_col].max() if vmax is None else vmax
    sc = None
    for ax, fl in zip(axes[row_idx], floors_sorted):
        sub = rp_agg[rp_agg['z'] == fl]
        sc = ax.scatter(sub['x'], sub['y'], c=sub[metric_col],
                        cmap=cmap, vmin=vmin_use, vmax=vmax_use,
                        s=140, marker='s', edgecolors='gray', linewidths=0.5)
        ax.set_title(f'{fl}F ({metric_label})')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc, ax=list(axes[row_idx]), shrink=0.8, label=metric_label)

fig.suptitle('Known RPs, RF, RP classification: per-RP floor accuracy (top) and horizontal error (bottom)', fontsize=14)
out_path = 'heatmap_known_rp_rf_rp_classification.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}  (RPs: {len(rp_agg)})')