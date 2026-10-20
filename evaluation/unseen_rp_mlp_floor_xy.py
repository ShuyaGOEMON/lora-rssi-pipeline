"""Unseen RPs: MLP floor classification and horizontal (x, y) regression.

Baseline evaluation for Table 4 of the paper.

Split : GroupKFold (5 folds, no shuffling); groups = RP key (floor, x, y).
        Same folds as unseen_rp_rf_floor_xy.py.
Model : Two MLPs (floor classifier, x-y regressor), hidden layers [64, 32], BatchNorm,
        ReLU, dropout 0.2, Adam (lr 1e-3), 200 epochs, batch size 256; StandardScaler.
Input : raw/rssi_samples.csv of the dataset v1.0.1 (MD5 4ede8e1a4cab309fa01470e3225f8b39).
        Features rssi_tx1 to rssi_tx4; missing values (no packet received) are filled with -140.

Usage:
    python unseen_rp_mlp_floor_xy.py --csv path/to/raw/rssi_samples.csv [--seed 42]

Outputs (current directory; the metrics are printed to standard output):
    unseen_rp_mlp_floor_xy_folds.csv  test fold (1-5) of every RP (columns: floor, x, y, fold)
    heatmap_unseen_rp_mlp_floor_xy.png   per-RP maps

The folds do not depend on --seed. --seed sets the random, numpy and torch
seeds. The paper reports --seed 42 and the range over the seeds 42, 0, 1, 2, 3.
The fold assignment depends on the group strings and the scikit-learn version.
GPU results may not be bit-identical across hardware and library versions.

Environment used for the paper: Python 3.13.15, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, torch 2.11.0+cu130 (CUDA).
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
FILL_VALUE = -140

# MLP hyperparameters
HIDDEN_SIZES = [64, 32]     # hidden units
LEARNING_RATE = 1e-3
EPOCHS = 200
BATCH_SIZE = 256
DROPOUT = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Record the execution environment and fix the random seeds
import sys, random, sklearn
print(f"Python {sys.version.split()[0]} / numpy {np.__version__} / pandas {pd.__version__}"
      f" / scikit-learn {sklearn.__version__} / torch {torch.__version__}")
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 1. Load the data
# ============================================================
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
X = df[feature_cols].values.astype(np.float32)
y_z = df[FLOOR_COL].astype(int).values
y_xy = df[['x', 'y']].values.astype(np.float32)

# Fill missing values
imputer = SimpleImputer(strategy='constant', fill_value=FILL_VALUE)
X = imputer.fit_transform(X)

floors = sorted(np.unique(y_z))
floor_to_idx = {f: i for i, f in enumerate(floors)}
y_z_idx = np.array([floor_to_idx[f] for f in y_z])  # 0-indexed

groups = (df[FLOOR_COL].astype(str) + '_' + df['x'].astype(str)
          + '_' + df['y'].astype(str)).values

n_features = X.shape[1]
n_floors = len(floors)
print(f"Samples: {len(X)}, features: {n_features}, floors: {n_floors}")
print(f"RPs: {len(np.unique(groups))}")
print()

# ============================================================
# 2. MLP models
# ============================================================
class FloorClassifier(nn.Module):
    """MLP for floor classification"""
    def __init__(self, input_dim, hidden_sizes, n_classes, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class XYRegressor(nn.Module):
    """MLP for horizontal (x, y) regression"""
    def __init__(self, input_dim, hidden_sizes, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))  # x, y
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()


def predict_model(model, X_tensor):
    model.eval()
    with torch.no_grad():
        return model(X_tensor.to(device)).cpu().numpy()


# ============================================================
# 3. Cross-validation (GroupKFold: split by RP)
# ============================================================
# No shuffling. The fold assignment is determined by the group strings and the scikit-learn version.
# unseen_rp_rf_floor_xy.py and unseen_rp_mlp_floor_xy.py build the groups in the same way and use the same folds.
gkf = GroupKFold(n_splits=N_SPLITS)

all_z_true = []
all_z_pred = []
all_distances_correct = []
all_distances_wrong = []
all_floor_errors = []
all_3d_errors = []
fold_z_acc = []

# --- Per-sample values for the per-RP maps ---
rp_true_x = []
rp_true_y = []
rp_true_z = []
rp_correct = []
rp_horiz = []

print("=" * 60)
print("Cross-validation (GroupKFold + MLP)")
print("=" * 60)

fold_test_idx = []  # (fold, test row indices) for the fold assignment file
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_z, groups=groups), 1):
    fold_test_idx.append((fold, test_idx))
    # Split
    X_train, X_test = X[train_idx], X[test_idx]
    y_z_train, y_z_test = y_z_idx[train_idx], y_z_idx[test_idx]
    y_xy_train, y_xy_test = y_xy[train_idx], y_xy[test_idx]

    train_rps = len(np.unique(groups[train_idx]))
    test_rps = len(np.unique(groups[test_idx]))

    # --- Standardization ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- To tensors ---
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_z_train_t = torch.tensor(y_z_train, dtype=torch.long)
    y_xy_train_t = torch.tensor(y_xy_train, dtype=torch.float32)

    # ---- (1) Floor classification (MLP) ----
    torch.manual_seed(RANDOM_STATE)  # fixed seed for each model in each fold
    clf_model = FloorClassifier(n_features, HIDDEN_SIZES, n_floors, DROPOUT).to(device)
    clf_optimizer = optim.Adam(clf_model.parameters(), lr=LEARNING_RATE)
    clf_criterion = nn.CrossEntropyLoss()

    clf_dataset = TensorDataset(X_train_t, y_z_train_t)
    clf_loader = DataLoader(clf_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_model(clf_model, clf_loader, clf_criterion, clf_optimizer, EPOCHS)

    z_pred_logits = predict_model(clf_model, X_test_t)
    z_pred_idx = np.argmax(z_pred_logits, axis=1)
    z_pred_floor = np.array([floors[i] for i in z_pred_idx])
    y_z_test_floor = np.array([floors[i] for i in y_z_test])

    z_acc = accuracy_score(y_z_test_floor, z_pred_floor)
    fold_z_acc.append(z_acc)
    all_z_true.extend(y_z_test_floor.tolist())
    all_z_pred.extend(z_pred_floor.tolist())

    floor_err = np.abs(y_z_test_floor - z_pred_floor)
    all_floor_errors.extend(floor_err.tolist())

    # ---- (2) Horizontal regression (MLP) ----
    torch.manual_seed(RANDOM_STATE)  # fixed seed for each model in each fold
    reg_model = XYRegressor(n_features, HIDDEN_SIZES, DROPOUT).to(device)
    reg_optimizer = optim.Adam(reg_model.parameters(), lr=LEARNING_RATE)
    reg_criterion = nn.MSELoss()

    reg_dataset = TensorDataset(X_train_t, y_xy_train_t)
    reg_loader = DataLoader(reg_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_model(reg_model, reg_loader, reg_criterion, reg_optimizer, EPOCHS)

    xy_pred = predict_model(reg_model, X_test_t)
    distances = np.sqrt(np.sum((xy_pred - y_xy_test) ** 2, axis=1))

    correct_mask = (y_z_test_floor == z_pred_floor)
    wrong_mask = ~correct_mask
    all_distances_correct.extend(distances[correct_mask].tolist())
    all_distances_wrong.extend(distances[wrong_mask].tolist())

    z_diff_m = np.abs(y_z_test_floor - z_pred_floor) * FLOOR_HEIGHT_M
    dist_3d = np.sqrt(distances ** 2 + z_diff_m ** 2)
    all_3d_errors.extend(dist_3d.tolist())

    # ---- Store values for the per-RP maps ----
    rp_true_x.extend(y_xy_test[:, 0].tolist())
    rp_true_y.extend(y_xy_test[:, 1].tolist())
    rp_true_z.extend(y_z_test_floor.tolist())
    rp_correct.extend(correct_mask.astype(int).tolist())
    rp_horiz.extend(distances.tolist())

    mean_xy_correct = np.mean(distances[correct_mask]) if correct_mask.sum() > 0 else float('nan')
    print(f"Fold {fold} | train RPs: {train_rps} test RPs: {test_rps}"
          f" | floor accuracy: {z_acc:.4f}"
          f" | horizontal error (correct floor): {mean_xy_correct:.2f}m"
          f" | wrong floor: {wrong_mask.sum()}/{len(z_pred_floor)}")

# ============================================================
# Fold assignment (test fold of every RP)
# ============================================================
FOLD_CSV = 'unseen_rp_mlp_floor_xy_folds.csv'
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
print("Overall results (MLP + GroupKFold)")
print("=" * 60)

print(f"\n[Floor classification]")
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

print("[Horizontal error (correct floor only)]")
dc = np.array(all_distances_correct)
print(f"  Samples    : {len(dc)}")
print(f"  Mean       : {np.mean(dc):.2f} m")
print(f"  Median     : {np.median(dc):.2f} m")
print(f"  75%ile     : {np.percentile(dc, 75):.2f} m")
print(f"  90%ile     : {np.percentile(dc, 90):.2f} m")
print(f"  Max        : {np.max(dc):.2f} m")

if len(all_distances_wrong) > 0:
    print(f"\n[Horizontal error (wrong floor, for reference)]")
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

print(f"\n[Summary for comparison with RF (unseen_rp_rf_floor_xy.py)]")
print(f"  MLP floor accuracy     : {np.mean(fold_z_acc):.4f}")
print(f"  MLP horizontal median : {np.median(dc):.2f} m")
print(f"  MLP horizontal p90    : {np.percentile(dc, 90):.2f} m")

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

fig.suptitle('Unseen RPs, MLP, GroupKFold: per-RP floor accuracy (top) and horizontal error (bottom)', fontsize=14)
out_path = 'heatmap_unseen_rp_mlp_floor_xy.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}  (RPs: {len(rp_agg)})')