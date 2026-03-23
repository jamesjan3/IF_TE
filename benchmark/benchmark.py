"""
benchmark.py
------------
Compares wall-clock training time and predictive performance across:
  - Ridge regression  (RidgeCV LOO)
  - ElasticNet        (ElasticNetCV coordinate descent)
  - LightGBM          (RandomizedSearchCV tree ensemble)

Two modes are timed per method:
  "search" — hyperparameter tuning included, matching configs in production scripts
  "fixed"  — single fit with pre-selected representative params, no inner search
             This isolates algorithm speed from tuning overhead.

Two feature subsets are tested:
  "basic"      — 16 features (nucleotide % + lengths)
  "full_model" — all numeric features (~200+)

Fixed hyperparameters are drawn from the existing .out files in Linear/Ridge/ and
Linear/ElasticNet/ (basic subset runs). For full_model, note that optimal alpha
for ElasticNet will differ — fixed-mode results for full_model are indicative only.

LGBM fixed params use mid-range values from the production search grid (LGBM.py),
since no LGBM .out files exist yet.

Output: benchmark/results/timing_summary.csv + printed table.

Run from repo root:
    source Linear/sklearn-env/bin/activate
    python benchmark/benchmark.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(script_dir, "..", "merged_features.csv")
OUTPUT_DIR  = os.path.join(script_dir, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SPLITS     = 5
RANDOM_STATE = 42

TARGET_COLS = {
    "eIF3d": ("eIF3d_control_logTE", "eIF3d_depletion_logTE"),
    "eIF4e": ("eIF4e_control_logTE", "eIF4e_depletion_logTE"),
}

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
# Ridge: alpha ~100 is the median selected value across folds in basic.out
#        (fold range: 52–139, log10 range 1.7–2.1)
RIDGE_FIXED_ALPHA = 100.0

# ElasticNet: alpha ~0.003 and l1_ratio ~0.3 from basic2.out
#   alpha range: 0.00084–0.0065 (mean 0.00335); l1_ratio mean 0.36 → nearest grid value 0.3
#   Note: these are calibrated for the basic (16-feature) subset.
#   For full_model, coordinate descent convergence may differ.
ENET_FIXED_ALPHA = 0.003
ENET_FIXED_L1    = 0.3

# LGBM: mid-range values from the production search grid in LGBM.py
#   (no existing LGBM .out files to draw from)
LGBM_FIXED_PARAMS = dict(
    n_estimators      = 500,
    learning_rate     = 0.05,
    num_leaves        = 31,
    max_depth         = 7,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    objective         = "regression",
    random_state      = RANDOM_STATE,
    n_jobs            = 1,
    verbosity         = -1,
)

# ── Search-mode hyperparameters — matching production scripts exactly ──────────

# Ridge/basic.py: np.logspace(-3, 4, 50)
RIDGE_SEARCH_ALPHAS = np.logspace(-3, 4, 50)

# ElasticNet/basic2.py: L1_RATIOS list + alphas=100 (auto-computed grid)
ENET_SEARCH_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
ENET_SEARCH_ALPHAS    = 100   # int → ElasticNetCV auto-computes grid from data

# LGBM/LGBM.py: RandomizedSearchCV n_iter=10, n_jobs=4
LGBM_SEARCH_PARAM_DIST = {
    "n_estimators"     : [300, 500, 800],
    "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
    "num_leaves"       : [7, 15, 31, 63, 127],
    "max_depth"        : [-1, 3, 5, 7, 10, 15],
    "min_child_samples": [5, 10, 20, 30, 50, 100],
    "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha"        : [0.0, 0.01, 0.1, 1.0, 5.0, 10.0],
    "reg_lambda"       : [0.0, 0.01, 0.1, 1.0, 5.0, 10.0],
}
LGBM_SEARCH_N_ITER = 10
LGBM_SEARCH_JOBS   = 4

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading merged_features.csv...")
df = pd.read_csv(INPUT_FILE)
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

# Build log fold-change targets (control − depletion, matching production scripts)
target_names = []
for label, (ctrl_col, dep_col) in TARGET_COLS.items():
    col_name = f"{label}_logFC"
    df[col_name] = df[ctrl_col] - df[dep_col]
    target_names.append(col_name)

# ── Feature subsets ────────────────────────────────────────────────────────────

_non_feature = {
    "eIF3d_control_logTE", "eIF3d_depletion_logTE",
    "eIF4e_control_logTE", "eIF4e_depletion_logTE",
    "eIF3d_logFC", "eIF4e_logFC",
    "Name",
}

_basic_candidates = [
    "tx_length", "utr5_fraction", "cds_fraction", "utr3_fraction",
    "utr5_A_pct", "utr5_C_pct", "utr5_G_pct", "utr5_T_pct",
    "cds_A_pct",  "cds_C_pct",  "cds_G_pct",  "cds_T_pct",
    "utr3_A_pct", "utr3_C_pct", "utr3_G_pct", "utr3_T_pct",
]
basic_features = [c for c in _basic_candidates if c in df.columns]

full_features = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in _non_feature
]

FEATURE_SUBSETS = {
    "basic"     : basic_features,
    "full_model": full_features,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def prepare_XY(df, feature_cols, target_names):
    data = df[feature_cols + target_names].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())
    data = data.dropna(subset=target_names)
    return data[feature_cols].values, data[target_names].values


def mean_r2_multioutput(Y_test, Y_pred):
    return float(np.mean([
        r2_score(Y_test[:, i], Y_pred[:, i])
        for i in range(Y_test.shape[1])
    ]))

# ── Benchmark runners ──────────────────────────────────────────────────────────
# Each runner takes (X, Y, kf) and returns (fold_times, fold_r2s).
# Timing wraps only the .fit() call — prediction is excluded so we measure
# training cost, not inference.

def run_ridge_search(X, Y, kf):
    fold_times, fold_r2s = [], []
    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(
                RidgeCV(alphas=RIDGE_SEARCH_ALPHAS, scoring="r2")
            )),
        ])
        t0 = time.perf_counter()
        pipe.fit(X[train_idx], Y[train_idx])
        fold_times.append(time.perf_counter() - t0)
        fold_r2s.append(mean_r2_multioutput(Y[test_idx], pipe.predict(X[test_idx])))
    return fold_times, fold_r2s


def run_ridge_fixed(X, Y, kf):
    fold_times, fold_r2s = [], []
    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(Ridge(alpha=RIDGE_FIXED_ALPHA))),
        ])
        t0 = time.perf_counter()
        pipe.fit(X[train_idx], Y[train_idx])
        fold_times.append(time.perf_counter() - t0)
        fold_r2s.append(mean_r2_multioutput(Y[test_idx], pipe.predict(X[test_idx])))
    return fold_times, fold_r2s


def run_enet_search(X, Y, kf):
    fold_times, fold_r2s = [], []
    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(
                ElasticNetCV(
                    l1_ratio = ENET_SEARCH_L1_RATIOS,
                    alphas   = ENET_SEARCH_ALPHAS,
                    cv       = 5,
                    max_iter = 100_000,
                    tol      = 1e-4,
                    n_jobs   = -1,
                ),
                n_jobs=1,  # avoid nested parallelism
            )),
        ])
        t0 = time.perf_counter()
        pipe.fit(X[train_idx], Y[train_idx])
        fold_times.append(time.perf_counter() - t0)
        fold_r2s.append(mean_r2_multioutput(Y[test_idx], pipe.predict(X[test_idx])))
    return fold_times, fold_r2s


def run_enet_fixed(X, Y, kf):
    fold_times, fold_r2s = [], []
    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(
                ElasticNet(
                    alpha    = ENET_FIXED_ALPHA,
                    l1_ratio = ENET_FIXED_L1,
                    max_iter = 100_000,
                    tol      = 1e-4,
                ),
                n_jobs=1,
            )),
        ])
        t0 = time.perf_counter()
        pipe.fit(X[train_idx], Y[train_idx])
        fold_times.append(time.perf_counter() - t0)
        fold_r2s.append(mean_r2_multioutput(Y[test_idx], pipe.predict(X[test_idx])))
    return fold_times, fold_r2s


def run_lgbm_search(X, Y, kf):
    """
    Mirrors LGBM.py: separate RandomizedSearchCV per target, then sum time.
    n_jobs=4 on the search (parallelises over CV folds × hyperparameter combos).
    """
    fold_times, fold_r2s = [], []
    for train_idx, test_idx in kf.split(X):
        t0 = time.perf_counter()
        preds = np.zeros((len(test_idx), Y.shape[1]))
        for i in range(Y.shape[1]):
            search = RandomizedSearchCV(
                LGBMRegressor(
                    objective    = "regression",
                    random_state = RANDOM_STATE,
                    n_jobs       = 1,
                    verbosity    = -1,
                ),
                param_distributions = LGBM_SEARCH_PARAM_DIST,
                n_iter       = LGBM_SEARCH_N_ITER,
                scoring      = "r2",
                cv           = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                refit        = True,
                verbose      = 0,
                random_state = RANDOM_STATE,
                n_jobs       = LGBM_SEARCH_JOBS,
            )
            search.fit(X[train_idx], Y[train_idx, i])
            preds[:, i] = search.predict(X[test_idx])
        fold_times.append(time.perf_counter() - t0)
        fold_r2s.append(mean_r2_multioutput(Y[test_idx], preds))
    return fold_times, fold_r2s


def run_lgbm_fixed(X, Y, kf):
    """Single fit per target with LGBM_FIXED_PARAMS, no inner search."""
    fold_times, fold_r2s = [], []
    for train_idx, test_idx in kf.split(X):
        t0 = time.perf_counter()
        preds = np.zeros((len(test_idx), Y.shape[1]))
        for i in range(Y.shape[1]):
            model = LGBMRegressor(**LGBM_FIXED_PARAMS)
            model.fit(X[train_idx], Y[train_idx, i])
            preds[:, i] = model.predict(X[test_idx])
        fold_times.append(time.perf_counter() - t0)
        fold_r2s.append(mean_r2_multioutput(Y[test_idx], preds))
    return fold_times, fold_r2s

# ── Run all benchmarks ─────────────────────────────────────────────────────────

METHODS = [
    ("Ridge",      "search", run_ridge_search),
    ("Ridge",      "fixed",  run_ridge_fixed),
    ("ElasticNet", "search", run_enet_search),
    ("ElasticNet", "fixed",  run_enet_fixed),
    ("LGBM",       "search", run_lgbm_search),
    ("LGBM",       "fixed",  run_lgbm_fixed),
]

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

all_rows = []

for subset_name, feature_cols in FEATURE_SUBSETS.items():
    print(f"\n{'='*65}")
    print(f"Feature subset : {subset_name}  ({len(feature_cols)} features)")
    print(f"{'='*65}")

    X, Y = prepare_XY(df, feature_cols, target_names)
    print(f"  Samples : {X.shape[0]:,}  |  Features : {X.shape[1]}")

    for method, mode, runner in METHODS:
        label = f"{method:<12} [{mode}]"
        print(f"  {label} ...", end="", flush=True)

        fold_times, fold_r2s = runner(X, Y, kf)

        total_t  = sum(fold_times)
        mean_t   = float(np.mean(fold_times))
        std_t    = float(np.std(fold_times))
        mean_r2v = float(np.mean(fold_r2s))

        print(f"  total={total_t:6.1f}s  "
              f"per-fold={mean_t:.2f}±{std_t:.2f}s  "
              f"mean R²={mean_r2v:.3f}")

        all_rows.append({
            "feature_subset"  : subset_name,
            "n_features"      : len(feature_cols),
            "method"          : method,
            "mode"            : mode,
            "total_time_s"    : round(total_t, 2),
            "mean_fold_time_s": round(mean_t, 3),
            "std_fold_time_s" : round(std_t, 3),
            "mean_r2"         : round(mean_r2v, 4),
        })

# ── Save and print summary ─────────────────────────────────────────────────────

results_df = pd.DataFrame(all_rows)

out_path = os.path.join(OUTPUT_DIR, "timing_summary.csv")
results_df.to_csv(out_path, index=False)

print(f"\n\nResults saved → {out_path}")
print("\n" + "="*80)
print("TIMING SUMMARY  (mean R² averaged over both targets: eIF3d + eIF4e)")
print("="*80)

col_order = ["feature_subset", "n_features", "method", "mode",
             "total_time_s", "mean_fold_time_s", "std_fold_time_s", "mean_r2"]
print(results_df[col_order].to_string(index=False))

# Print a compact search-vs-fixed overhead table
print("\n" + "="*80)
print("SEARCH OVERHEAD  (search_total / fixed_total  — how much tuning costs)")
print("="*80)
pivot = results_df.pivot_table(
    index=["feature_subset", "method"],
    columns="mode",
    values="total_time_s",
)
if "search" in pivot.columns and "fixed" in pivot.columns:
    pivot["overhead_x"] = (pivot["search"] / pivot["fixed"]).round(1)
    print(pivot.to_string())
