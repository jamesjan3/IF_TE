"""
benchmark.py
------------
Benchmarks Ridge, ElasticNet, and LightGBM on two feature sets:
  basic — 16 features (lengths + nucleotide frequencies)
  full  — all numeric sequence features (~1131 features)

Both sets match the corresponding rows in Table 1 of the report.
Comparing the two reveals how training time scales with dimensionality.

Methodology
-----------
Follows nested cross-validation standards used in bioinformatics benchmarks
(Vabalas et al. 2019; Krstajic et al. 2014; Littmann et al. 2021):

  Outer CV  : 5-fold, matching the cross-validation scheme used in the
               main analysis (80/20 splits)
  Inner CV  : 5-fold within each training fold, for hyperparameter selection

Each method uses its most computationally efficient tuning strategy:
  Ridge       — analytical regularisation path (RidgeCV); evaluating all
                50 alpha candidates costs O(p³) once via SVD, not per alpha
  ElasticNet  — warm-started coordinate descent path (ElasticNetCV); the
                full alpha grid is traversed in one pass per l1_ratio per fold
  LightGBM    — two-stage: randomised search (n_iter=10) over structural
                hyperparameters with fixed n_estimators=300, followed by a
                single refit with early stopping to determine tree count
                (Ke et al. 2017). This separates architecture search from
                depth selection, avoiding the cost of evaluating many
                n_estimators candidates.

Metrics reported per depletion condition (eIF3d, eIF4e):
  R²   — coefficient of determination on held-out fold
  RMSE — root mean squared error on held-out fold

Timing (wall-clock seconds for the full .fit() call including inner CV) is
reported as secondary output.

Outputs
-------
  benchmark/results/benchmark_folds.csv   — per-fold raw scores
  benchmark/results/benchmark_summary.csv — mean ± SD across folds

Run from repo root:
    source Linear/sklearn-env/bin/activate
    python benchmark/benchmark.py

References
----------
Ke G et al. (2017) LightGBM: A highly efficient gradient boosting decision
    tree. NeurIPS 30.
Krstajic D et al. (2014) Cross-validation pitfalls when selecting and
    assessing regression and classification models. J Cheminform 6:10.
Vabalas A et al. (2019) Machine learning algorithm validation with a limited
    sample size. PLoS ONE 14:e0224365.
"""

import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

script_dir   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(script_dir, "..", "merged_features.csv")
OUTPUT_DIR   = os.path.join(script_dir, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_OUTER      = 5
N_INNER      = 5
RANDOM_STATE = 42

TARGET_COLS = {
    "eIF3d": ("eIF3d_control_logTE", "eIF3d_depletion_logTE"),
    "eIF4e": ("eIF4e_control_logTE", "eIF4e_depletion_logTE"),
}

# ── Hyperparameter search spaces ───────────────────────────────────────────────

# Ridge: 50 log-spaced candidates, evaluated analytically via SVD
RIDGE_ALPHAS = np.logspace(-3, 4, 50)

# ElasticNet: data-driven alpha grid (n_alphas=100) × 7 l1_ratio values,
# traversed via warm-started coordinate descent
ENET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
ENET_N_ALPHAS  = 100

# LightGBM two-stage tuning:
#   Stage 1 — randomised search over structural params, fixed n_estimators
#   Stage 2 — refit best config with early stopping to determine tree count
LGBM_N_ESTIMATORS_SEARCH = 300   # fixed during architecture search
LGBM_N_ESTIMATORS_MAX    = 1000  # ceiling for early-stopping refit
LGBM_EARLY_STOPPING      = 50    # halt if no improvement for 50 rounds
LGBM_PARAM_DIST = {
    "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
    "num_leaves"       : [7, 15, 31, 63, 127],
    "max_depth"        : [-1, 3, 5, 7, 10, 15],
    "min_child_samples": [5, 10, 20, 30, 50, 100],
    "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha"        : [0.0, 0.01, 0.1, 1.0, 5.0, 10.0],
    "reg_lambda"       : [0.0, 0.01, 0.1, 1.0, 5.0, 10.0],
}
LGBM_N_ITER = 10
LGBM_N_JOBS = 4   # parallelise RandomizedSearchCV over inner CV folds

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading merged_features.csv...")
df = pd.read_csv(INPUT_FILE)
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

target_names = []
for label, (ctrl_col, dep_col) in TARGET_COLS.items():
    col_name = f"{label}_logFC"
    df[col_name] = df[ctrl_col] - df[dep_col]
    target_names.append(col_name)

# ── Feature subsets ────────────────────────────────────────────────────────────

BASIC_FEATURES = [
    "tx_length",    "utr5_fraction", "cds_fraction",  "utr3_fraction",
    "utr5_A_pct",   "utr5_C_pct",    "utr5_G_pct",    "utr5_T_pct",
    "cds_A_pct",    "cds_C_pct",     "cds_G_pct",     "cds_T_pct",
    "utr3_A_pct",   "utr3_C_pct",    "utr3_G_pct",    "utr3_T_pct",
]

# Full set: all numeric columns except targets and ID — mirrors full_model in
# LGBM.py and the union of all active subsets in ElasticNet/full2.py
NON_FEATURE_COLS = set(target_names) | {
    "eIF3d_control_logTE", "eIF3d_depletion_logTE",
    "eIF4e_control_logTE", "eIF4e_depletion_logTE",
    "sample_id",
}
numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
FULL_FEATURES = [c for c in numeric_cols if c not in NON_FEATURE_COLS]

FEATURE_SUBSETS = {
    "basic": [c for c in BASIC_FEATURES if c in df.columns],
    "full":  FULL_FEATURES,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def prepare_XY(df, feature_cols, target_names):
    """Extract arrays; replace ±inf; drop rows with missing targets.
    Feature NaNs are left for per-fold SimpleImputer to prevent leakage."""
    data = df[feature_cols + target_names].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=target_names)
    return data[feature_cols].values, data[target_names].values


def per_target_metrics(Y_true, Y_pred):
    """Return (r2, rmse) arrays of shape (n_targets,)."""
    n = Y_true.shape[1]
    r2s   = np.array([r2_score(Y_true[:, i], Y_pred[:, i]) for i in range(n)])
    rmses = np.array([np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i]))
                      for i in range(n)])
    return r2s, rmses

# ── Runners ────────────────────────────────────────────────────────────────────
# Each takes (X, Y, outer_kf) and returns:
#   fold_times  — list[float], wall-clock seconds per fold (fit only)
#   fold_r2s    — list[ndarray(n_targets,)], R² per fold per target
#   fold_rmses  — list[ndarray(n_targets,)], RMSE per fold per target

def run_ridge(X, Y, kf):
    """RidgeCV: all 50 alpha candidates evaluated via one SVD per fold."""
    fold_times, fold_r2s, fold_rmses = [], [], []
    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   MultiOutputRegressor(
                RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2")
            )),
        ])
        t0 = time.perf_counter()
        pipe.fit(X[train_idx], Y[train_idx])
        fold_times.append(time.perf_counter() - t0)
        r2s, rmses = per_target_metrics(Y[test_idx], pipe.predict(X[test_idx]))
        fold_r2s.append(r2s)
        fold_rmses.append(rmses)
    return fold_times, fold_r2s, fold_rmses


def run_elasticnet(X, Y, kf):
    """ElasticNetCV: warm-started path over 100 alphas × 7 l1_ratios, 5-fold."""
    fold_times, fold_r2s, fold_rmses = [], [], []
    for train_idx, test_idx in kf.split(X):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   MultiOutputRegressor(
                ElasticNetCV(
                    l1_ratio = ENET_L1_RATIOS,
                    n_alphas = ENET_N_ALPHAS,
                    cv       = N_INNER,
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
        r2s, rmses = per_target_metrics(Y[test_idx], pipe.predict(X[test_idx]))
        fold_r2s.append(r2s)
        fold_rmses.append(rmses)
    return fold_times, fold_r2s, fold_rmses


def run_lgbm(X, Y, kf):
    """Two-stage tuning per target:
    1. RandomizedSearchCV (n_iter=10, fixed n_estimators=300) selects
       structural hyperparameters via 5-fold inner CV.
    2. Best config is refit with early stopping on a 10% validation split
       (within the outer training fold) to determine optimal tree count.
    Separating the two stages avoids evaluating many n_estimators candidates
    while still allowing adaptive depth selection.
    """
    fold_times, fold_r2s, fold_rmses = [], [], []
    es_callbacks = [lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(period=-1)]

    for train_idx, test_idx in kf.split(X):
        imputer  = SimpleImputer(strategy="median")
        X_train  = imputer.fit_transform(X[train_idx])
        X_test   = imputer.transform(X[test_idx])

        # 10% of training fold held out for early stopping only
        n_val   = max(1, int(0.1 * len(train_idx)))
        X_tr    = X_train[:-n_val]
        X_val   = X_train[-n_val:]

        t0    = time.perf_counter()
        preds = np.zeros((len(test_idx), Y.shape[1]))

        for i in range(Y.shape[1]):
            y_tr  = Y[train_idx[:-n_val], i]
            y_val = Y[train_idx[-n_val:],  i]

            # Stage 1: architecture search
            search = RandomizedSearchCV(
                LGBMRegressor(
                    n_estimators = LGBM_N_ESTIMATORS_SEARCH,
                    objective    = "regression",
                    random_state = RANDOM_STATE,
                    n_jobs       = 1,
                    verbosity    = -1,
                ),
                param_distributions = LGBM_PARAM_DIST,
                n_iter       = LGBM_N_ITER,
                scoring      = "r2",
                cv           = KFold(n_splits=N_INNER, shuffle=True,
                                     random_state=RANDOM_STATE),
                refit        = False,
                verbose      = 0,
                random_state = RANDOM_STATE,
                n_jobs       = LGBM_N_JOBS,
            )
            search.fit(X_tr, y_tr)

            # Stage 2: refit best config with early stopping
            model = LGBMRegressor(
                n_estimators = LGBM_N_ESTIMATORS_MAX,
                objective    = "regression",
                random_state = RANDOM_STATE,
                n_jobs       = 1,
                verbosity    = -1,
                **search.best_params_,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=es_callbacks)
            preds[:, i] = model.predict(X_test)

        fold_times.append(time.perf_counter() - t0)
        r2s, rmses = per_target_metrics(Y[test_idx], preds)
        fold_r2s.append(r2s)
        fold_rmses.append(rmses)

    return fold_times, fold_r2s, fold_rmses

# ── Run ────────────────────────────────────────────────────────────────────────

METHODS = [
    ("Ridge",      run_ridge),
    ("ElasticNet", run_elasticnet),
    ("LightGBM",   run_lgbm),
]

outer_kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

fold_rows    = []
summary_rows = []

for subset_name, feature_cols in FEATURE_SUBSETS.items():
    X, Y = prepare_XY(df, feature_cols, target_names)

    print(f"\n{'='*72}")
    print(f"Feature set  : {subset_name} ({len(feature_cols)} features)")
    print(f"Samples      : {X.shape[0]:,}")
    print(f"Outer folds  : {N_OUTER}  |  Inner folds : {N_INNER}")
    print(f"{'='*72}")

    for method, runner in METHODS:
        print(f"  {method:<12} ...", end="", flush=True)
        fold_times, fold_r2s, fold_rmses = runner(X, Y, outer_kf)

        r2_mat   = np.array(fold_r2s)
        rmse_mat = np.array(fold_rmses)

        parts = []
        for j, tname in enumerate(target_names):
            parts.append(f"{tname.split('_')[0]} R²={r2_mat[:,j].mean():.3f}"
                         f"±{r2_mat[:,j].std():.3f}")
        print("  " + "  ".join(parts) + f"  time={sum(fold_times):.1f}s")

        # Per-fold rows
        for fold_i, (t, r2s, rmses) in enumerate(
                zip(fold_times, fold_r2s, fold_rmses)):
            row = {"subset": subset_name, "method": method,
                   "fold": fold_i + 1, "time_s": round(t, 3)}
            for j, tname in enumerate(target_names):
                short = tname.replace("_logFC", "")
                row[f"r2_{short}"]   = round(float(r2s[j]),   4)
                row[f"rmse_{short}"] = round(float(rmses[j]), 4)
            fold_rows.append(row)

        # Summary row
        row = {"subset": subset_name, "method": method,
               "n_features": len(feature_cols),
               "total_time_s": round(sum(fold_times), 2)}
        for j, tname in enumerate(target_names):
            short = tname.replace("_logFC", "")
            row[f"mean_r2_{short}"]   = round(float(r2_mat[:,j].mean()),   4)
            row[f"sd_r2_{short}"]     = round(float(r2_mat[:,j].std()),    4)
            row[f"mean_rmse_{short}"] = round(float(rmse_mat[:,j].mean()), 4)
            row[f"sd_rmse_{short}"]   = round(float(rmse_mat[:,j].std()),  4)
        summary_rows.append(row)

# ── Save outputs ───────────────────────────────────────────────────────────────

pd.DataFrame(fold_rows).to_csv(
    os.path.join(OUTPUT_DIR, "benchmark_folds.csv"), index=False)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "benchmark_summary.csv"), index=False)

# ── Print summary ──────────────────────────────────────────────────────────────

print(f"\n{'='*72}")
print("BENCHMARK SUMMARY")
print(f"  Nested CV : {N_OUTER}-fold outer / {N_INNER}-fold inner")
print(f"  R² and RMSE = mean ± SD over {N_OUTER} outer folds")
print(f"{'='*72}")
print(summary_df.to_string(index=False))

print(f"\nOutputs saved to {OUTPUT_DIR}/")
print(f"  benchmark_folds.csv   — per-fold raw scores")
print(f"  benchmark_summary.csv — mean ± SD summary")
