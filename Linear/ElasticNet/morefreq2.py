"""
train_linear.py
---------------
Trains a multi-output ElasticNet regression model to predict:
    target = log(TE_control / TE_depletion)
           = eIF_control_logTE - eIF_depletion_logTE

for both eIF3d and eIF4e simultaneously (2-output regression).

ElasticNet combines L1 (Lasso) and L2 (Ridge) regularization:
  - L1 drives irrelevant feature coefficients to exactly zero (feature selection)
  - L2 handles correlated features gracefully (shrinks together)
  - l1_ratio controls the mix: 0 = pure Ridge, 1 = pure Lasso

Evaluation strategy: 5-fold CV. The data is conceptually divided into 10
chunks of 10% each. Each fold holds out 2 chunks (20%) and trains on the
remaining 8 (80%). 5 folds cover all data exactly once with no overlap.

Hyperparameter tuning: ElasticNetCV tunes alpha and l1_ratio jointly via
inner CV on each training fold.

Feature importance: standardized coefficients (coef × feature std),
averaged across folds, ranked per output. Zero-coefficient features
(zeroed out by L1) are reported separately.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# ── Config ─────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(script_dir, "..", "..", "merged_features.csv")

TARGET_COLS = {
    "eIF3d": ("eIF3d_control_logTE", "eIF3d_depletion_logTE"),
    "eIF4e": ("eIF4e_control_logTE", "eIF4e_depletion_logTE"),
}

# Static subsets: fully specified here
feature_subsets = {
    "min_dg": ["min_dg_5p60", "min_dg_startCentered60"],
    "utr5_nt": ["utr5_A_pct", "utr5_C_pct", "utr5_G_pct", "utr5_T_pct"],
    "cds_nt":  ["cds_A_pct",  "cds_C_pct",  "cds_G_pct",  "cds_T_pct"],
    "utr3_nt": ["utr3_A_pct", "utr3_C_pct", "utr3_G_pct", "utr3_T_pct"],
    "cds_wobble_nt": [
        "cds_wobble_A_pct", "cds_wobble_C_pct",
        "cds_wobble_G_pct", "cds_wobble_T_pct",
    ],
    "lengths":        ["tx_length", "utr5_fraction", "cds_fraction", "utr3_fraction"],
    "non5_lengths": ["tx_length", "cds_fraction"],
    "kozak": [
        "-3_A", "-3_C", "-3_G",
        "-2_A", "-2_C", "-2_G",
        "-1_A", "-1_C", "-1_G",
        "+4_A", "+4_C", "+4_G",
        "+5_A", "+5_C", "+5_G",
    ],
}

# Dynamic subsets: defined by column prefix, expanded after data load
DYNAMIC_SUBSETS = {
    "codon_freq": "codon_",
    "aa_freq":    "aa_",
    "utr5_k2":    "utr5_k2",
    "utr5_k3":    "utr5_k3",
    "utr5_k4":    "utr5_k4",
    "cds_k2":     "cds_k2",
    "cds_k3":     "cds_k3",
    "cds_k4":     "cds_k4",
    "utr3_k2":    "utr3_k2",
    "utr3_k3":    "utr3_k3",
    "utr3_k4":    "utr3_k4",
}

# ── Edit here to change which features are used ────────────────────────────────
ACTIVE_SUBSETS = ["utr5_nt", "cds_nt", "utr3_nt", "lengths", "kozak", "codon_freq", "aa_freq", "utr5_k2", "utr5_k3", "utr5_k4","cds_k2","cds_k3","cds_k4","utr3_k2","utr3_k3","utr3_k4"]
# ──────────────────────────────────────────────────────────────────────────────

N_SPLITS       = 5                          # 5-fold = 80/20 splits
# l1_ratio grid: 0=Ridge, 1=Lasso; values near 1 encourage sparsity
L1_RATIOS      = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
# ALPHAS: number of alphas in the auto-computed grid.
# ElasticNetCV derives alpha_max from the data and builds a log-spaced grid
# downward. This is always appropriate regardless of dimensionality, avoiding
# convergence failures that occur when manually supplying too-small alphas.
ALPHAS         = 40    # size of auto-computed alpha grid per fold
TOP_N_FEATURES = 10

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

# Expand only the dynamic subsets that are actually requested
for name in ACTIVE_SUBSETS:
    if name in DYNAMIC_SUBSETS:
        feature_subsets[name] = [c for c in df.columns if c.startswith(DYNAMIC_SUBSETS[name])]

# ── Build targets ──────────────────────────────────────────────────────────────

target_names = []
new_cols = {}
for label, (ctrl_col, dep_col) in TARGET_COLS.items():
    col_name = f"{label}_logFC"
    new_cols[col_name] = df[ctrl_col] - df[dep_col]
    target_names.append(col_name)

df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

print("\nTargets (control_logTE − depletion_logTE = log fold-change):")
for name in target_names:
    print(f"  {name}: mean={df[name].mean():.3f}, std={df[name].std():.3f}")

# ── Assemble feature matrix ────────────────────────────────────────────────────

all_features = []
for subset in ACTIVE_SUBSETS:
    cols    = [c for c in feature_subsets[subset] if c in df.columns]
    missing = [c for c in feature_subsets[subset] if c not in df.columns]
    if missing:
        print(f"  [WARN] subset '{subset}' missing columns: {missing}")
    all_features.extend(cols)

print(f"\nActive subsets : {ACTIVE_SUBSETS}")
print(f"Total features : {len(all_features)}")

keep_cols = all_features + target_names
df_clean  = df[keep_cols].copy()

# Replace ±inf with NaN, then impute with column median
inf_counts = np.isinf(df_clean[all_features]).sum()
inf_cols   = inf_counts[inf_counts > 0]
if not inf_cols.empty:
    print("\nInf values replaced with column median:")
    for col, n in inf_cols.items():
        median = df_clean[col].replace([np.inf, -np.inf], np.nan).median()
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        print(f"  {col}: {n} inf value(s) → imputed with median {median:.3g}")

df_clean = df_clean.dropna()
dropped  = len(df) - len(df_clean)
if dropped:
    print(f"Dropped {dropped} rows with NaN in features or targets")
print(f"Rows for training : {len(df_clean):,}")

X = df_clean[all_features].values
Y = df_clean[target_names].values

# ── Cross-validation ───────────────────────────────────────────────────────────

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_results = []
coef_folds   = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # ElasticNetCV tunes alpha and l1_ratio jointly via inner CV on the training fold.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MultiOutputRegressor(
            ElasticNetCV(
                l1_ratio   = L1_RATIOS,
                alphas     = ALPHAS,    # int = size of auto-computed grid
                cv         = 3,
                max_iter   = 50000,
                tol        = 1e-4,
                n_jobs     = -1,        # parallelise over CV folds/l1_ratios
            ),
        ))
    ])
    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)

    fold_metrics = {"fold": fold}
    for i, name in enumerate(target_names):
        est       = pipeline.named_steps["model"].estimators_[i]
        r2        = r2_score(Y_test[:, i], Y_pred[:, i])
        rmse      = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
        pearson_r = np.corrcoef(Y_test[:, i], Y_pred[:, i])[0, 1]
        n_nonzero = np.sum(est.coef_ != 0)

        fold_metrics[f"r2_{name}"]       = r2
        fold_metrics[f"rmse_{name}"]     = rmse
        fold_metrics[f"pearson_{name}"]  = pearson_r
        fold_metrics[f"alpha_{name}"]    = est.alpha_
        fold_metrics[f"l1ratio_{name}"]  = est.l1_ratio_
        fold_metrics[f"nonzero_{name}"]  = n_nonzero

    fold_results.append(fold_metrics)
    print(f"  Fold {fold}: "
          + " | ".join(
              f"{n} R²={fold_metrics[f'r2_{n}']:.3f} "
              f"(α={fold_metrics[f'alpha_{n}']:.2g}, "
              f"l1={fold_metrics[f'l1ratio_{n}']:.2f}, "
              f"nz={fold_metrics[f'nonzero_{n}']})"
              for n in target_names))

    # Standardized coefficients averaged across folds
    feature_std = X_train.std(axis=0)
    fold_coefs  = [est.coef_ * feature_std
                   for est in pipeline.named_steps["model"].estimators_]
    coef_folds.append(np.stack(fold_coefs, axis=1))  # (n_features, n_outputs)

# ── Summary ────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(fold_results)

print("\n" + "=" * 65)
print(f"CROSS-VALIDATION SUMMARY (mean ± std across {N_SPLITS} folds, 80/20 splits)")
print("=" * 65)

for name in target_names:
    r2_v       = results_df[f"r2_{name}"]
    rmse_v     = results_df[f"rmse_{name}"]
    pearson_v  = results_df[f"pearson_{name}"]
    alpha_v    = results_df[f"alpha_{name}"]
    l1ratio_v  = results_df[f"l1ratio_{name}"]
    nonzero_v  = results_df[f"nonzero_{name}"]
    y_all      = Y[:, list(target_names).index(name)]
    null_rmse  = y_all.std()
    l1_pct     = l1ratio_v.mean() * 100
    l2_pct     = (1 - l1ratio_v.mean()) * 100

    print(f"\n  {name}")
    print(f"    R²              : {r2_v.mean():.3f} ± {r2_v.std():.3f}")
    print(f"    Pearson r       : {pearson_v.mean():.3f} ± {pearson_v.std():.3f}")
    print(f"    RMSE            : {rmse_v.mean():.3f} ± {rmse_v.std():.3f}")
    print(f"    Null RMSE       : {null_rmse:.3f}  (RMSE reduction = "
          f"{100*(1 - rmse_v.mean()/null_rmse):.1f}%)")
    print(f"    Alpha (λ)       : {alpha_v.mean():.3g}  "
          f"(range {alpha_v.min():.3g} – {alpha_v.max():.3g}  |  "
          f"log₁₀ {np.log10(alpha_v.min()):.1f} – {np.log10(alpha_v.max()):.1f})")
    print(f"    l1_ratio        : {l1ratio_v.mean():.3f} ± {l1ratio_v.std():.3f}  "
          f"→  {l1_pct:.0f}% L1 (Lasso) / {l2_pct:.0f}% L2 (Ridge)")
    print(f"    Non-zero coefs  : {nonzero_v.mean():.1f} ± {nonzero_v.std():.1f}  "
          f"out of {len(all_features)} features  "
          f"({100*nonzero_v.mean()/len(all_features):.1f}% retained)")

# ── Feature importance ─────────────────────────────────────────────────────────

mean_coefs = np.mean(coef_folds, axis=0)  # (n_features, n_outputs)

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE  (standardized coefficient = coef × feature_std)")
print("Interpretation: expected change in log-FC per 1-SD increase in feature")
print("Zero-coefficient features were zeroed out by L1 regularization.")
print("=" * 65)

for i, name in enumerate(target_names):
    coefs    = mean_coefs[:, i]
    nonzero  = np.sum(coefs != 0)
    order    = np.argsort(np.abs(coefs))[::-1]

    print(f"\n  {name}  —  top {TOP_N_FEATURES} features by |standardized coef|  "
          f"({nonzero} non-zero across folds)")
    print(f"  {'Rank':<5} {'Feature':<30} {'Std. Coef':>10}  {'Direction'}")
    print(f"  {'-'*5} {'-'*30} {'-'*10}  {'-'*9}")
    for rank, idx in enumerate(order[:TOP_N_FEATURES], 1):
        direction = "↑ positive" if coefs[idx] > 0 else "↓ negative"
        print(f"  {rank:<5} {all_features[idx]:<30} {coefs[idx]:>+10.4f}  {direction}")
