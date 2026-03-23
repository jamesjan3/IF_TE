import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor


#Import merged dataframe

script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(script_dir, "..", "merged_features.csv")
df = pd.read_csv(INPUT_FILE)

ID_COL = "sample_id" if "sample_id" in df.columns else None

#Define the possible feature subsets

feature_subsets = {
    "min_dg": ['min_dg_5p60', 'min_dg_startCentered60'],

    "utr5_nt": ["utr5_A_pct", "utr5_C_pct", "utr5_G_pct", "utr5_T_pct"],
    "cds_nt":  ["cds_A_pct", "cds_C_pct", "cds_G_pct", "cds_T_pct"],
    "utr3_nt": ["utr3_A_pct", "utr3_C_pct", "utr3_G_pct", "utr3_T_pct"],

    "dicodon": ["dicodon_count", "dicodon_density"],

    "cds_wobble_nt": [
        "cds_wobble_A_pct",
        "cds_wobble_C_pct",
        "cds_wobble_G_pct",
        "cds_wobble_T_pct",
    ],

    "lengths": [
        "tx_length",
        "utr5_fraction",
        "cds_fraction",
        "utr3_fraction",
    ],

    "kozak": [
        "-3_A", "-3_C", "-3_G",
        "-2_A", "-2_C", "-2_G",
        "-1_A", "-1_C", "-1_G",
        "+4_A", "+4_C", "+4_G",
        "+5_A", "+5_C", "+5_G",
    ],
}

feature_subsets["codon_freq"] = [col for col in df.columns if col.startswith("codon_")]
feature_subsets["aa_freq"] = [col for col in df.columns if col.startswith("aa_")]

feature_subsets["utr3_k2"] = [col for col in df.columns if col.startswith("utr3_k2")]
feature_subsets["utr3_k3"] = [col for col in df.columns if col.startswith("utr3_k3")]
feature_subsets["utr3_k4"] = [col for col in df.columns if col.startswith("utr3_k4")]

feature_subsets["utr5_k2"] = [col for col in df.columns if col.startswith("utr5_k2")]
feature_subsets["utr5_k3"] = [col for col in df.columns if col.startswith("utr5_k3")]
feature_subsets["utr5_k4"] = [col for col in df.columns if col.startswith("utr5_k4")]

feature_subsets["cds_k2"] = [col for col in df.columns if col.startswith("cds_k2")]
feature_subsets["cds_k3"] = [col for col in df.columns if col.startswith("cds_k3")]
feature_subsets["cds_k4"] = [col for col in df.columns if col.startswith("cds_k4")]

#Define the target values (We are predicting the TE differences for the different depletion conditions)


df["eIF3d_TE_diff"] = df["eIF3d_depletion_logTE"] - df["eIF3d_control_logTE"]
df["eIF4e_TE_diff"] = df["eIF4e_depletion_logTE"] - df["eIF4e_control_logTE"]

TARGETS = {
    "eIF3d_model": "eIF3d_TE_diff",
    "eIF4e_model": "eIF4e_TE_diff"
}


#Define functions for combining feature subsets as well as performance metrics 

def combine_features(*groups):
    combined = []
    seen = set()
    for group in groups:
        for col in group:
            if col not in seen:
                combined.append(col)
                seen.add(col)
    return combined

def validate_feature_subset(df, feature_cols, subset_name):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Subset '{subset_name}' has missing columns: {missing}")
    if len(feature_cols) == 0:
        raise ValueError(f"Subset '{subset_name}' is empty.")

def clean_feature_matrix(data, feature_cols):
    X = data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)

    return X

from scipy.stats import pearsonr, spearmanr

def safe_corr(func, y_true, y_pred):
    try:
        return float(func(y_true, y_pred)[0])
    except:
        return np.nan


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": safe_corr(pearsonr, y_true, y_pred),
        "Spearman_r": safe_corr(spearmanr, y_true, y_pred),
    }


#Build the feature subsets as described in the document

feature_subsets["Elongation_dynamics"] = combine_features(
    feature_subsets["cds_wobble_nt"],
    feature_subsets["kozak"],
    feature_subsets["min_dg"],
)

feature_subsets["Elongation_dynamics_w_dicodon"] = combine_features(
    feature_subsets["cds_wobble_nt"],
    feature_subsets["kozak"],
    feature_subsets["min_dg"],
    feature_subsets["dicodon"],
)

three_prime_length_features = [c for c in ["tx_length", "cds_fraction"] if c in df.columns]

feature_subsets["3_prime_dynamics"] = combine_features(
    three_prime_length_features,
    feature_subsets["cds_nt"],
    feature_subsets["utr3_nt"],
)

non_feature_cols = {
    "eIF3d_control_logTE",
    "eIF3d_depletion_logTE",
    "eIF4e_control_logTE",
    "eIF4e_depletion_logTE",
    "eIF3d_TE_diff",
    "eIF4e_TE_diff",
}
if ID_COL:
    non_feature_cols.add(ID_COL)

numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

feature_subsets["full_model"] = [
    col for col in numeric_cols if col not in non_feature_cols
]




feature_subsets["basic"] = combine_features(
    feature_subsets["lengths"],
    feature_subsets["utr5_nt"],
    feature_subsets["cds_nt"],
    feature_subsets["utr3_nt"],
)

feature_subsets["codon_aa_lengths_kmers"] = combine_features(
    feature_subsets["codon_freq"],
    feature_subsets["aa_freq"],
    feature_subsets["lengths"],
    feature_subsets["utr5_k2"], feature_subsets["utr5_k3"], feature_subsets["utr5_k4"],
    feature_subsets["cds_k2"],  feature_subsets["cds_k3"],  feature_subsets["cds_k4"],
    feature_subsets["utr3_k2"], feature_subsets["utr3_k3"], feature_subsets["utr3_k4"],
    feature_subsets["utr5_nt"], feature_subsets["cds_nt"], feature_subsets["utr3_nt"],
)

#Choose which feature subsets to run

SUBSETS_TO_RUN = [
    "Elongation_dynamics",
    "Elongation_dynamics_w_dicodon",
    "3_prime_dynamics",
    "full_model",
    "basic",
    "codon_aa_lengths_kmers",
]

#Define the lgbm search function with a parameter space and the number of splits (5), choose model that returns the best r2 score

def make_lgbm_search(random_state=42, n_splits=5, n_iter=10, search_jobs=4):
    base_model = LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_jobs=1,
        verbosity=-1
    )

    param_dist = {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [7, 15, 31, 63, 127],
        "max_depth": [-1, 3, 5, 7, 10, 15],
        "min_child_samples": [5, 10, 20, 30, 50, 100],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0, 5.0, 10.0],
        "reg_lambda": [0.0, 0.01, 0.1, 1.0, 5.0, 10.0],
    }

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        refit=True,
        verbose=1,
        random_state=random_state,
        n_jobs=search_jobs,
        return_train_score=True,
        error_score="raise"
    )

    return search

#Define the function for training on the feature subsets and make it save the results to its own directory

def train_one_target(
    data,
    feature_cols,
    target_col,
    output_dir,
    test_size=0.2,
    random_state=42,
    n_splits=5,
    n_iter=10,
    search_jobs=4
):
    os.makedirs(output_dir, exist_ok=True)

    X = clean_feature_matrix(data, feature_cols)
    y = data[target_col].copy()

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    search = make_lgbm_search(
        random_state=random_state,
        n_splits=n_splits,
        n_iter=n_iter,
        search_jobs=search_jobs
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_test_pred = best_model.predict(X_test)
    test_metrics = regression_metrics(y_test, y_test_pred)

    best_cv_r2 = float(search.best_score_)

    cv_results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    pd.DataFrame({
        "y_true_test": y_test.reset_index(drop=True),
        "y_pred_test": pd.Series(y_test_pred)
    }).to_csv(os.path.join(output_dir, f"{target_col}_test_predictions.csv"), index=False)

    cv_results.to_csv(os.path.join(output_dir, f"{target_col}_cv_search_results.csv"), index=False)
    importance_df.to_csv(os.path.join(output_dir, f"{target_col}_feature_importance.csv"), index=False)

    summary = {
        "target_col": target_col,
        "n_features": len(feature_cols),
        "features_used": feature_cols,
        "n_samples_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_params": search.best_params_,
        "best_cv_r2": best_cv_r2,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(output_dir, f"{target_col}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "search": search,
        "best_model": best_model,
        "best_params": search.best_params_,
        "best_cv_r2": best_cv_r2,
        "test_metrics": test_metrics,
        "feature_importance": importance_df,
        "cv_results": cv_results,
    }

# Run the function for all feature subsets

all_results = []
base_output_dir = "lgbm_results_final"

for subset_name in SUBSETS_TO_RUN:
    FEATURES = feature_subsets[subset_name]
    validate_feature_subset(df, FEATURES, subset_name)

    required_cols = FEATURES + list(TARGETS.values())
    if ID_COL:
        required_cols.append(ID_COL)

    data = df[required_cols].copy()

    print(f"\n{'#'*80}")
    print(f"Running feature subset: {subset_name}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Rows before target filtering / cleaning: {len(data)}")
    print(f"{'#'*80}")

    for model_name, target_col in TARGETS.items():
        print(f"\n{'='*70}")
        print(f"Training model: {model_name}")
        print(f"Target: {target_col}")
        print(f"Feature subset: {subset_name}")
        print(f"{'='*70}")

        output_dir = os.path.join(base_output_dir, subset_name, model_name)

        result = train_one_target(
            data=data,
            feature_cols=FEATURES,
            target_col=target_col,
            output_dir=output_dir,
            test_size=0.2,
            random_state=42,
            n_splits=5,
            n_iter=10,
            search_jobs=4
        )

        print("\nBest hyperparameters:")
        print(result["best_params"])

        print(f"\nBest 5-fold CV R²: {result['best_cv_r2']:.4f}")

        print("\nHeld-out test metrics:")
        for k, v in result["test_metrics"].items():
            print(f"{k}: {v:.4f}")

        print("\nTop features:")
        print(result["feature_importance"].head(10).to_string(index=False))

        all_results.append({
    "feature_subset": subset_name,
    "model_name": model_name,
    "target_col": target_col,
    "n_features": len(FEATURES),

    # CV
    "best_cv_r2": result["best_cv_r2"],

    # Test metrics
    "test_r2": result["test_metrics"]["R2"],
    "test_rmse": result["test_metrics"]["RMSE"],
    "test_mae": result["test_metrics"]["MAE"],
    "pearson_r": result["test_metrics"]["Pearson_r"],
    "spearman_r": result["test_metrics"]["Spearman_r"],

    "best_params": json.dumps(result["best_params"]),
})


#Summarize all results in a CSV file

summary_df = pd.DataFrame(all_results).sort_values(
    by=["target_col", "test_r2"],
    ascending=[True, False]
)

os.makedirs(base_output_dir, exist_ok=True)
summary_df.to_csv(os.path.join(base_output_dir, "all_model_results_summary.csv"), index=False)

print("\nFinished training all requested feature subsets.")
print("\nOverall performance summary:")
print(summary_df.to_string(index=False))