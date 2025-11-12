"""
PCA + RandomForest 5-fold: Evaluate both RMSE and R²
Author: Yuhao Rao
"""

import os, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# ---------- Data ----------
FEATURE_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score_protBERT.parquet"
RAW_CSV      = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_ER&Score.csv"

df = pd.read_parquet(FEATURE_PATH)
if "score" not in df.columns:
    df = df.merge(
        pd.read_csv(RAW_CSV, usecols=["peptide_sequence", "score"]),
        on="peptide_sequence",
        how="left"
    )

feat_cols = [c for c in df.columns if c.startswith("protBERT_")]
X = df[feat_cols].astype(np.float32).values
y = df["score"].values

# ---------- Compute 90/95% variance dimensions ----------
pca_full = PCA().fit(X)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
k90 = np.where(cum_var >= 0.90)[0][0] + 1
k95 = np.where(cum_var >= 0.95)[0][0] + 1
candidate_dims = sorted(set([k90, 50, k95, 100, 150]))
print("Candidate dimensions:", candidate_dims)

# ---------- Pipeline + GridSearch ----------
pipe = Pipeline([
    ("pca", PCA(random_state=42)),
    ("rf",  RandomForestRegressor(
                n_estimators=300,
                n_jobs=-1,
                random_state=42))
])

# Custom RMSE scorer (higher positive scores are better)
def rmse(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))
rmse_scorer = make_scorer(rmse)

scoring = {
    "RMSE": rmse_scorer,    # Negate RMSE so that higher scores are better
    "R2":   "r2"
}

gcv = GridSearchCV(
    estimator=pipe,
    param_grid={"pca__n_components": candidate_dims},
    scoring=scoring,
    refit="RMSE",           # Use RMSE as the criterion to select the final model
    cv=KFold(5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=2
)

gcv.fit(X, y)

# ---------- Print RMSE & R² for each k ----------
for i in range(len(gcv.cv_results_["params"])):
    k  = gcv.cv_results_["params"][i]["pca__n_components"]
    rm = -gcv.cv_results_["mean_test_RMSE"][i]        # Restore positive RMSE
    sd =  gcv.cv_results_["std_test_RMSE"][i]
    r2 =  gcv.cv_results_["mean_test_R2"][i]
    print(f"k={k:<4}  RMSE={rm:.4f}±{sd:.4f}   R²={r2:.4f}")

best_k  = gcv.best_params_["pca__n_components"]
best_rm = -gcv.best_score_
best_r2 = gcv.cv_results_["mean_test_R2"][list(gcv.cv_results_["params"]).index({"pca__n_components": best_k})]
print(f"\nBest dimension: {best_k}   RMSE={best_rm:.4f}   R²={best_r2:.4f}")

# ---------- Baseline (mean prediction) ----------
y_mean = np.full_like(y, y.mean())
baseline_rmse = np.sqrt(mean_squared_error(y, y_mean))
baseline_r2   = r2_score(y, y_mean)  # Always 0
print(
    "\nBaseline  RMSE =", round(baseline_rmse, 4),
    "  R² =", round(baseline_r2, 4)
)
