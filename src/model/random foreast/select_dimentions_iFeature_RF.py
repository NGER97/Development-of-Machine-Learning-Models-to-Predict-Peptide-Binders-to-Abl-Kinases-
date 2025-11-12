"""
PCA(多维) + RandomForest on iFeatures data
Author: Yuhao Rao
"""

import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# ---------- 路径 ----------
FEAT_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score-final_noconst.csv"
RAW_CSV  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_ER&Score.csv"

# ① 读入 iFeature   （dtype float32 节省内存）
df_feat  = pd.read_csv(FEAT_CSV, dtype=np.float32)
df_feat.insert(0, "row_id", range(len(df_feat)))     # 行号当键

# ② 读入 score.csv，只保留标签
df_score = pd.read_csv(RAW_CSV, usecols=["score"])
df_score.insert(0, "row_id", range(len(df_score)))

# ③ inner merge（按行号）
df = df_feat.merge(df_score, on="row_id")
df = df.drop(columns=["row_id"])          # 行号列用完可删


X = df.drop(columns=["score"]).values.astype(np.float32)
y = df["score"].values
print("样本数:", X.shape[0], "  特征维数:", X.shape[1])

# ---------- 先整体 PCA 拟合，拿累积方差 ----------
pca_full = PCA().fit(X)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
k90 = np.where(cum_var >= 0.90)[0][0] + 1
k95 = np.where(cum_var >= 0.95)[0][0] + 1
candidate_dims = sorted(set([100, 300, 500, k90, k95]))
print("候选维数:", candidate_dims)

# ---------- Pipeline ----------
pipe = Pipeline([
    ("sc",  StandardScaler(with_mean=False)),   # 稀疏很大时可改 MaxAbsScaler
    ("pca", PCA(random_state=42)),
    ("rf",  RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=42))
])

def rmse(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))
scoring = {"RMSE": make_scorer(rmse), "R2": "r2"}

gcv = GridSearchCV(
        pipe,
        param_grid={"pca__n_components": candidate_dims},
        scoring=scoring,
        refit="RMSE",
        cv=KFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2)

gcv.fit(X, y)

# ---------- 输出结果 ----------
print("\n---- 5-fold 结果 ----")
for mean_rmse, std_rmse, mean_r2, params in zip(
        -gcv.cv_results_["mean_test_RMSE"],
         gcv.cv_results_["std_test_RMSE"],
         gcv.cv_results_["mean_test_R2"],
         gcv.cv_results_["params"]):
    k = params["pca__n_components"]
    print(f"k={k:<4}  RMSE={mean_rmse:.4f}±{std_rmse:.4f}   R²={mean_r2:.4f}")

best_k   = gcv.best_params_["pca__n_components"]
best_rm  = -gcv.best_score_
best_r2  = gcv.cv_results_["mean_test_R2"][list(gcv.cv_results_["params"]).index(
             {"pca__n_components": best_k})]

print(f"\n最佳维数: {best_k}   RMSE={best_rm:.4f}   R²={best_r2:.4f}")

# ---------- Baseline ----------
y_mean = np.full_like(y, y.mean())
baseline_rmse = np.sqrt(mean_squared_error(y, y_mean))
baseline_r2   = r2_score(y, y_mean)
print("\nBaseline  RMSE =", round(baseline_rmse,4),
      "  R² =", round(baseline_r2,4))
