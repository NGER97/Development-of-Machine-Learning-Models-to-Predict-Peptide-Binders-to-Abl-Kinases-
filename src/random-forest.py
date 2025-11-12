"""
Random-Forest (regression) on ProtBERT embeddings
Combine two training sets before fitting
Author: Yuhao Rao
"""

import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# ========= 1. 路径 =========
FEATURE_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\regression\BB_add_pseudocount_R1toR4_significant_ALLfeatures.csv"
SCORE_PATH   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR4_significant_ER&Score.csv"
MODEL_OUT    = r"D:\Me\IMB\Data\Yuhao\models\rf_R1toR4_significant.pkl"

# ========= 2. 读取并粘贴 score（按行对齐） =========
import pandas as pd, numpy as np, joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df_feat  = pd.read_csv(FEATURE_PATH)          # 或 .read_parquet
df_score = pd.read_csv(SCORE_PATH, usecols=["score"])

if len(df_feat) != len(df_score):
    raise ValueError("行数不一致，无法按顺序合并！")

df_feat["score"] = df_score["score"].values
print(f"[INFO] 样本数: {len(df_feat)}")

# ========= 3. 特征矩阵 / 标签 =========
drop_cols = {"Protein", "protein", "Score", "score", "class"}
feat_cols = [c for c in df_feat.columns if c not in drop_cols]
X = df_feat[feat_cols].astype(np.float32).values
y = df_feat["score"].values
print(f"[INFO] 使用 {len(feat_cols)} 个特征")

# ========= 4. 建模 & 交叉验证 =========
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [None, 10, 30],
    "max_features": ["sqrt", 0.3, 0.5],
}
gcv = GridSearchCV(rf, param_grid, cv=cv,
                   scoring="neg_mean_squared_error",
                   n_jobs=-1, verbose=2)
gcv.fit(X, y)
best_model = gcv.best_estimator_

# ========= 5. 训练全集 + 输出指标 =========
best_model.fit(X, y)
y_pred = best_model.predict(X)
print("\n=== Train-set metrics ===")
print("RMSE :", np.sqrt(mean_squared_error(y, y_pred)))
print("MAE  :", mean_absolute_error(y, y_pred))
print("R²   :", r2_score(y, y_pred))

# ========= 6. 保存模型 =========
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump({"model": best_model, "feat_cols": feat_cols}, MODEL_OUT)
print(f"[SAVE] 模型已写入 → {MODEL_OUT}")
