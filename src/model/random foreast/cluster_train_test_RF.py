import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# === 路径，按需修改 ===
EMBED_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score_protBERT.parquet"
CLUSTER_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_tsne_n_iter=1000.csv"
RAW_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_ER&Score.csv"

# ---------- 1. 读取 & 合并 ----------
df_feat = pd.read_parquet(EMBED_PATH)
df_cluster = pd.read_csv(CLUSTER_CSV, usecols=["peptide_sequence", "cluster"])
df_score   = pd.read_csv(RAW_CSV,   usecols=["peptide_sequence", "score"])

df = df_feat.merge(df_cluster, on="peptide_sequence").merge(df_score, on="peptide_sequence")
feat_cols = [c for c in df.columns if c.startswith("protBERT_")]

X = df[feat_cols].astype(np.float32).values
y = df["score"].values
groups = df["cluster"].values
clusters = np.sort(df["cluster"].unique())

print("样本:", len(y), "  特征维度:", X.shape[1], "  簇数:", len(clusters))

# ---------- 2. 过滤掉 cluster 4 & 5 ----------
drop_set = {4, 5}
df = df[~df["cluster"].isin(drop_set)].reset_index(drop=True)

feat_cols = [c for c in df.columns if c.startswith("protBERT_")]
X = df[feat_cols].astype(np.float32).values
y = df["score"].values
groups  = df["cluster"].values
clusters = np.sort(df["cluster"].unique())

print("剩余簇：", clusters, "   样本:", len(y))

# ---------- 3. 单簇测试 ----------
test_cluster = 3            # ← 想评估的簇编号
mask_test = (groups == test_cluster)
X_train, y_train = X[~mask_test], y[~mask_test]
X_test,  y_test  = X[mask_test],  y[mask_test]

rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse_single = np.sqrt(mean_squared_error(y_test, y_pred))
r2_single   = r2_score(y_test, y_pred)
print(f"\n=== 单簇评估：cluster {test_cluster} ===")
print("RMSE =", round(rmse_single,4), "   R² =", round(r2_single,4),
      f"   (n_test = {len(y_test)})")

# ---------- 4. Leave-One-Cluster-Out CV ----------
rmse_list, r2_list = [], []
gkf = GroupKFold(n_splits=len(clusters))

for train_idx, test_idx in gkf.split(X, y, groups):
    c_label = groups[test_idx[0]]
    rf.fit(X[train_idx], y[train_idx])
    y_hat = rf.predict(X[test_idx])
    rmse = np.sqrt(mean_squared_error(y[test_idx], y_hat))
    r2   = r2_score(y[test_idx], y_hat)
    rmse_list.append(rmse); r2_list.append(r2)
    print(f"cluster {c_label:<2}  RMSE={rmse:.4f}  R²={r2:.4f}  (n={len(test_idx)})")

print("\n=== LOCO-CV 结果 ===")
print("平均 RMSE =", round(np.mean(rmse_list),4), "±", round(np.std(rmse_list),4))
print("平均 R²   =", round(np.mean(r2_list),4), "±", round(np.std(r2_list),4))
