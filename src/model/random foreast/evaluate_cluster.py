#!/usr/bin/env python
# evaluate_by_cluster_LOCO.py  (Leave-One-Cluster-Out)
# ===========================================================
# 1. 依赖 & 路径
# ===========================================================
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

EMBED_PATH  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_minCount1_R1-4_ER&Score_protBERT.parquet"
RAW_CSV     = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
CLUSTER_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_minCount1_R1-4_.csv"
OUT_DIR     = r"D:\Me\IMB\Data\Yuhao\outputs\by_cluster_eval"
os.makedirs(OUT_DIR, exist_ok=True)

RF_PARAMS = dict(max_depth=10, max_features=0.3, n_estimators=500, random_state=42, n_jobs=-1)

# ===========================================================
# 2. 读取数据并合并
# ===========================================================
print("Loading data ...")
df_embed   = pd.read_parquet(EMBED_PATH)           # protBERT 特征
df_score   = pd.read_csv(RAW_CSV)                  # true Weighted ER
df_cluster = pd.read_csv(CLUSTER_CSV)[["peptide_sequence", "cluster"]]

df = (df_embed
      .merge(df_score[["peptide_sequence", "score"]], on="peptide_sequence", how="inner")
      .merge(df_cluster,                           on="peptide_sequence", how="inner"))

print(f"Total sequences with cluster label & score: {len(df)}")

feat_cols = [c for c in df_embed.columns if c.startswith("protBERT_")]

# ===========================================================
# 3–5. Leave-One-Cluster-Out 训练 + 评估
# ===========================================================
metrics      = []
pred_dfs     = []            # 保存各簇预测结果，方便后续合并
unique_clust = sorted(df["cluster"].unique())

for clust_id in unique_clust:
    print(f"\n=== Cluster {clust_id} as test set ===")
    test_df  = df[df["cluster"] == clust_id].reset_index(drop=True)
    train_df = df[df["cluster"] != clust_id].reset_index(drop=True)

    X_train, y_train = train_df[feat_cols].values, train_df["score"].values
    X_test,  y_test  = test_df[feat_cols].values,  test_df["score"].values

    # 重新训练 RandomForest
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    test_df = test_df.assign(pred=y_pred)      # 保存预测

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    metrics.append({"cluster": clust_id,
                    "n_test": len(test_df),
                    "RMSE": rmse,
                    "R2": r2})

    pred_dfs.append(test_df[["peptide_sequence", "cluster", "score", "pred"]])

# ===========================================================
# 6. 汇总指标 & 可视化
# ===========================================================
metrics_df = (pd.DataFrame(metrics)
              .sort_values("cluster")
              .reset_index(drop=True))

print("\nPer-cluster metrics (test on that cluster):")
print(metrics_df.to_string(index=False, float_format="%.4f"))

# -- 条形图 --
plt.figure(figsize=(6,3))
sns.barplot(data=metrics_df, x="cluster", y="RMSE", color="steelblue")
plt.title("RMSE (cluster = test set)"); plt.ylabel("RMSE"); plt.xlabel("Held-out cluster")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
sns.barplot(data=metrics_df, x="cluster", y="R2", color="seagreen")
plt.title("R² (cluster = test set)"); plt.ylabel("R²"); plt.xlabel("Held-out cluster")
plt.ylim(0, 1)
plt.tight_layout(); plt.show()

# ===========================================================
# 7. 保存预测明细 & 汇总表
# ===========================================================
pred_all = pd.concat(pred_dfs, ignore_index=True)
pred_path = os.path.join(OUT_DIR, "BB_protBERT_minCount1_R1-4_pred_by_cluster.csv")
pred_all.to_csv(pred_path, index=False)
print("\nMerged prediction file saved to:", pred_path)

metrics_path = os.path.join(OUT_DIR, "metrics_by_cluster.csv")
metrics_df.to_csv(metrics_path, index=False)
print("Metrics summary saved to:", metrics_path)
