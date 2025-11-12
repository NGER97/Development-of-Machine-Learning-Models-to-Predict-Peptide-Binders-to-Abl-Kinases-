from sklearn.metrics import mean_squared_error, r2_score
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

EMBED_PATH  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score_protBERT.parquet"
RAW_CSV     = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_wRoundER.csv"
CLUSTER_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_tsne_n_iter=1000.csv"
MODEL_OUT   = r"D:\Me\IMB\Data\Yuhao\models\rf_protBERT_score.pkl"

# ===========================================================
# 2. 读取数据并合并
# ===========================================================
print("Loading data ...")
df_embed   = pd.read_parquet(EMBED_PATH)           # protBERT 特征
df_score   = pd.read_csv(RAW_CSV)                  # true Weighted ER
df_cluster = pd.read_csv(CLUSTER_CSV)

# 这里只保留 peptide_sequence 与 cluster 两列，避免混淆 cluster CSV 里的旧 score
df_cluster = df_cluster[["peptide_sequence", "cluster"]]

# 根据 peptide_sequence 多表合并
df = (df_embed
      .merge(df_score[["peptide_sequence", "score"]],
             on="peptide_sequence", how="inner")
      .merge(df_cluster, on="peptide_sequence", how="inner"))

print(f"Sequences with cluster label & score: {len(df)}")

feat_cols = [c for c in df_embed.columns if c.startswith("protBERT_")]
X = df[feat_cols].values.astype("float32")
y_true = df["score"].values.astype("float32")

model = joblib.load(MODEL_OUT)
y_pred = model.predict(X)

df["pred"] = y_pred    # 方便后面按簇分组

metrics = []
for clust_id, sub in df.groupby("cluster"):
    rmse = np.sqrt(mean_squared_error(sub["score"], sub["pred"]))
    r2   = r2_score(sub["score"], sub["pred"])
    metrics.append({"cluster": clust_id, "n": len(sub), "RMSE": rmse, "R2": r2})

metrics_df = pd.DataFrame(metrics).sort_values("cluster")
print("\nPer-cluster metrics")
print(metrics_df.to_string(index=False, float_format="%.4f"))


# ---------------------------------------------
# 1. 选择高分段序列
# ---------------------------------------------
# ① 用绝对阈值
thr_abs = 0.5
df_high_abs = df[df["score"] >= thr_abs]

# ② 用分位阈值 (Top 10 %)
q = 0.90
thr_quant = df["score"].quantile(q)
df_high_q = df[df["score"] >= thr_quant]

print(f"High-score ABS  (thr={thr_abs}) : {len(df_high_abs)} seqs")
print(f"High-score TOP{int((1-q)*100)}% (thr={thr_quant:.3f}): {len(df_high_q)} seqs")

# 选择其中一种作为后续评估对象
df_eval = df_high_q          # or df_high_abs

# ---------------------------------------------
# 2. 整体高分段指标
# ---------------------------------------------
rmse_hi = np.sqrt(mean_squared_error(df_eval["score"], df_eval["pred"]))
r2_hi   = r2_score(df_eval["score"], df_eval["pred"])
print(f"\n[High-score overall]  RMSE={rmse_hi:.4f}   R²={r2_hi:.3f}")

# ---------------------------------------------
# 3. 逐簇高分段 RMSE / R²
# ---------------------------------------------
metrics_hi = []
for cid, sub in df_eval.groupby("cluster"):
    rmse = np.sqrt(mean_squared_error(sub["score"], sub["pred"]))
    r2   = r2_score(sub["score"], sub["pred"])
    metrics_hi.append({"cluster": cid, "n": len(sub), "RMSE_hi": rmse, "R2_hi": r2})

metrics_hi_df = pd.DataFrame(metrics_hi).sort_values("cluster")
print("\nPer-cluster high-score metrics")
print(metrics_hi_df.to_string(index=False, float_format="%.4f"))

# ---------------------------------------------
# 4. Top-K Recall / Precision (簇内 10 %)
# ---------------------------------------------
topk_metrics = []
for cid, sub in df.groupby("cluster"):
    k = max(1, int(0.10 * len(sub)))              # 取簇内 10 %
    
    true_top = sub.nlargest(k, "score")["peptide_sequence"]
    pred_top = sub.nlargest(k, "pred")["peptide_sequence"]
    
    overlap = len(set(true_top) & set(pred_top))
    precision = overlap / k
    recall    = overlap / k                       # k = |真实Top|，所以 precision=recall
    
    topk_metrics.append({"cluster": cid, "k": k,
                         "precision@10%": precision,
                         "recall@10%": recall})

topk_df = pd.DataFrame(topk_metrics).sort_values("cluster")
print("\nPer-cluster Top-10% Precision / Recall")
print(topk_df.to_string(index=False, float_format="%.3f"))
