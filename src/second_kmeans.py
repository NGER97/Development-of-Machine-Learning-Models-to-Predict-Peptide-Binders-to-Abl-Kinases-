"""
Second-level K-means on clusters 2 & 3
Author: Yuhao Rao
"""

import os, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===== 路径 =====
CSV_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_tsne_n_iter=1000.csv"
OUT_DIR  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 读取并筛选 cluster 2 & 3 =====
cols_needed = ["peptide_sequence", "score", "tSNE1", "tSNE2", "cluster"]
df_all = pd.read_csv(CSV_PATH, usecols=cols_needed)
df_sub = df_all[df_all["cluster"].isin([2, 3])].copy()

# ===== 对每个旧簇单独做二次 K-means =====
def best_kmeans(X, k_min=2, k_max=10, seed=42):
    """返回 silhouette 最优的 KMeans 模型"""
    best_k, best_sil, best_model = k_min, -1, None
    for k in range(k_min, min(k_max, len(X)) + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(X)
        if k == 1:   # silhouette 无法计算 k=1，跳过
            continue
        sil = silhouette_score(X, km.labels_) 
        if sil > best_sil:
            best_k, best_sil, best_model = k, sil, km
    return best_model, best_k, best_sil

sub_labels = []
for old_c in [2, 3]:
    block = df_sub[df_sub["cluster"] == old_c]
    X      = block[["tSNE1", "tSNE2"]].values
    model, k, sil = best_kmeans(X)
    print(f"Cluster {old_c}: best k = {k}, silhouette = {sil:.3f}")
    # 生成唯一的二级标签：例如 2-0, 2-1, …  或 3-0, 3-1…
    block_labels = [f"{old_c}_{lbl}" for lbl in model.labels_]
    sub_labels.extend(block_labels)
    # 写回
    df_sub.loc[block.index, "sub_cluster"] = block_labels

# ===== 保存结果 =====
base_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
out_csv   = os.path.join(OUT_DIR, f"{base_name}_cluster2_3_subkmeans.csv")
df_sub.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"二次聚类结果已保存 → {out_csv}")
