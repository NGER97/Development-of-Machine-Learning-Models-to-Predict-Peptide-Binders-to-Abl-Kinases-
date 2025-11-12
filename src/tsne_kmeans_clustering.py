"""
tsne_kmeans_clustering.py
Author: Yuhao Rao
t-SNE (n_iter=1000) + K-means clustering on ProtBERT CLS embeddings.
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ========= 路径 =========
FEAT_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_minCount1_R1-4_ER&Score_protBERT.parquet"
RAW_CSV   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
OUT_DIR   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE"
BASE_NAME = "BB_protBERT_minCount1_R1-4_"   # 统一文件名前缀

os.makedirs(OUT_DIR, exist_ok=True)

# ========= 读取特征 =========
df = (pd.read_parquet(FEAT_PATH)
      if FEAT_PATH.endswith(".parquet") else pd.read_csv(FEAT_PATH))

# 补齐 score
if "score" not in df.columns:
    df = df.merge(pd.read_csv(RAW_CSV, usecols=["peptide_sequence", "score"]),
                  on="peptide_sequence", how="left")

feat_cols = [c for c in df.columns if c.startswith("protBERT_")]
X = df[feat_cols].astype(np.float32).values
scores = df["score"].values

# ========= 1) PCA → 50 维 =========
X_std = StandardScaler().fit_transform(X)
X_pca50 = PCA(n_components=50, random_state=42).fit_transform(X_std)

# ========= 2) t-SNE =========
tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
        metric="euclidean",
        n_iter=1000
      )
X_tsne = tsne.fit_transform(X_pca50)

# ========= 3) 自动选 K + K-means =========
best_k, best_score = 2, -1
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_tsne)
    sil = silhouette_score(X_tsne, km.labels_)
    if sil > best_score:
        best_k, best_score = k, sil
kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42).fit(X_tsne)
labels = kmeans.labels_
print(f"最佳簇数 K = {best_k}, silhouette = {best_score:.3f}")

# ========= 4) 保存 CSV =========
df_out = df[["peptide_sequence", "score"]].copy()
df_out["tSNE1"], df_out["tSNE2"] = X_tsne[:, 0], X_tsne[:, 1]
df_out["cluster"] = labels
csv_path = os.path.join(OUT_DIR, f"{BASE_NAME}.csv")
df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"结果 CSV 保存至: {csv_path}")

# ========= 绘图 =========
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(8, 6))          # ← 建议这样创建 fig / ax

# —— 1) 预处理 —— #
score_min, score_max = scores.min(), scores.max()
score_norm = (scores - score_min) / (score_max - score_min)

base_cmap = plt.colormaps.get_cmap("tab20", best_k)  # ← 新 API，避免警告
base_colors = [base_cmap(i) for i in range(best_k)]

def blend_with_white(rgb, alpha):
    white = np.array([1.0, 1.0, 1.0])
    return tuple(white * (1 - alpha) + np.array(rgb[:3]) * alpha) + (1.0,)

point_colors = [
    blend_with_white(base_colors[lab], score_norm[i])
    for i, lab in enumerate(labels)
]

# —— 3) 散点 —— #
ax.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=point_colors, s=6, alpha=0.9, linewidths=0
)
ax.set_title(f"t-SNE (n_iter=1000) + K-means (K={best_k})")
ax.set_xlabel("tSNE1"); ax.set_ylabel("tSNE2")

# —— 4) legend —— #
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=base_colors[k], markersize=6,
                  label=f'Cluster {k}')
           for k in range(best_k)]
ax.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left')

# —— 5) colorbar —— #
norm = plt.Normalize(score_min, score_max)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, pad=0.02).set_label("Score (darker = higher)")

fig.tight_layout()
png_path = os.path.join(OUT_DIR, f"{BASE_NAME}.png")
fig.savefig(png_path, dpi=300); plt.show()
print(f"聚类图保存至: {png_path}")
