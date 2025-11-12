import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

# ==== 配置 ====
PCA_CSV   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand\all_features_pca.csv"
OUT_DIR   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand\clusters"
SELECT    = "high"   # "high" 选高分(更深色)；"low" 选低分
Q         = 0.85     # 选分位（比如取分数最高的15%或最低的15%）
MIN_SAMPLES = 30     # OPTICS 的 min_samples（相当于 DBSCAN 的 minPts）
MIN_CLUSTER_SIZE = 100  # 认为有效簇的最小成员数
XI        = 0.05     # OPTICS 的xi参数，控制层级切割灵敏度

os.makedirs(OUT_DIR, exist_ok=True)

# ==== 读取 ====
df = pd.read_csv(PCA_CSV)   # 需要列：peptide_sequence, pred_score, PC1, PC2

# 1) 选“深色点”子集（按分位）
if SELECT == "high":
    thr = df["pred_score"].quantile(Q)
    sub = df[df["pred_score"] >= thr].copy()
else:
    thr = df["pred_score"].quantile(1 - Q)
    sub = df[df["pred_score"] <= thr].copy()

print(f"[INFO] selected {len(sub):,} points by {SELECT} score (threshold={thr:.4f})")

# 2) 标准化 PC 坐标（对 DBSCAN/OPTICS 稳定很重要）
X = sub[["PC1", "PC2"]].values
X = StandardScaler().fit_transform(X)

# 3) OPTICS 聚类（不需要手动设 eps，能处理变密度）
clu = OPTICS(min_samples=MIN_SAMPLES, xi=XI, min_cluster_size=MIN_CLUSTER_SIZE)
clu.fit(X)

labels = clu.labels_        # -1 是噪声
sub["cluster"] = labels

# 4) 导出簇统计
stats = (sub
         .groupby("cluster")
         .agg(
             n=("peptide_sequence", "size"),
             mean_score=("pred_score", "mean"),
             median_score=("pred_score", "median"),
             pc1_center=("PC1", "median"),
             pc2_center=("PC2", "median"),
             pc1_min=("PC1", "min"), pc1_max=("PC1", "max"),
             pc2_min=("PC2", "min"), pc2_max=("PC2", "max"),
         )
         .sort_values(["n","median_score"], ascending=[False, False])
         .reset_index())

stats_path = os.path.join(OUT_DIR, f"optics_{SELECT}_q{Q}_cluster_stats.csv")
stats.to_csv(stats_path, index=False)
print(f"[SAVE] cluster stats → {stats_path}")

# 5) 导出每个簇的成员（方便后续筛序列）
members_path = os.path.join(OUT_DIR, f"optics_{SELECT}_q{Q}_members.csv")
sub[["cluster","peptide_sequence","pred_score","PC1","PC2"]].to_csv(members_path, index=False)
print(f"[SAVE] cluster members → {members_path}")

# 6) 画图（子集上）
plt.figure(figsize=(7,6))
# 给簇着色；噪声 -1 用灰色
from matplotlib import colormaps
cmap = colormaps.get_cmap("tab20")
colors = cmap(np.linspace(0, 1, 200))           # (200,4)

cond = (labels == -1)[:, None]                  # (N,1)
gray = np.array([0.6, 0.6, 0.6, 0.4])[None, :]  # (1,4)
clr  = np.where(cond, gray, colors[(labels % 200).astype(int)])

plt.scatter(sub["PC1"], sub["PC2"], s=4, c=clr, edgecolors='none')


plt.scatter(sub["PC1"], sub["PC2"], s=4, c=clr, edgecolors='none')
plt.title(f"OPTICS clusters on {SELECT} score subset (q={Q}, xi={XI})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, f"optics_{SELECT}_q{Q}_scatter.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[SAVE] plot → {fig_path}")
