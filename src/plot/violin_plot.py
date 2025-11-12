"""
Violin plot of score distribution by cluster
Author: Yuhao Rao
"""

import os, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ===== 路径 =====
CSV_TSNE  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_minCount1_R1-4_.csv"
CSV_SCORE = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
OUT_DIR   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE"

# ===== 读取 =====
df_cluster = pd.read_csv(CSV_TSNE,  usecols=["peptide_sequence", "cluster"])
df_score   = pd.read_csv(CSV_SCORE, usecols=["peptide_sequence", "score"])

df = df_cluster.merge(df_score, on="peptide_sequence", how="inner")

df = df.dropna(subset=["score"])

clusters = sorted(df["cluster"].unique())

# ===== 与散点图保持一致的基准色 =====
preset_colors = ["red", "yellow", "green"]
if len(clusters) <= len(preset_colors):
    palette = [mcolors.to_rgb(preset_colors[i]) for i in range(len(clusters))]
else:
    cmap_base = cm.get_cmap("tab20", len(clusters))
    palette = [cmap_base(i) for i in range(len(clusters))]

# ===== 绘制 violin plot =====
sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(6, 4))

sns.violinplot(
    x="cluster",
    y="score",
    data=df,
    order=clusters,
    palette=palette,
    cut=0,         # 不外推尾部
    inner="quartile"   # 显示四分位
)

plt.xlabel("Cluster")
plt.ylabel("Score")
plt.title("Score distribution by cluster")
plt.tight_layout()

# ===== 保存 =====
png_name = os.path.splitext(os.path.basename(CSV_TSNE))[0] + "_violin.png"
plt.savefig(os.path.join(OUT_DIR, png_name), dpi=300)
plt.show()
