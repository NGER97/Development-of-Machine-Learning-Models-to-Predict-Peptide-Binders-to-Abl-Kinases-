"""
Plot t-SNE clusters with score shading (score from weighted-ER file)
Author: Yuhao Rao
"""

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.cm as cm

# ===== 路径 =====
TSNE_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_minCount1_R1-4_.csv"
SCORE_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
OUT_DIR   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE"

# ===== 读取并合并 =====
df_tsne  = pd.read_csv(TSNE_CSV,  usecols=["peptide_sequence", "tSNE1", "tSNE2", "cluster"])
df_score = pd.read_csv(SCORE_CSV, usecols=["peptide_sequence", "score"])
df = df_tsne.merge(df_score, on="peptide_sequence", how="inner")

clusters = sorted(df["cluster"].unique())

# ===== 每簇基准色 =====
preset_colors = ["red", "yellow", "green"]          # 不够时用 tab20
if len(clusters) <= len(preset_colors):
    base_colors = {c: mcolors.to_rgb(preset_colors[i]) for i, c in enumerate(clusters)}
else:
    cmap_base = cm.get_cmap("tab20", len(clusters))
    base_colors = {c: cmap_base(i)[:3] for i, c in enumerate(clusters)}

def blend_with_white(rgb, alpha):
    white = np.array([1.0, 1.0, 1.0])
    return tuple(white * (1 - alpha) + np.array(rgb) * alpha)

point_colors = [
    blend_with_white(base_colors[row.cluster],
                     0.2 + 0.8 * row.score)     # ★ 修改处
    for row in df.itertuples()
]



# ===== 绘图 =====
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["tSNE1"], df["tSNE2"],
           c=point_colors, s=20, alpha=0.9, linewidths=0)

ax.set_xlabel("t-SNE1")
ax.set_ylabel("t-SNE2")
ax.set_title("t-SNE clusters (shade = minCount1_Score)")

# 图例：只显示簇颜色，不受 score 淡化影响
handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=base_colors[c], markersize=8,
                  label=f"Cluster {c}") for c in clusters]
ax.legend(handles=handles, title="Cluster",
          bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout()

# ===== 保存 =====
png_name = os.path.splitext(os.path.basename(TSNE_CSV))[0] + ".png"
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, png_name), dpi=300)
plt.show()
