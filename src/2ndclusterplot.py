"""
Visualise second-level K-means results: scatter, violin, weighted logos
Author: Yuhao Rao
"""

import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import logomaker as lm
from matplotlib.lines import Line2D

# ===== 路径 =====
CSV_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_tsne_n_iter=1000_cluster2_3_subkmeans.csv"
OUT_DIR  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 读取 =====
cols = ["peptide_sequence", "score", "tSNE1", "tSNE2", "sub_cluster"]
df   = pd.read_csv(CSV_PATH, usecols=cols)
subs = sorted(df["sub_cluster"].unique())

# ========= 1) 散点图 =========
# 给每个 sub_cluster 分配基准色（tab10 / tab20）
cmap_base = cm.get_cmap("tab20", len(subs))
base_colors = {s: cmap_base(i)[:3] for i, s in enumerate(subs)}

def blend_with_white(rgb, alpha):
    white = np.array([1., 1., 1.])
    return tuple(white*(1-alpha) + np.array(rgb)*alpha)

point_colors = [
    blend_with_white(base_colors[row.sub_cluster], row.score)
    for row in df.itertuples()
]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df["tSNE1"], df["tSNE2"],
           c=point_colors, s=10, linewidths=0, alpha=0.9)

ax.set_xlabel("tSNE1"); ax.set_ylabel("tSNE2")
ax.set_title("Second-level K-means scatter (shade = score)")

handles = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=base_colors[s], markersize=7,
                  label=s) for s in subs]
ax.legend(handles=handles, title="Sub-cluster",
          bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()

scatter_png = os.path.join(
    OUT_DIR, os.path.splitext(os.path.basename(CSV_PATH))[0] + "_scatter.png")
fig.savefig(scatter_png, dpi=300)
plt.close(fig)
print(f"Scatter saved → {scatter_png}")

# ========= 2) violin plot =========
sns.set(style="whitegrid")
plt.figure(figsize=(max(6, len(subs)*0.8), 4))
palette = [base_colors[s] for s in subs]

sns.violinplot(x="sub_cluster", y="score", data=df,
               order=subs, palette=palette, cut=0, inner="quartile")
plt.xlabel("Sub-cluster"); plt.ylabel("Score")
plt.title("Score distribution by sub-cluster")
plt.tight_layout()

violin_png = os.path.join(
    OUT_DIR, os.path.splitext(os.path.basename(CSV_PATH))[0] + "_violin.png")
plt.savefig(violin_png, dpi=300); plt.close()
print(f"Violin plot saved → {violin_png}")

# ========= 3) sequence logos =========
def weighted_alignment_to_matrix(seqs, weights):
    L = len(seqs[0])
    alphabet = sorted(set("".join(seqs)))
    mat = pd.DataFrame(0.0, index=range(L), columns=alphabet)
    for seq, w in zip(seqs, weights):
        for pos, aa in enumerate(seq):
            mat.at[pos, aa] += w
    return mat

base_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
for sub in subs:
    block = df[df["sub_cluster"] == sub]
    seqs   = block["peptide_sequence"].tolist()
    wts    = block["score"].tolist()
    if len(seqs) == 0:
        continue

    mat = weighted_alignment_to_matrix(seqs, wts)

    fig, ax = plt.subplots(figsize=(max(len(seqs[0])*0.6, 8), 3.2))
    try:
        lm.Logo(mat, color_scheme="hydrophobicity", ax=ax)
    except Exception as e:
        print(f"[{sub}] logo failed: {e}")
        plt.close(fig); continue

    ax.set_title(f"Sub-cluster {sub} (n={len(seqs)})")
    ax.set_xlabel("Position"); ax.set_ylabel("Weighted count")
    plt.tight_layout()

    logo_png = os.path.join(OUT_DIR, f"{base_name}_sub{sub}_logo.png")
    plt.savefig(logo_png, dpi=300); plt.close(fig)
    print(f"[{sub}] Logo saved → {logo_png}")
