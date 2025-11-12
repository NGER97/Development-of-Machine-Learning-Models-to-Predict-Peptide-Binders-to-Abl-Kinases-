"""
Third-level K-means on sub-clusters 2_0, 3_0, 3_1 + visualisation
Author: Yuhao Rao
"""

import os, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

targets = {"2_0", "3_0", "3_1"}          # 仅对这些子簇再聚类
base_cols = ["peptide_sequence", "score", "tSNE1", "tSNE2", "sub_cluster"]
df = pd.read_csv(CSV_PATH, usecols=base_cols)

df_third = df[df["sub_cluster"].isin(targets)].copy()

# ===== 工具：最佳 KMeans =====
def best_kmeans(X, k_min=2, k_max=10, seed=42):
    best_k, best_sil, best_model = k_min, -1, None
    for k in range(k_min, min(k_max, len(X)) + 1):
        if k == 1:
            continue
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_)
        if sil > best_sil:
            best_k, best_sil, best_model = k, sil, km
    return best_model, best_k, best_sil

# ===== 对每个 sub_cluster 再聚类 =====
for sub in targets:
    block = df_third[df_third["sub_cluster"] == sub]
    X     = block[["tSNE1", "tSNE2"]].values
    model, k, sil = best_kmeans(X)
    print(f"{sub}: best k = {k}, silhouette = {sil:.3f}")
    new_labels = [f"{sub}_{lbl}" for lbl in model.labels_]
    df_third.loc[block.index, "subsub_cluster"] = new_labels

# ===== 保存新 CSV =====
base_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
csv_out   = os.path.join(OUT_DIR, f"{base_name}_subsubkmeans.csv")
df_third.to_csv(csv_out, index=False, encoding="utf-8-sig")
print(f"三级聚类 CSV → {csv_out}")

# ==================== 可视化 ==================== #
subs = sorted(df_third["subsub_cluster"].unique())
cmap_base = cm.get_cmap("tab20", len(subs))
base_colors = {s: cmap_base(i)[:3] for i, s in enumerate(subs)}

def blend(rgb, a):
    w = np.array([1.,1.,1.]); return tuple(w*(1-a)+np.array(rgb)*a)

point_colors = [blend(base_colors[r.subsub_cluster], r.score) for r in df_third.itertuples()]

# —— 散点 —— #
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(df_third["tSNE1"], df_third["tSNE2"],
           c=point_colors, s=10, linewidths=0, alpha=0.9)
ax.set_xlabel("tSNE1"); ax.set_ylabel("tSNE2")
ax.set_title("Third-level K-means scatter (shade = score)")
handles = [Line2D([0],[0], marker='o', color='w',
                  markerfacecolor=base_colors[s], markersize=7, label=s)
           for s in subs]
ax.legend(handles=handles, title="Sub-sub-cluster",
          bbox_to_anchor=(1.02,1), loc="upper left")
fig.tight_layout()
scatter_png = os.path.join(OUT_DIR, f"{base_name}_subsubkmeans_scatter.png")
fig.savefig(scatter_png, dpi=300); plt.close(fig)
print(f"Scatter → {scatter_png}")

# —— violin —— #
sns.set(style="whitegrid")
plt.figure(figsize=(max(6,len(subs)*0.9),4))
palette = [base_colors[s] for s in subs]
sns.violinplot(x="subsub_cluster", y="score", data=df_third,
               order=subs, palette=palette, cut=0, inner="quartile")
plt.xlabel("Sub-sub-cluster"); plt.ylabel("Score")
plt.title("Score distribution (third level)")
plt.tight_layout()
violin_png = os.path.join(OUT_DIR, f"{base_name}_subsubkmeans_violin.png")
plt.savefig(violin_png, dpi=300); plt.close()
print(f"Violin → {violin_png}")

# —— sequence logo —— #
def weighted_matrix(seqs, wts):
    L = len(seqs[0]); alpha = sorted(set("".join(seqs)))
    mat = pd.DataFrame(0., index=range(L), columns=alpha)
    for s,w in zip(seqs,wts):
        for i,a in enumerate(s): mat.at[i,a]+=w
    return mat

for sub2 in subs:
    blk = df_third[df_third["subsub_cluster"]==sub2]
    seqs, wts = blk["peptide_sequence"].tolist(), blk["score"].tolist()
    if not seqs: continue
    mat = weighted_matrix(seqs,wts)
    fig, ax = plt.subplots(figsize=(max(len(seqs[0])*0.6,8),3.2))
    try:
        lm.Logo(mat, color_scheme="hydrophobicity", ax=ax)
    except Exception as e:
        print(f"{sub2} logo err: {e}"); plt.close(fig); continue
    ax.set_title(f"{sub2} (n={len(seqs)})"); ax.set_xlabel("Pos"); ax.set_ylabel("Weighted")
    plt.tight_layout()
    logo_png = os.path.join(OUT_DIR, f"{base_name}_subsub{sub2}_logo.png")
    plt.savefig(logo_png, dpi=300); plt.close(fig)
    print(f"Logo → {logo_png}")
