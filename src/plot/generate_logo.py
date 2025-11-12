"""
Generate weighted sequence logos per cluster
Author: Yuhao Rao
"""

import os, pandas as pd
import matplotlib.pyplot as plt
import logomaker as lm

# ===== 路径 =====
TSNE_CSV  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE\BB_protBERT_minCount1_R1-4_.csv"
SCORE_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
OUT_DIR   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\t-SNE"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 读取并合并 =====
df_tsne  = pd.read_csv(TSNE_CSV,  usecols=["peptide_sequence", "cluster"])
df_score = pd.read_csv(SCORE_CSV, usecols=["peptide_sequence", "score"])
df = df_tsne.merge(df_score, on="peptide_sequence", how="inner")

# ===== 工具函数：加权计数矩阵 =====
def weighted_alignment_to_matrix(seqs, weights):
    L = len(seqs[0])
    alphabet = sorted(set("".join(seqs)))
    mat = pd.DataFrame(0.0, index=range(L), columns=alphabet)
    for seq, w in zip(seqs, weights):
        for pos, aa in enumerate(seq):
            mat.at[pos, aa] += w
    return mat

# ===== 逐簇生成 Logo =====
clusters  = sorted(df["cluster"].unique())
base_name = os.path.splitext(os.path.basename(TSNE_CSV))[0]

for c in clusters:
    sub = df[df["cluster"] == c]
    sequences = sub["peptide_sequence"].tolist()
    weights   = sub["score"].tolist()

    if not sequences:
        continue

    mat = weighted_alignment_to_matrix(sequences, weights)

    fig, ax = plt.subplots(figsize=(max(len(sequences[0]) * 0.6, 8), 3.5))
    try:
        lm.Logo(mat, color_scheme="hydrophobicity", ax=ax)
    except Exception as e:
        print(f"[Cluster {c}] 生成 Logo 失败: {e}")
        plt.close(fig)
        continue

    ax.set_title(f"Cluster {c} (n={len(sequences)})  —  weight = score")
    ax.set_xlabel("Position"); ax.set_ylabel("Weighted count")
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{base_name}_cluster{c}_logo.png")
    try:
        plt.savefig(out_png, dpi=300)
        print(f"[Cluster {c}] Logo saved → {out_png}")
    except Exception as e:
        print(f"[Cluster {c}] 保存失败: {e}")
    finally:
        plt.close(fig)
