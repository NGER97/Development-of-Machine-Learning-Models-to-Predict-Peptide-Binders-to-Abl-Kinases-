import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# ==== 路径 ====
MEMBERS_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand\clusters\optics_high_q0.85_members.csv"
OUT_TOP5    = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand\clusters\top5_representatives.csv"
OUT_ALT     = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand\clusters\top5_neighbors.csv"

K_FINAL = 5             # 想选几个代表
TOP_CLUSTERS = 10       # 先取排名前多少个簇来出候选（可调）

# 读入（需要列：cluster, peptide_sequence, pred_score, PC1, PC2）
df = pd.read_csv(MEMBERS_CSV)

# 去掉噪声簇（-1）
df = df[df["cluster"] != -1].copy()
if df.empty:
    raise SystemExit("没有有效簇（全是 -1），请调低 OPTICS 的参数或放宽子集筛选。")

# 标准化PC坐标（保证距离公平）
scaler = StandardScaler()
df[["PC1_z","PC2_z"]] = scaler.fit_transform(df[["PC1","PC2"]].values)

# --- 按簇做统计：偏好“分数高、点数多、形状紧凑”的簇 ---
stats = (df.groupby("cluster")
           .agg(n=("peptide_sequence","size"),
                med_score=("pred_score","median"),
                pc1_std=("PC1_z","std"),
                pc2_std=("PC2_z","std"))
           .reset_index())
stats["compact"] = 1.0 / (stats["pc1_std"].fillna(0)+1e-6) / (stats["pc2_std"].fillna(0)+1e-6)
stats["rank_score"] = stats["med_score"] * np.log1p(stats["n"]) * stats["compact"]

top_clusters = (stats.sort_values("rank_score", ascending=False)
                      .head(TOP_CLUSTERS)["cluster"].tolist())

# --- 计算每个入选簇的 medoid（到同簇所有点距离和最小的样本）---
candidates = []
for c in top_clusters:
    sub = df[df["cluster"]==c].copy()
    X = sub[["PC1_z","PC2_z"]].values
    # 成本 O(m^2)，m 为该簇大小；你的数据规模可承受
    D = pairwise_distances(X, metric="euclidean")
    medoid_local_idx = np.argmin(D.sum(axis=1))
    row = sub.iloc[medoid_local_idx]
    candidates.append({
        "cluster": c,
        "peptide_sequence": row["peptide_sequence"],
        "pred_score": row["pred_score"],
        "PC1": row["PC1"], "PC2": row["PC2"],
        "PC1_z": row["PC1_z"], "PC2_z": row["PC2_z"]
    })

cand_df = pd.DataFrame(candidates)
if cand_df.empty:
    raise SystemExit("没有候选 medoid，请检查上一步统计。")

# --- 在候选 medoid 间做“最远优先”挑 K_FINAL 个，保证分散覆盖 ---
Xc = cand_df[["PC1_z","PC2_z"]].values
chosen = []

# 先选 rank_score 最高簇的 medoid 作为第一个
best_cluster = stats.loc[stats["cluster"].isin(top_clusters)].sort_values("rank_score", ascending=False)["cluster"].iloc[0]
first_idx = cand_df.index[cand_df["cluster"]==best_cluster][0]
chosen.append(first_idx)

# 迭代选择下一个：使其到“已选集合”的最小距离最大
while len(chosen) < min(K_FINAL, len(cand_df)):
    remain = [i for i in range(len(cand_df)) if i not in chosen]
    dists = pairwise_distances(Xc[remain], Xc[chosen], metric="euclidean")
    min_to_chosen = dists.min(axis=1)
    next_local = int(remain[np.argmax(min_to_chosen)])
    chosen.append(next_local)

top5 = cand_df.iloc[chosen].copy()
top5.to_csv(OUT_TOP5, index=False)
print(f"[SAVE] representatives → {OUT_TOP5}")

# --- （可选）给每个代表找几个最近邻，便于实验替代 ---
neighbors_rows = []
for _, r in top5.iterrows():
    c = r["cluster"]
    sub = df[df["cluster"]==c].copy()
    X = sub[["PC1_z","PC2_z"]].values
    x0 = np.array([[r["PC1_z"], r["PC2_z"]]])
    d = pairwise_distances(X, x0, metric="euclidean").ravel()
    order = np.argsort(d)[:6]  # 代表+最近5个
    near = sub.iloc[order][["peptide_sequence","pred_score","PC1","PC2"]].copy()
    near.insert(0, "rep_peptide", r["peptide_sequence"])
    near.insert(1, "cluster", int(c))
    neighbors_rows.append(near)

neighbors = pd.concat(neighbors_rows, ignore_index=True)
neighbors.to_csv(OUT_ALT, index=False)
print(f"[SAVE] neighbors of reps → {OUT_ALT}")
