import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from matplotlib import colormaps

# ========== 路径与参数 ==========
MEMBERS_CSV   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand\clusters\optics_high_q0.85_members.csv"
OUT_DIR       = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand\clusters"
PLOT_PNG      = os.path.join(OUT_DIR, "optics_q0.85_xi0.05_marked_18.png")
SELECT_CSV    = os.path.join(OUT_DIR, "representatives_3medoids_15backups.csv")

# 手动指定三个簇（如已知）：例如 [0,1,2]；若设为 None 则自动挑选最强的 3 个
CLUSTER_IDS   = None

# 标题注释（可根据你的实际参数修改/留空）
TITLE_NOTE = "OPTICS (q=0.85, xi=0.05) — medoids (★) & backups (■)"

# ========== 读取与预处理 ==========
df = pd.read_csv(MEMBERS_CSV)  # 需要列：cluster, peptide_sequence, pred_score, PC1, PC2
df = df[df["cluster"] != -1].copy()
if df.empty:
    raise SystemExit("没有有效簇（全为 -1）。请检查 OPTICS 输出或筛选阈值。")

# 自动挑选三个簇（如未手动指定）：按 rank = median_score * log(1+n)
if CLUSTER_IDS is None:
    stats = (df.groupby("cluster")
               .agg(n=("peptide_sequence","size"),
                    med_score=("pred_score","median"))
               .reset_index())
    stats["rank"] = stats["med_score"] * np.log1p(stats["n"])
    CLUSTER_IDS = stats.sort_values("rank", ascending=False)["cluster"].head(3).tolist()

print(f"[INFO] 使用的簇: {CLUSTER_IDS}")

# 标准化 PC 坐标用于距离计算
scaler = StandardScaler()
df[["PC1_z","PC2_z"]] = scaler.fit_transform(df[["PC1","PC2"]].values)

selected_rows = []  # 收集 3 个 medoids + 15 条备选

def select_medoid_and_backups(sub: pd.DataFrame, k_backups=5):
    """返回该簇的 medoid 行，以及 k_backups 个 farthest-first 备选（DataFrame 行索引）"""
    X = sub[["PC1_z","PC2_z"]].values
    # 计算簇内距离矩阵（欧氏）
    D = pairwise_distances(X, metric="euclidean")
    # medoid：总距离最小
    medoid_local_idx = int(np.argmin(D.sum(axis=1)))
    chosen = [medoid_local_idx]
    # farthest-first：每次选使得到已选集合的最小距离最大的点
    while len(chosen) < min(k_backups + 1, len(sub)):
        remain = [i for i in range(len(sub)) if i not in chosen]
        dists = D[np.ix_(remain, chosen)]
        min_to_chosen = dists.min(axis=1)
        next_idx = int(remain[int(np.argmax(min_to_chosen))])
        chosen.append(next_idx)
    return chosen  # [medoid, backup1, ...,]

# 遍历三个簇，挑 medoid + 5 备选
for cid in CLUSTER_IDS:
    sub = df[df["cluster"] == cid].reset_index(drop=True)
    if sub.empty:
        continue
    idxs = select_medoid_and_backups(sub, k_backups=5)
    # 标注角色
    for j, li in enumerate(idxs):
        row = sub.iloc[li].copy()
        row["role"] = "medoid" if j == 0 else "backup"
        selected_rows.append(row)

sel = pd.DataFrame(selected_rows)
# 只保留需要的列
sel = sel[["cluster","role","peptide_sequence","pred_score","PC1","PC2"]]

# 保存 18 条清单
os.makedirs(OUT_DIR, exist_ok=True)
sel.to_csv(SELECT_CSV, index=False)
print(f"[SAVE] representatives → {SELECT_CSV}")
print(sel.groupby(["cluster","role"]).size())

# ========== 画图 ==========
plt.figure(figsize=(7,6))

# 背景：全部样本淡灰
plt.scatter(df["PC1"], df["PC2"], s=6, c=(0.85,0.85,0.85,0.6), edgecolors='none', label="all (background)")

# 为三个簇着不同淡色（可选）
tab10 = colormaps.get_cmap("tab10")
for i, cid in enumerate(CLUSTER_IDS):
    sub = df[df["cluster"]==cid]
    color = tab10(i % 10)
    plt.scatter(sub["PC1"], sub["PC2"], s=8, c=[(color[0], color[1], color[2], 0.3)],
                edgecolors='none', label=f"cluster {cid} (bg)")

# 叠加：备选（蓝色方块）与 medoid（红色五角星）
medoids = sel[sel["role"]=="medoid"]
backs   = sel[sel["role"]=="backup"]

plt.scatter(backs["PC1"], backs["PC2"], s=70, marker="s",
            facecolors="none", edgecolors="#1f77b4", linewidths=1.5, label="backups (5/cluster)")

plt.scatter(medoids["PC1"], medoids["PC2"], s=140, marker="*",
            c="#d62728", edgecolors="k", linewidths=0.8, label="medoid (per cluster)")

# 给 medoid 标注簇号
for _, r in medoids.iterrows():
    plt.text(r["PC1"], r["PC2"], f" C{int(r['cluster'])}", fontsize=9,
             ha="left", va="bottom", color="#d62728", weight="bold")

plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title(TITLE_NOTE)
plt.legend(loc="best", fontsize=8, frameon=True)
plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=300)
plt.close()
print(f"[SAVE] plot → {PLOT_PNG}")
