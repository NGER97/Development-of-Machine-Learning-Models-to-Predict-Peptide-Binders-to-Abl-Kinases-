import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

# ========= 配置 =========
IN_CSV  = r"D:\Me\IMB\Data\Yuhao\models\predictions_lib_expand_2_gt0p8_with_features.csv"
OUT_DIR = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand"
CHUNK   = 50_000          # 分块大小：按内存调 20k~100k
N_COMP  = 2
SEED    = 42
MAX_PLOT_POINTS = 200_000 # 绘图的最大点数（随机采样控制图像体积）
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 读表头，确定列 =========
hdr = pd.read_csv(IN_CSV, nrows=0)
all_cols = list(hdr.columns)

# 预测/元信息列（PCA中要排除）
META = {"peptide_sequence", "pred_label", "pred_score", "class"}

# 特征列集合
all_feat_cols    = [c for c in all_cols if c not in META]
binary_feat_cols = [c for c in all_cols if c.startswith("binary::")]

def stream_fit_ipca(in_csv, feat_cols, chunk=CHUNK, n_comp=N_COMP, random_state=SEED):
    """第一遍：在全量数据上增量拟合 IPCA（不保存坐标，只学参数）；同时统计总行数。"""
    ipca = IncrementalPCA(n_components=n_comp)
    total = 0
    for df in pd.read_csv(in_csv, usecols=feat_cols, chunksize=chunk, low_memory=True):
        X = df.astype(np.float32, copy=False).values
        ipca.partial_fit(X)
        total += len(df)
    return ipca, total

def stream_transform_and_save(in_csv, feat_cols, ipca, total_rows, out_prefix,
                              chunk=CHUNK, max_plot_points=MAX_PLOT_POINTS, seed=SEED):
    """第二遍：transform 得到 PC 坐标，保存 CSV；同时随机下采样作图。"""
    # 输出降维坐标 CSV
    out_csv = os.path.join(OUT_DIR, f"{out_prefix}_pca.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("peptide_sequence,pred_score,PC1,PC2\n")

    # 采样概率（尽量均匀）
    k = min(max_plot_points, total_rows)
    p = k / total_rows if total_rows > 0 else 1.0
    rng = np.random.default_rng(seed)

    # 用于绘图的采样容器
    samp_pc1, samp_pc2, samp_score = [], [], []

    # 同步读取需要的列：序列、分数 + 特征
    usecols = ["peptide_sequence", "pred_score"] + feat_cols
    written = 0
    for df in pd.read_csv(in_csv, usecols=usecols, chunksize=chunk, low_memory=True):
        seq   = df["peptide_sequence"].astype(str).values
        score = pd.to_numeric(df["pred_score"], errors="coerce").fillna(0.0).values
        X     = df[feat_cols].astype(np.float32, copy=False).values

        Xp = ipca.transform(X)
        pc1 = Xp[:, 0]
        pc2 = Xp[:, 1]

        # 追加写出本块
        out_block = pd.DataFrame({
            "peptide_sequence": seq,
            "pred_score": score,
            "PC1": pc1,
            "PC2": pc2
        })
        out_block.to_csv(out_csv, mode="a", header=False, index=False, float_format="%.6f")
        written += len(out_block)

        # 随机采样用于作图
        if p >= 1.0:
            mask = np.ones(len(df), dtype=bool)
        else:
            mask = rng.random(len(df)) < p

        if mask.any():
            samp_pc1.append(pc1[mask])
            samp_pc2.append(pc2[mask])
            samp_score.append(score[mask])

    # 拼接采样点
    if samp_pc1:
        samp_pc1 = np.concatenate(samp_pc1)
        samp_pc2 = np.concatenate(samp_pc2)
        samp_score = np.concatenate(samp_score)
        # 若略超出目标，裁到最多 k 个点
        if len(samp_pc1) > k:
            idx = rng.choice(len(samp_pc1), size=k, replace=False)
            samp_pc1, samp_pc2, samp_score = samp_pc1[idx], samp_pc2[idx], samp_score[idx]
    else:
        samp_pc1 = np.array([])
        samp_pc2 = np.array([])
        samp_score = np.array([])

    # 保存方差解释率
    evr_txt = os.path.join(OUT_DIR, f"{out_prefix}_explained_variance.txt")
    with open(evr_txt, "w", encoding="utf-8") as f:
        evr = ipca.explained_variance_ratio_
        f.write(f"Explained variance ratio (PC1..PC{N_COMP}): {evr}\n")
        f.write(f"Sum: {evr.sum():.4f}\n")

    # 绘图（对采样点）
    out_png = os.path.join(OUT_DIR, f"{out_prefix}_pca.png")
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(samp_pc1, samp_pc2, c=samp_score, cmap="viridis_r", s=1, alpha=0.6)
    plt.colorbar(sc, label="Prediction Score")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA: {out_prefix}\nExplained var (PC1+PC2)={ipca.explained_variance_ratio_.sum():.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[OK] {out_prefix}: rows={written:,}, "
          f"EVR_sum={ipca.explained_variance_ratio_.sum():.4f}\n"
          f"     CSV={out_csv}\n     PNG={out_png}")

# ====== 跑 “全部特征” ======
print("[INFO] Fitting IPCA on ALL features ...")
ipca_all, total_rows = stream_fit_ipca(IN_CSV, all_feat_cols)
print(f"[INFO] Total rows = {total_rows:,}")
stream_transform_and_save(IN_CSV, all_feat_cols, ipca_all, total_rows, out_prefix="all_features")

# ====== 跑 “binary 特征” ======
print("[INFO] Fitting IPCA on BINARY features ...")
ipca_bin, total_rows_b = stream_fit_ipca(IN_CSV, binary_feat_cols)
print(f"[INFO] Total rows (binary) = {total_rows_b:,}")
stream_transform_and_save(IN_CSV, binary_feat_cols, ipca_bin, total_rows_b, out_prefix="binary_features")
