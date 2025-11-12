"""
run_pca_protbert.py
Author: Yuhao Rao
Generate PCA of ProtBERT CLS embeddings and save results + figure, auto-merge score if missing
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- 路径 ----------
FEATURE_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_minCount1_R1-4_ER&Score_protBERT.parquet"
RAW_CSV      = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
OUT_DIR      = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA"
CSV_NAME     = "BB_protBERT_minCount1_R1-4_pca.csv"
PNG_NAME     = "BB_protBERT_minCount1_R1-4_pca.png"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 读取特征 ----------
df_feat = (pd.read_parquet(FEATURE_PATH)
           if FEATURE_PATH.endswith(".parquet")
           else pd.read_csv(FEATURE_PATH))

# ---------- 若缺失 score，自动合并 ----------
if "score" not in df_feat.columns:
    print("⚠️  'score' 列缺失，尝试从原始 CSV 合并…")
    df_raw = pd.read_csv(RAW_CSV, usecols=["peptide_sequence", "score"])
    df_feat = df_feat.merge(df_raw, on="peptide_sequence", how="left")
    if df_feat["score"].isna().any():
        raise ValueError("合并后仍有缺失 score，请检查序列是否一致。")

feat_cols = [c for c in df_feat.columns if c.startswith("protBERT_")]
X      = df_feat[feat_cols].values.astype(np.float32)
scores = df_feat["score"].values

# ---------- 标准化 + PCA ----------
X_std = StandardScaler().fit_transform(X)
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)

# ---------- 保存 CSV ----------
df_out = df_feat[["peptide_sequence", "score"]].copy()
df_out["PC1"], df_out["PC2"] = X_pca[:, 0], X_pca[:, 1]
csv_path = os.path.join(OUT_DIR, CSV_NAME)
df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"PCA 坐标已保存至: {csv_path}")

# ---------- 绘图 ----------
plt.figure(figsize=(8, 6))
norm = plt.Normalize(scores.min(), scores.max())
cmap = plt.cm.YlOrBr
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=scores, cmap=cmap, norm=norm,
            s=4, alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA of ProtBERT Features (color = score)")
cbar = plt.colorbar()
cbar.set_label("score (higher = darker)")
plt.tight_layout()

png_path = os.path.join(OUT_DIR, PNG_NAME)
plt.savefig(png_path, dpi=300)
plt.show()
print(f"PCA 图已保存至: {png_path}")
