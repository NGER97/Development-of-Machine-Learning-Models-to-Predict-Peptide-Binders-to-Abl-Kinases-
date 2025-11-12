import os
import numpy as np
import pandas as pd

# ========= 参数 =========
BASE_DIR   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing"
ALL_FILE   = os.path.join(BASE_DIR, "BB_Allrounds.csv")
SCORE_FILE = os.path.join(BASE_DIR, "BB_add_pseudocount_remove_r5.csv")

MIN_COUNT  = 1                       # 与前面过滤脚本保持一致
ROUNDS     = ["R1", "R2", "R3", "R4"]  # 仅看 R1–R4
BIN_WIDTH  = 0.1                     # score 分箱宽度
MAX_SCORE  = 1.0                     # 分箱上限（>MAX_SCORE 的归入最后一档）

# ========= 1) 读入原始 counts =========
df_all = pd.read_csv(ALL_FILE)
count_cols = [f"{r}_counts" for r in ROUNDS]

# ========= 2) 判断哪些序列被删除 =========
mask_keep = (df_all[count_cols] >= MIN_COUNT).all(axis=1)
removed_ids = df_all.loc[~mask_keep, "peptide_sequence"]

# ========= 3) 取出被删序列对应的 score =========
df_score = pd.read_csv(SCORE_FILE, usecols=["peptide_sequence", "score"])
removed_scores = df_score[df_score["peptide_sequence"].isin(removed_ids)].copy()

# ========= 4) 分箱统计 =========
bins = np.arange(0, MAX_SCORE + BIN_WIDTH, BIN_WIDTH)      # 0,0.1,0.2,...1.0
bins = np.append(bins, np.inf)                             # 最后一档收所有 >1.0
labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" if i < len(bins)-2 
          else f">{MAX_SCORE}"
          for i in range(len(bins)-1)]

removed_scores["score_bin"] = pd.cut(
        removed_scores["score"], bins=bins, labels=labels, right=False)

tbl = (removed_scores["score_bin"]
       .value_counts(sort=False)
       .rename_axis("score_range")
       .reset_index(name="count"))
tbl["percent"] = tbl["count"] / tbl["count"].sum() * 100

# ========= 5) 打印结果 =========
print("\n=== Dropped Sequences: Score Distribution ===")
print(tbl.to_string(index=False, formatters={
      "percent": "{:.2f}%".format}))

# ========= 6) 如需保存成 CSV / Excel =========
out_path = os.path.join(BASE_DIR, "dropped_score_distribution.csv")
tbl.to_csv(out_path, index=False)
print(f"\n结果已保存：{out_path}")

# ===== Plot: score distribution of dropped sequences =====
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.bar(tbl["score_range"].astype(str), tbl["count"])
plt.xlabel("Score range")
plt.ylabel("Number of dropped sequences")
plt.title("Score distribution of dropped sequences")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Uncomment the next line if you’d like to save the figure:
plt.savefig(os.path.join(BASE_DIR, "dropped_score_distribution.png"), dpi=300)

plt.show()
