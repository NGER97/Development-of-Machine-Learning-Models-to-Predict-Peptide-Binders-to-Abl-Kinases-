import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== 路径 ====
IN_CSV  = r"D:\Me\IMB\Data\Yuhao\models\predictions_lib_expand_2.csv"  # 包含 pred_score
OUT_DIR = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA\lib_expand"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "violin_pred_score_all.png")
OUT_CSV = os.path.join(OUT_DIR, "violin_pred_score_all_summary.csv")

# ==== 读取 & 清洗 ====
df = pd.read_csv(IN_CSV, usecols=["pred_score"])   # 全量读入，不分块
scores = pd.to_numeric(df["pred_score"], errors="coerce").dropna().values

# ==== 统计并保存 ====
summary = pd.DataFrame({
    "n":[scores.size],
    "mean":[float(np.mean(scores))],
    "median":[float(np.median(scores))],
    "p10":[float(np.percentile(scores, 10))],
    "p90":[float(np.percentile(scores, 90))]
})
summary.to_csv(OUT_CSV, index=False)

# ==== 绘制小提琴图（全量） ====
plt.figure(figsize=(6,5))
parts = plt.violinplot([scores], positions=[1],
                       showmeans=False, showmedians=False, showextrema=False)

# 美化（可选）：设透明度与边框
for pc in parts['bodies']:
    pc.set_facecolor("#4c72b0")  # 蓝色
    pc.set_alpha(0.8)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.8)

# 中位数线样式（如果需要更明显，可加粗）
if 'cmedians' in parts:
    parts['cmedians'].set_color("black")
    parts['cmedians'].set_linewidth(1.2)

plt.xticks([1], [f"All (n={scores.size:,})"])
plt.ylabel("Prediction score")
plt.title("Prediction score distribution (All)")
plt.grid(alpha=0.3, linestyle="--", axis="y")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"[DONE] saved figure -> {OUT_PNG}")
print(f"[DONE] saved summary -> {OUT_CSV}")
