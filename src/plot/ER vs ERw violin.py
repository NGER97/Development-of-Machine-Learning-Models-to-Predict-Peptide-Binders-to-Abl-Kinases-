import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ① 读入带有 ER_round_weight 的结果文件
csv_file = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
df = pd.read_csv(csv_file)

# ② 绘制单列 violin plot
plt.figure(figsize=(6, 4))
sns.violinplot(
    y=df["score"],
    inner="quartile",          # 在小提琴内部画出四分位
    color="skyblue"
)
plt.title("Distribution of ER ")
plt.ylabel("ER (0–1)")
plt.xlabel("")
plt.tight_layout()
plt.show()

# --- 如果想同时比较 Raw ER 与 Weighted ER ---
plt.figure(figsize=(6, 4))
sns.violinplot(
    data=df[["ER_raw", "ER_round_weight"]],
    palette="viridis", inner="quartile"
)
plt.title("Raw ER  vs.  Round-Weighted ER")
plt.ylabel("Score")
plt.tight_layout()
plt.show()
