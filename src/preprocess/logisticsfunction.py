# =========================================
# ① 读入数据并整理为频率矩阵（仅前4轮）
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount.csv"
df = pd.read_csv(csv_file)

# 只取前4轮的频率和计数列
freq_cols = [c for c in df.columns if c.endswith("_frequency")][:4]
cnt_cols  = [c for c in df.columns if c.endswith("_counts")][:4]

# 构造频率矩阵
freq = df[freq_cols].values      # ndarray (n_seq × 4)
# 计算相邻轮次的 log-ratio（共3个）
log_ratio = np.log(freq[:, 1:] / freq[:, :-1])   # n_seq × (4-1)

# =========================================
# ② 轮次统一 Logistic-权重并计算 ER_round
# =========================================
# 2-1: 以每轮 log-ratio 的 0.6 分位数作为代表
r_all = np.quantile(log_ratio, 0.6, axis=0)       # 长度 = 3

# 2-2: Logistic 参数
x0 = np.median(r_all)                            # 曲线中心
k  = 2 / np.std(r_all)                           # 经验值，可调整

def S_prime(x, k=k, x0=x0):
    exp_term = np.exp(-k*(x - x0))
    return k * exp_term / (1 + exp_term)**2

w_round = S_prime(r_all)
w_round /= w_round.max()                         # 归一化到 0–1
print("Per-round weights:", w_round)

# 2-3: 加权 ER
ER_raw   = log_ratio.sum(axis=1)
ER_round = (log_ratio * w_round).sum(axis=1)

# ReLU + 0-1 归一化
score_round = np.maximum(ER_raw, 0)
score_round /= score_round.max()

df["ER_raw"]           = ER_raw
df["ER_round_weight"]  = score_round

# =========================================
# ③ 可视化
# =========================================
plt.figure(figsize=(6,4))
sns.kdeplot(df["ER_raw"],          label="Raw ER")
sns.kdeplot(df["ER_round_weight"], label="Round-weighted ER")
plt.title("Distribution before vs after weighting")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# —— 保存结果，文件名带上 remove_r5 —— 
# =========================================
out_csv = csv_file.replace(".csv", "_remove_r5.csv")
df.to_csv(out_csv, index=False)
print("结果写出:", out_csv)
