# ============================================
# 0. 依赖 & 路径
# ============================================
import math
import pandas as pd

CSV_IN   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount.csv"
CSV_OUT  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_quadER&Score.csv"

# ============================================
# 1. 读入数据（保留 counts 列）
# ============================================
df = pd.read_csv(CSV_IN)

# ============================================
# 2. 找到 R1–R4 的 frequency 列并排序
# ============================================
rep_freq_cols = sorted(
    [c for c in df.columns
     if c.endswith("_frequency") and c.startswith("R") and c[1:-10].isdigit()
        and int(c[1:-10]) <= 4],                      # 只用 R1–R5
    key=lambda x: int(x[1:-10])                       # 按数字排序
)

def signed_J(p: float, q: float) -> float:
    """
    (p - q) * log2(p/q) 先计算“幅度”，再按方向赋符号：
      - 若 p > q  → 正
      - 若 p < q  → 负
    p、q 均 > 0 才计算；否则返回 0
    """
    if p <= 0 or q <= 0:
        return 0.0
    base = (p - q) * math.log2(p / q)        # >=0
    return base if p >= q else -base          # 按方向加符号

def relu(x: float) -> float:
    return max(0.0, x)

# ============================================
# 4. 逐行累加 ER
# ============================================
ER_vals, ER_relu_vals = [], []

for _, row in df.iterrows():
    total = 0.0
    for k in range(len(rep_freq_cols) - 1):
        p = row[rep_freq_cols[k + 1]]
        q = row[rep_freq_cols[k]]
        total += signed_J(p, q)               # 正增富集，负减衰减
    ER = total
    ER_relu = relu(ER)                        # 只保留正值
    ER_vals.append(ER)
    ER_relu_vals.append(ER_relu)

# ============================================
# 5. score 缩放到 [0,1]
# ============================================
max_ER = max(ER_relu_vals)
alpha  = 1.0 / max_ER if max_ER > 0 else 1.0
score_vals = [alpha * x for x in ER_relu_vals]

# ============================================
# 6. 写回 CSV
# ============================================
df["ER_signedJ"] = ER_vals
df["score"]      = score_vals

df.to_csv(CSV_OUT, index=False)
print(f"New CSV saved → {CSV_OUT}")