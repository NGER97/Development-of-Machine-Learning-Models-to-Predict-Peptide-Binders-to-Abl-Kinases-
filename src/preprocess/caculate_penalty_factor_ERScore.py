import os
import math
import numpy as np
import pandas as pd

# ====== Global hyper-parameters ======
alpha = 1.0                 # scaling factor for scores (will be re-computed)
penalty_factor = 1.0        # multiply negative log2 ratios to penalise depletion

# ====== I/O paths ======
csv_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR4_significant.csv"
output_path = (
    rf"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR4_significant_ER&Score.csv"
)

# 1) Load data ---------------------------------------------------------------
df = pd.read_csv(csv_path)

# 2) 这一步原本会删除 *_counts 列；现保留所有列即可
# df = df[[c for c in df.columns if not c.endswith("_counts")]]

# 3) Pick replicate frequency columns (R1-R4) --------------------------------
rep_freq_cols = [
    c for c in df.columns
    if c.endswith("_frequency") and c.startswith("R") and c[1:-10].isdigit()
       and int(c[1:-10]) <= 3      # keep only rounds 1-4
]
rep_freq_cols = sorted(rep_freq_cols, key=lambda x: int(x[1:-10]))

# 4) Helper functions --------------------------------------------------------
def safe_log2_ratio(num: float, den: float) -> float:
    if num <= 0 or den <= 0:
        return 0.0
    return math.log2(num / den)

def relu(x: float) -> float:
    return max(0.0, x)

# 5) Compute ER, ER_relu, score ---------------------------------------------
ER_vals, ER_relu_vals = [], []

for _, row in df.iterrows():
    sum_log = 0.0
    for k in range(len(rep_freq_cols) - 1):
        r_current = row[rep_freq_cols[k]]
        r_next    = row[rep_freq_cols[k + 1]]

        log_ratio = safe_log2_ratio(r_next, r_current)
        if log_ratio < 0:
            log_ratio *= penalty_factor   # 惩罚负富集
        sum_log += log_ratio

    ER = sum_log
    ER_relu = relu(ER)
    ER_vals.append(ER)
    ER_relu_vals.append(ER_relu)

# 6) Auto-scale --------------------------------------------------------------
max_ER = max(ER_relu_vals)
alpha = 1.0 / max_ER if max_ER > 0 else 1.0
score_vals = [alpha * x for x in ER_relu_vals]

# 7) Attach new columns & save ----------------------------------------------
df["ER"]    = ER_vals
df["score"] = score_vals

df.to_csv(output_path, index=False)
print(f"New CSV saved to → {output_path}")
