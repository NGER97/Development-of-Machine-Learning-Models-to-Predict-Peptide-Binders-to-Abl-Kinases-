import os
import math
import numpy as np
import pandas as pd

# Define your global constant alpha
alpha = 1.0

# Path to your merged CSV file
csv_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_remove_add_pseudocount.csv"
# Output path for the new CSV
output_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_remove_ER&Score.csv"

# 1) Load merged CSV
df = pd.read_csv(csv_path)

# 2) Remove any columns that end with "_counts"
df = df[[col for col in df.columns if not col.endswith("_counts")]]

# 3) Identify all replicate frequency columns, e.g. R1_frequency, R2_frequency, etc.
#    We'll sort them in ascending order of replicate number.
replicate_freq_cols = [
    col for col in df.columns 
    if col.endswith("_frequency") and col.startswith("R")
]
# Sort them by the integer in the column name (e.g., R1, R2, R3...)
# This assumes the replicate is always a single digit or that the integer is directly after "R".
# Adjust if your naming scheme is more complex.
replicate_freq_cols = sorted(
    replicate_freq_cols,
    key=lambda x: int(x[1:-10])  # remove 'R' at start and '_frequency' at end, leaving just the digit(s)
)

# 4) Define helper functions
def relu(x):
    return max(0, x)

def safe_log2_ratio(num, den):
    """
    Safely compute log2(num/den).
    If either num <= 0 or den <= 0, return 0 (or you can choose NaN if you prefer).
    """
    if num <= 0 or den <= 0:
        return 0
    else:
        return math.log2(num / den)

# 5) Compute ER and score for each row
ER_values = []
ER_relu_values = []

for _, row in df.iterrows():
    # Sum of log2 ratios between consecutive rounds
    sum_log = 0.0
    
    # For example, if replicate_freq_cols = ["R1_frequency", "R2_frequency", ...]
    # we do log2(R2 / R1) + log2(R3 / R2) + ...
    for k in range(len(replicate_freq_cols) - 1):
        freq_k     = row[replicate_freq_cols[k]]
        freq_kplus = row[replicate_freq_cols[k+1]]
        
        sum_log += safe_log2_ratio(freq_kplus, freq_k)
    
    ER = sum_log
    # ReLU(ER)
    ER_relu = relu(ER)
    # Score = alpha * ReLU(ER)
    score = alpha * ER_relu
    
    ER_values.append(ER)
    ER_relu_values.append(ER_relu)

max_ER = max(ER_relu_values)
if max_ER > 0:
    alpha = 1.0 / max_ER
else:
    alpha = 1.0  # 如果所有 ER_relu 均为 0，则保持原值

score_values = [alpha * er_relu for er_relu in ER_relu_values]

# 6) Add the new columns to the DataFrame
df["ER"] = ER_values
df["score"] = score_values

# 7) (Optional) If you have NaNs from zero frequencies, you can fill them with 0 or drop rows:
# df["ER"].fillna(0, inplace=True)
# df["score"].fillna(0, inplace=True)

# 8) Save the updated DataFrame to a new CSV
df.to_csv(output_path, index=False)

print(f"New CSV with ER and score saved to: {output_path}")