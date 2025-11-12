import pandas as pd

# ---------- 1. 读取数据 ----------
csv_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_Allrounds.csv"
df = pd.read_csv(csv_path)

# 如果某些 counts 列可能有缺失值，先用 0 填充
for rnd in range(1, 6):
    df[f"R{rnd}_counts"] = df[f"R{rnd}_counts"].fillna(0)

# ---------- 2. 统计各轮信息 ----------
results = []

for rnd in range(1, 6):
    counts_col = f"R{rnd}_counts"
    
    # 序列总数：该轮 counts > 0 的行数
    n_sequences = (df[counts_col] > 0).sum()
    
    # counts == 1 的序列数
    n_counts_eq_1 = (df[counts_col] == 1).sum()
    
    results.append({
        "Round": f"R{rnd}",
        "Total_sequences": n_sequences,
        "Counts_eq_1": n_counts_eq_1
    })

# ---------- 3. 打印 / 保存结果 ----------
summary_df = pd.DataFrame(results)
# 在打印前加一行，让 pandas 打印所有列
pd.set_option('display.max_columns', None)

print(summary_df)

# 如需保存为 CSV：
# out_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_RoundSummary.csv"
# summary_df.to_csv(out_path, index=False)
