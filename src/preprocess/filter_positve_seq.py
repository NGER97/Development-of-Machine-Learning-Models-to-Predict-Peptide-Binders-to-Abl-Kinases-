import pandas as pd

# 1. 读取数据
file_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount.csv"
df = pd.read_csv(file_path)

# 2. 判断 R1→R5 频率单调不升
mask = True
for i in range(1, 4):
    prev = f"R{i}_frequency"
    curr = f"R{i+1}_frequency"
    mask &= (df[curr] <= df[prev])

# 3. 筛选并统计
filtered = df[mask]
count = len(filtered)
print(f"从 R1 到 R4 单调下降（或持平）的序列共有：{count} 条")

# 4. 保存结果
output_path = file_path.replace(".csv", "_monotonic_decrease_R1toR4.csv")
filtered.to_csv(output_path, index=False)
print(f"已将筛选结果保存至：{output_path}")
