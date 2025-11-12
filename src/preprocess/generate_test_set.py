import pandas as pd

# 1. 定义文件路径（请根据实际路径调整）
raw_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount.csv"
neg_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_monotonic_decrease_R1toR4.csv"
pos_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR3_filtered.csv"

# 2. 读取数据
df_raw = pd.read_csv(raw_path)
df_neg = pd.read_csv(neg_path)
df_pos = pd.read_csv(pos_path)

# 3. 构造要排除的肽段集合
remove_seqs = set(df_neg['peptide_sequence']) | set(df_pos['peptide_sequence'])

# 4. 在原始数据中排除这些肽段
df_test = df_raw[~df_raw['peptide_sequence'].isin(remove_seqs)].reset_index(drop=True)

# 5. 保存新的测试集 CSV
output_path = raw_path.replace(".csv", "_generalization_test.csv")
df_test.to_csv(output_path, index=False)

# 6. 输出基本信息
print(f"生成新的测试集，共 {len(df_test)} 条剩余序列，已保存至：{output_path}")
