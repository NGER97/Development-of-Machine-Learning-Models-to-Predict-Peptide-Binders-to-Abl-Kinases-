import pandas as pd

# Path to your merged CSV file
csv_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_Allrounds.csv"
# Output path for the new CSV
output_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_remove.csv"

df = pd.read_csv(csv_path)

# 筛选出 peptide_sequence 不为 "YLHWDYVW" 的行
df_filtered = df[df['peptide_sequence'] != "YLHWDYVW"]

# 将处理过的数据保存到一个新的 CSV 文件中
df_filtered.to_csv(output_path, index=False)

print("处理完成，已保存新的 CSV 文件")
