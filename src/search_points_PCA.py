import os
import glob
import pandas as pd

input_csv = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\05.Onehot\BB_ER&Score_onehot_encoded.csv"

df = pd.read_csv(input_csv)

df_filtered = df[(df['PC1'] > 6) & (df['PC2'] < 0) & (df['score'] > 0.5)]

columns_to_keep = ['peptide_sequence','score', 'PC1', 'PC2']
df_filtered = df_filtered[columns_to_keep]

output_csv = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\05.Onehot\BB_onehot_PC1M6_PC2L0_scoreM0.5.csv"
df_filtered.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"Filtered rows saved to: {output_csv}")
