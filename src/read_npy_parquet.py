import numpy as np
import pandas as pd
import os

input_file = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score_protBERT.npy" 
output_dir = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created folder: {output_dir}")

arr = np.load(input_file)
df = pd.DataFrame(arr)

output_csv = f"{output_dir}\\BB_ER&Score_protBERT.csv"
df.to_csv(output_csv, index=False)

df = pd.read_csv(output_csv)  # 读入刚才生成的 CSV
total = df.size                # 等价于 df.shape[0] * df.shape[1]
zero = (df == 0).sum().sum()   # 统计所有列中等于 0 的单元格数量
sparsity = zero / total

print(f"总单元格数：{total}")
print(f"零单元格数：{zero}，比例：{sparsity:.2%}")