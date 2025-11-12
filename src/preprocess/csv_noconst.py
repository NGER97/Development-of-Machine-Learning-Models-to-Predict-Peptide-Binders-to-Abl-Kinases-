import pandas as pd

IN_CSV  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score-final.csv"
OUT_CSV = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score-final_noconst.csv"

print("读取中...")
df = pd.read_csv(IN_CSV, low_memory=False)

# Protein 列先留着
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
print(f"检测到 {len(const_cols)} 列为常数列，将被删除：")
print(const_cols[:20], "..." if len(const_cols) > 20 else "")

df_clean = df.drop(columns=const_cols)
print("删除后形状：", df_clean.shape)

df_clean.to_csv(OUT_CSV, index=False)
print("已保存到：", OUT_CSV)
