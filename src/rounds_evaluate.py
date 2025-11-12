import pandas as pd
from pathlib import Path

# 路径
CSV_ALL = Path(r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_Allrounds.csv")
OUT_DIR = CSV_ALL.parent / "03.QC_stats"
OUT_DIR.mkdir(exist_ok=True)

# 读取
df = pd.read_csv(CSV_ALL)

# 找出所有轮次
rounds = sorted({c.split("_")[0] for c in df.columns if c.endswith("_counts")})

# 结果列表
results = []
for r in rounds:
    col = f"{r}_counts"
    total_reads    = df[col].sum()          # 所有序列的reads总和
    unique_peptides = (df[col] > 0).sum()   # 计数>0的序列数
    results.append({
        "round": r,
        "total_reads": int(total_reads),
        "unique_peptides": int(unique_peptides)
    })

# 转成 DataFrame 并保存
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "reads_vs_peptides_per_round.csv", index=False)

# 在终端打印 Markdown 预览
print("\n### Reads vs. Unique Peptides per Round")
print(res_df.to_markdown(index=False))
