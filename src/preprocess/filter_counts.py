import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# 1. 读取原始数据
raw_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount.csv"
df = pd.read_csv(raw_path)

# 2. 计算 R1–R4 总 reads
totals = {f"R{i}": df[f"R{i}_counts"].sum() for i in range(1, 5)}

# 3. 对 R1→R2, R2→R3, R3→R4 做 Fisher 精确检验
for i in range(1, 4):
    a, b = f"R{i}", f"R{i+1}"
    p_col = f"p_{b}_vs_{a}"
    df[p_col] = df.apply(
        lambda row: fisher_exact(
            [[row[f"{a}_counts"], totals[a] - row[f"{a}_counts"]],
             [row[f"{b}_counts"], totals[b] - row[f"{b}_counts"]]],
            alternative="two-sided"
        )[1],
        axis=1
    )

# 4. 分别做 Benjamini–Hochberg 校正
for i in range(1, 4):
    raw_p = df[f"p_R{i+1}_vs_R{i}"]
    df[f"adj_p_R{i+1}_vs_R{i}"] = multipletests(raw_p, method="fdr_bh")[1]

# 5. 标记：任意相邻对显著 (adj_p < 0.01)
df["significant_any"] = (
    (df["adj_p_R2_vs_R1"] < 0.01) |
    (df["adj_p_R3_vs_R2"] < 0.01) |
    (df["adj_p_R4_vs_R3"] < 0.01)
)

# 6. 过滤出显著变化的肽段
df_sig = df[df["significant_any"]].reset_index(drop=True)

# 7. 输出文件
out_all = raw_path.replace(".csv", "_with_pvalues.csv")
out_sig = raw_path.replace(".csv", "_R1toR4_significant.csv")
df.to_csv(out_all, index=False)
df_sig.to_csv(out_sig, index=False)

print(f"已保存完整结果：{out_all}")
print(f"已保存显著肽段（任一对 adj_p<0.01）：{out_sig}，共 {len(df_sig)} 条")
