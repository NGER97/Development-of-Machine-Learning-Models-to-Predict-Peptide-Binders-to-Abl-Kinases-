import pandas as pd
from pathlib import Path

# ---------- 配置 ----------
CSV_ALL   = Path(r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_Allrounds.csv")
OUT_DIR   = CSV_ALL.parent / "03.QC_stats"
OUT_DIR.mkdir(exist_ok=True)

# ---------- 读取 & 准备 presence 矩阵 ----------
df = pd.read_csv(CSV_ALL)
rounds = sorted({c.split("_")[0] for c in df.columns if c.endswith("_counts")})

# presence_df[r] = 布尔 Series，True 表示该序列在轮次 r 中 counts>0
presence_df = pd.DataFrame({
    r: df[f"{r}_counts"] > 0
    for r in rounds
})

# ---------- 计算互现频率矩阵 ----------
# freq_matrix.loc[i,j] = P(出现在 Rj | 已出现在 Ri)
freq_matrix = pd.DataFrame(index=rounds, columns=rounds, dtype=float)

for ri in rounds:
    mask_i = presence_df[ri]
    n_i = mask_i.sum()    # Ri 中的活跃序列数
    for rj in rounds:
        # 同时出现在 Ri 和 Rj 的序列数
        n_both = (mask_i & presence_df[rj]).sum()
        freq_matrix.loc[ri, rj] = (n_both / n_i) if n_i>0 else 0.0

# ---------- 保存 & 打印结果 ----------
out_csv = OUT_DIR / "sequence_cooccurrence_rate.csv"
freq_matrix.to_csv(out_csv, index=True)

# Markdown 预览
print("\n### Sequence co-occurrence rate (rows→cols)")
print(freq_matrix.round(3).to_markdown())

print(f"\n结果已保存到: {out_csv}")
