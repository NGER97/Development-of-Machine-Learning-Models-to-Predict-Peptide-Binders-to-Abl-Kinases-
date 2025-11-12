import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# 读取文件
weight_file = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_wRoundER.csv"
remove_file = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_remove_r5.csv"
df_score = pd.read_csv(weight_file)      # 包含 "peptide_sequence" 和 "score"
df_er    = pd.read_csv(remove_file)      # 包含 "peptide_sequence" 和 "ER_round_weight"

# 合并两张表，确保同一序列对齐
df = pd.merge(
    df_er[['peptide_sequence','ER_round_weight']],
    df_score[['peptide_sequence','score']],
    on='peptide_sequence'
)

# 1. Pearson & Spearman 相关性
pearson_corr, pearson_p = pearsonr(df['ER_round_weight'], df['score'])
spearman_corr, spearman_p = spearmanr(df['ER_round_weight'], df['score'])
print(f"Pearson r = {pearson_corr:.3f} (p = {pearson_p:.3e})")
print(f"Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.3e})")

# 散点图 + 回归直线
plt.figure(figsize=(6,6))
sns.regplot(
    x='ER_round_weight', y='score', data=df,
    scatter_kws={'s':10, 'alpha':0.6},
    line_kws={'color':'red'}
)
plt.xlabel('ER_round_weight')
plt.ylabel('remove_r5_score')
plt.title('ER_round_weight vs. remove_r5_score')
plt.tight_layout()
plt.show()

# 2. Kendall’s τ 排名一致性
kendall_corr, kendall_p = kendalltau(df['ER_round_weight'], df['score'])
print(f"Kendall’s τ = {kendall_corr:.3f} (p = {kendall_p:.3e})")

# 3. Jaccard 相似度（取前 10% 高分序列重叠度）
top_frac = 0.10
n_top = int(len(df) * top_frac)
idx_er_top    = set(df.nlargest(n_top, 'ER_round_weight').index)
idx_score_top = set(df.nlargest(n_top, 'score').index)
jaccard = len(idx_er_top & idx_score_top) / len(idx_er_top | idx_score_top)
print(f"Jaccard similarity for top {int(top_frac*100)}%: {jaccard:.3f}")
