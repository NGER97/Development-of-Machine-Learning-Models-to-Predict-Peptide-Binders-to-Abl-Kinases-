import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt

# 设置两个文件的路径（请根据实际情况修改路径）
input_cluster0 = r"Yuhao/NGS/Abl L1/06.Clustering/onehot_k-means/BB_ER&Score_onehot_encoded_cluster_2_cluster_0.csv"
input_cluster5 = r"Yuhao/NGS/Abl L1/06.Clustering/onehot_k-means/BB_ER&Score_onehot_encoded_cluster_2_cluster_5.csv"

# 读取数据
df1 = pd.read_csv(input_cluster0)
df5 = pd.read_csv(input_cluster5)

# 方法一：使用 describe() 打印统计摘要
print("Statistical Summary for Cluster 0:")
print(df1['score'].describe())
print("\nStatistical Summary for Cluster 5:")
print(df5['score'].describe())

# 方法二：单独计算均值、中位数以及25%与75%的四分位数
mean_1 = df1['score'].mean()
median_1 = df1['score'].median()
q1_1 = df1['score'].quantile(0.25)
q3_1 = df1['score'].quantile(0.75)

mean_5 = df5['score'].mean()
median_5 = df5['score'].median()
q1_5 = df5['score'].quantile(0.25)
q3_5 = df5['score'].quantile(0.75)

print("\nDetailed Statistics:")
print(f"Cluster 0: Mean = {mean_1:.3f}, Median = {median_1:.3f}, 25th percentile = {q1_1:.3f}, 75th percentile = {q3_1:.3f}")
print(f"Cluster 5: Mean = {mean_5:.3f}, Median = {median_5:.3f}, 25th percentile = {q1_5:.3f}, 75th percentile = {q3_5:.3f}")

# 获取 score 列数据
score_cluster1 = df1['score']
score_cluster5 = df5['score']

# 进行 Mann-Whitney U 检验
# alternative 参数可以设置为 'two-sided'（双侧检验）、'less' 或 'greater'，
# 这里使用双侧检验。
u_stat, p_val = mannwhitneyu(score_cluster1, score_cluster5, alternative='two-sided')
print(f"Mann-Whitney U statistic: {u_stat}")
print(f"P-value: {p_val}")

def hamming_distance(seq1, seq2):
    """计算两个等长序列的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def compute_pairwise_hamming(sequences):
    """
    计算一组序列中两两之间的汉明距离，返回包含所有 pairwise 距离的列表
    """
    n = len(sequences)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(hamming_distance(sequences[i], sequences[j]))
    return distances

# 提取各文件中的序列（假设存储在列 "peptide_sequence"）
cluster0_seqs = df1["peptide_sequence"].tolist()
cluster5_seqs = df5["peptide_sequence"].tolist()

# 检查序列是否长度一致（Hamming 距离要求序列等长）
if not cluster0_seqs:
    raise ValueError("Cluster 0 文件中没有序列数据")
if not cluster5_seqs:
    raise ValueError("Cluster 5 文件中没有序列数据")
seq_len_0 = len(cluster0_seqs[0])
seq_len_5 = len(cluster5_seqs[0])
if any(len(seq) != seq_len_0 for seq in cluster0_seqs):
    raise ValueError("Cluster 0 中存在长度不一致的序列")
if any(len(seq) != seq_len_5 for seq in cluster5_seqs):
    raise ValueError("Cluster 5 中存在长度不一致的序列")

# 如果两个簇应该来自同一参考长度，也可以检查它们是否一致
if seq_len_0 != seq_len_5:
    print("Warning: Cluster 0 and Cluster 5 sequences have different lengths.")

# 计算每个簇内部所有序列的 pairwise Hamming 距离
distances_0 = compute_pairwise_hamming(cluster0_seqs)
distances_5 = compute_pairwise_hamming(cluster5_seqs)

# 定义函数打印统计指标
def print_stats(distances, cluster_name):
    print(f"Statistics for {cluster_name}:")
    if distances:
        print(f"  Count: {len(distances)}")
        print(f"  Mean: {np.mean(distances):.3f}")
        print(f"  Median: {np.median(distances):.3f}")
        print(f"  Min: {np.min(distances)}")
        print(f"  Max: {np.max(distances)}\n")
    else:
        print("  Not enough sequences to compute pairwise distances.\n")

print_stats(distances_0, "Cluster 0")
print_stats(distances_5, "Cluster 5")

# 绘制箱线图比较两个簇内部的 Hamming 距离分布
plt.figure(figsize=(8, 6))
plt.boxplot([distances_0, distances_5], labels=["Cluster 0", "Cluster 5"])
plt.ylabel("Hamming Distance")
plt.title("Pairwise Hamming Distance Distribution")
plt.show()

# 若需要进行统计检验（例如 Mann-Whitney U 检验），确保各自有足够数据
if len(distances_0) > 0 and len(distances_5) > 0:
    u_stat, p_val = mannwhitneyu(distances_0, distances_5, alternative="two-sided")
    print(f"Mann-Whitney U Statistic: {u_stat}, P-value: {p_val}")
else:
    print("Not enough data to perform Mann-Whitney U test.")