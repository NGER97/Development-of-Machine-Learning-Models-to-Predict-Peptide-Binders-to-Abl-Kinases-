import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

input_csv = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\DBSCAN\eps0.12_min_samples5\BB_onehot_1_eps0.12_min_samples5.csv"
output_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\DBSCAN\eps0.12_min_samples5\special cluster 1"

df = pd.read_csv(input_csv)
# 选择 PC1 和 PC2 为聚类输入数据
X = df[['PC1', 'PC2']].values

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# 将聚类标签添加到 DataFrame
df['cluster'] = cluster_labels

# 计算轮廓系数进行评估
sil_score = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {sil_score:.3f}")

# 可视化聚类结果
plt.figure(figsize=(8, 6))
edge_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink']  
# 这里我们统一使用 'viridis' 作为内部色彩的 colormap
for i in range(n_clusters):
    subset = df[df['cluster'] == i]
    plt.scatter(
        subset['PC1'],
        subset['PC2'],
        c=subset['score'],      # 根据 score 值控制内部颜色深浅
        cmap='viridis_r',         # 指定 colormap
        edgecolor=edge_colors[i],  # 不同簇用不同边缘颜色
        label=f"Cluster {i}",
        alpha=0.6
        #s=10
    )

plt.title("K-Means Clustering on PCA Result with Score-based Coloring")
plt.title("K-Means Clustering on PCA Result")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()

base_name = os.path.basename(input_csv).replace(".csv", "")
output_fig_path = os.path.join(output_path, f"{base_name}_onehot_pca_cluster_kmeans.png")
plt.savefig(output_fig_path)
plt.show()

for i in range(n_clusters):
    cluster_df = df[df['cluster'] == i]  # 提取簇 i
    output_csv_path = os.path.join(output_path, f"{base_name}_cluster_{i}.csv")
    cluster_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Cluster {i} has been saved to {base_name}_cluster_{i}.csv")
