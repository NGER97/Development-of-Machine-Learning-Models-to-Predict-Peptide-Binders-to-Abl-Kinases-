import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

input_csv = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\DBSCAN\eps0.12_min_samples5\BB_onehot_1_eps0.12_min_samples5.csv"
output_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\DBSCAN\eps0.12_min_samples5\BB_onehot_1_eps0.12_min_samples5_Elbow Method_Silhouette Score.png"

df = pd.read_csv(input_csv)

# 假设你已有包含 PCA 结果的 DataFrame df，其中包含 'PC1' 和 'PC2'
X = df[['PC1', 'PC2']].values

# 设置不同的聚类数范围
cluster_range = range(2, 11)
sse = []
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sse.append(kmeans.inertia_)
    sil_score = silhouette_score(X, labels)
    silhouette_scores.append(sil_score)
    print(f"k={k}: SSE={kmeans.inertia_:.2f}, Silhouette Score={sil_score:.3f}")

# 绘制肘部法则图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, sse, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method")

# 绘制轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='green')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")
plt.tight_layout()

plt.savefig(output_path)
plt.show()