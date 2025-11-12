import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# 1. 读取数据
input_csv = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\DBSCAN\eps0.12_min_samples5\BB_onehot_1_eps0.12_min_samples5.csv"  # 修改为你的数据文件路径
output_dir = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.Clustering\DBSCAN\eps0.12_min_samples5\special cluster 1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created folder: {output_dir}")

df = pd.read_csv(input_csv)

# 2. 选择用于聚类的特征，通常为 PC1 与 PC2
X = df[['PC1', 'PC2']].values

# 3. 数据标准化（建议进行标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 使用 DBSCAN 进行聚类
# 设定 DBSCAN 的参数
eps = 0.3       # 根据 k-distance 图和实验结果确定
min_samples = 5   # 根据数据维度和噪声情况选择

'''
# 可选：绘制 k-distance 图帮助选择 eps
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])
plt.figure(figsize=(8,4))
plt.plot(k_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
plt.title("k-distance Graph for DBSCAN")
plt.tight_layout()
base_name = os.path.basename(input_csv).replace(".csv", "")
output_fig_path = os.path.join(output_dir, f"{base_name}_k-distance Graph.png")
plt.savefig(output_fig_path)
plt.show()
'''

# 使用 DBSCAN 进行聚类
dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan_model.fit_predict(X_scaled)

# -1 表示噪声点
df['dbscan_label'] = labels

# 5. 聚类评估（如果有多个簇）
unique_labels = set(labels)
if len(unique_labels - {-1}) > 0:  # 排除噪声后至少有1个簇
    # 若存在噪声，建议排除噪声进行内部评估
    if -1 in labels:
        X_eval = X_scaled[labels != -1]
        labels_eval = labels[labels != -1]
    else:
        X_eval = X_scaled
        labels_eval = labels
    sil_score = silhouette_score(X_eval, labels_eval)
    print("Silhouette Score (excluding noise if present):", sil_score)
else:
    print("Not enough clusters for silhouette score calculation.")

# 6. 可视化聚类结果
plt.figure(figsize=(8, 6))
# 为每个簇（包括噪声）分配颜色
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 黑色用于噪声
        col = 'k'
    class_member_mask = (labels == k)
    plt.scatter(
        X_scaled[class_member_mask, 0],
        X_scaled[class_member_mask, 1],
        c=[col],  # 使用分配的颜色
        label=f"Cluster {k}",
        alpha=0.6,
        s=10 
    )

plt.title("DBSCAN Clustering Results")
plt.xlabel("PC1 (standardized)")
plt.ylabel("PC2 (standardized)")
plt.legend()
plt.tight_layout()
output_fig = f"{output_dir}\\BB_onehot_eps{eps}_min_samples{min_samples}.png"
plt.savefig(output_fig)
plt.show()

output_csv = f"{output_dir}\\BB_onehot_eps{eps}_min_samples{min_samples}.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"Clustering result saved to: {output_csv}")

subfolder = os.path.join(output_dir, f"eps{eps}_min_samples{min_samples}")
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
    print(f"Created folder: {subfolder}")

# 提取唯一的聚类标签（包括噪声 -1，如果不需要噪声，可以在循环中跳过）
unique_labels = df['dbscan_label'].unique()

for label in unique_labels:
    # 如果你想忽略噪声，可以取消下面行的注释
    if label == -1:
        continue

    # 提取属于当前簇的子集
    cluster_df = df[df['dbscan_label'] == label]
    
    # 构造输出文件名，例如 "your_data_cluster_0_eps0.5_min_samples5.csv"
    output_filename = f"BB_onehot_{label}_eps{eps}_min_samples{min_samples}.csv"
    output_cluster_csv = os.path.join(subfolder, output_filename)
    
    # 保存到 CSV 文件
    cluster_df.to_csv(output_cluster_csv, index=False, encoding='utf-8-sig')
    print(f"Cluster {label} extracted and saved as: {output_cluster_csv}")
