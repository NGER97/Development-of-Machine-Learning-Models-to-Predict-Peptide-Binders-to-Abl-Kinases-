import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取含有预测概率的 CSV 文件
csv_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_add_pseudocount_generalization_test-final_with_preds.csv"
df = pd.read_csv(csv_path)

# 2. 提取 pred_prob 列数据
data = df['pred_prob'].dropna().values

# 3. 绘制 violin plot
plt.figure(figsize=(6, 4))
plt.violinplot(data)
plt.xlabel('All samples')
plt.ylabel('Predicted Probability')
plt.title('Distribution of Predicted Probabilities')
plt.tight_layout()
plt.show()
