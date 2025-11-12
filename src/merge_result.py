import pandas as pd

# ----------------------------
# 1. 定义文件路径
# ----------------------------
pred_path    = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\prediction_results.csv"
pos_raw_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR3_filtered.csv"
neg_raw_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_monotonic_decrease_R1toR4.csv"

# ----------------------------
# 2. 读取预测结果
# ----------------------------
df_pred = pd.read_csv(pred_path)
# 保留原始行顺序
df_pred['orig_idx'] = df_pred.index

# ----------------------------
# 3. 读取原始阳性/阴性数据，不合并编号
# ----------------------------
df_pos = pd.read_csv(pos_raw_path).reset_index(drop=True)
df_pos['Protein'] = df_pos.index + 1    # 阳性：1..40

df_neg = pd.read_csv(neg_raw_path).reset_index(drop=True)
df_neg['Protein'] = df_neg.index + 1    # 阴性：1..270

# ----------------------------
# 4. 按 true_label 分组并 merge
# ----------------------------
# 阳性预测——和 df_pos 合并
df_pred_pos = df_pred[df_pred['true_label']==1].copy()
df_merged_pos = pd.merge(
    df_pred_pos,
    df_pos,
    on='Protein',
    how='left'
)

# 阴性预测——和 df_neg 合并
df_pred_neg = df_pred[df_pred['true_label']==0].copy()
df_merged_neg = pd.merge(
    df_pred_neg,
    df_neg,
    on='Protein',
    how='left'
)

# ----------------------------
# 5. 重新拼回原始预测顺序
# ----------------------------
df_all = pd.concat([df_merged_pos, df_merged_neg], ignore_index=True)
df_all = df_all.sort_values('orig_idx').reset_index(drop=True)

# ----------------------------
# 6. 保存并查看
# ----------------------------
output_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\detailed_predictions.csv"
df_all.to_csv(output_path, index=False)

print(f"合并完成，共 {len(df_all)} 条记录，已保存至：{output_path}")
# 如果想看前几行：
print(df_all.head())
