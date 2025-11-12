import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

import warnings

# ① _check_n_features -> 将在 1.7 删除
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

# --------------------------------------------
# 后续正式代码（import pandas, sklearn 等）...


# -------------------------
# 1. 读取阳性 / 阴性数据并打标签
# -------------------------
pos_path = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_R1toR3_filtered_ALLfeatures.csv"
neg_path = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_monotonic_decrease_R1toR4_ALLfeatures.csv"

df_pos = pd.read_csv(pos_path)
df_neg = pd.read_csv(neg_path)
df_pos['label'] = 1
df_neg['label'] = 0

# 合并并分离特征与标签
df_all = pd.concat([df_pos, df_neg], ignore_index=True)
# 假设第一列是 “Protein”，最后一列是 “label”
feature_cols = [c for c in df_all.columns if c not in ("Protein", "label")]
X = df_all[feature_cols].values
y = df_all['label'].values

# -------------------------
# 2. 定义交叉验证与下采样策略
# -------------------------
# 6 折分层 CV，保证每折中正负比例
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
# 训练时下采样：正负各 1:1，即 40:40
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)

# -------------------------
# 3. 构建 Pipeline（下采样 → 归一化 → 特征选择 → 分类器）
# -------------------------
pipe = ImbPipeline([
    ('undersample', rus),
    ('scaler', MinMaxScaler()),
    ('feat_sel', SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold= "200*median"    # 保留 importance ≥ 中位数 的特征
    )),
    ('clf', RandomForestClassifier(random_state=42))
])

# -------------------------
# 4. 定义超参数分布并设置 RandomizedSearchCV
# -------------------------
param_dist = {
    'clf__n_estimators': [100, 300, 500],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_leaf': [1, 2, 5, 10]
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=20,                # 随机搜索 20 次
    cv=skf,
    scoring=['roc_auc', 'precision', 'recall', 'f1'],
    refit='roc_auc',          # 以 ROC AUC 最优模型重拟
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# -------------------------
# 5. 训练并输出结果
# -------------------------
search.fit(X, y)

print("=== 最佳模型参数（Based on ROC AUC） ===")
print(search.best_params_)
print(f"最佳 ROC AUC: {search.best_score_:.4f}")

# 可查看其他指标的 CV 结果，例如：
results = pd.DataFrame(search.cv_results_)
metrics = ['mean_test_precision', 'mean_test_recall', 'mean_test_f1']
print("\n其他指标（CV 平均）：")
print(results[metrics].loc[results['rank_test_roc_auc']==1].to_string(index=False))

# 从搜索出的最佳模型里取出 SelectFromModel 这一步
selector = search.best_estimator_.named_steps['feat_sel']

# 布尔掩码，True 表示该特征被选中
mask = selector.get_support()

# 最终被选中的特征名称
selected_features = [feat for feat, keep in zip(feature_cols, mask) if keep]

print("保留下来的特征共有：", len(selected_features))
print(selected_features)

import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. 用最佳模型预测概率
y_prob = cross_val_predict(search.best_estimator_, X, y, cv=skf, method='predict_proba')[:, 1]

# 2. 基于阈值 0.6 生成新的预测标签
threshold = 0.58
y_pred = (y_prob >= threshold).astype(int)

# 2. 混淆矩阵
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=search.best_estimator_.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 3. 计算 MCC
mcc = matthews_corrcoef(y, y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# 4. (可选) 详细报告
print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4))

import joblib

# 假设 search 已经 fit 完毕
best_model = search.best_estimator_

# 把模型存为文件，保存在当前工作目录
model_path = r"D:\Me\IMB\Data\Yuhao\models\rf_classfication_iFeature_model_tuning.pkl"
joblib.dump(best_model, model_path)

print(f"模型已保存至：{model_path}")

import pandas as pd
from sklearn.model_selection import cross_val_predict

# ...（前面：模型训练、search 已完成）...

# -------------------------------------------------
# A. 重新读取原始 CSV，只拿 peptide_sequence 列
# -------------------------------------------------
pos_seq_csv = r"Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR3_filtered.csv"
neg_seq_csv = r"Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_monotonic_decrease_R1toR4.csv"

seq_pos = pd.read_csv(pos_seq_csv, usecols=['peptide_sequence'])['peptide_sequence']
seq_neg = pd.read_csv(neg_seq_csv, usecols=['peptide_sequence'])['peptide_sequence']

# 与特征表行顺序一致地拼接
seqs = pd.concat([seq_pos, seq_neg], ignore_index=True)

# 安全检查：长度必须一致
assert len(seqs) == len(df_all), "肽序列条数与特征表行数不符！"

# -------------------------------------------------
# B. 用最佳模型做 cross-val 预测
# -------------------------------------------------
model = search.best_estimator_
y_prob = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
y_pred = cross_val_predict(model, X, y, cv=skf, method='predict')

# -------------------------------------------------
# C. 汇总并保存结果
# -------------------------------------------------
results_df = pd.DataFrame({
    'peptide_sequence': seqs,         # ← 真实肽段序列
    'true_label'     : df_all['label'],
    'pred_label'     : y_pred,
    'pred_prob'      : y_prob
})

print(results_df.head())

output_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\prediction_results.csv"
results_df.to_csv(output_path, index=False)
print(f"已保存预测结果到：{output_path}")
