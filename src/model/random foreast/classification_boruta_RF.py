#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Boruta 做特征选择 + RandomForest 分类
================================================
1. 读取阳性 / 阴性特征表，打标签
2. RandomUnderSampler 下采样平衡
3. MinMaxScaler 归一化
4. Boruta 特征选择
5. RandomForest 分类 + RandomizedSearchCV 超参搜索
6. CV 评估、混淆矩阵、MCC、分类报告
7. 保存最佳模型、保存交叉验证预测结果（含肽段序列）
"""

# -------------------------------------------------
# 参数区 —— 根据需要自行调整
# -------------------------------------------------
POS_FEATURE_CSV = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_R1toR3_filtered_ALLfeatures.csv"
NEG_FEATURE_CSV = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_monotonic_decrease_R1toR4_ALLfeatures.csv"

POS_SEQ_CSV = r"Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR3_filtered.csv"
NEG_SEQ_CSV = r"Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_monotonic_decrease_R1toR4.csv"

MODEL_OUT   = r"D:\Me\IMB\Data\Yuhao\models\rf_boruta_model.pkl"
PRED_OUT    = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\prediction_results.csv"

N_JOBS      = -1    # 并行核心数
RANDOM_SEED = 42
# -------------------------------------------------

# ================ 基本包 & 环境 ================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r"`BaseEstimator\._check_n_features`")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r"'force_all_finite' was renamed to 'ensure_all_finite'")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             classification_report, ConfusionMatrixDisplay)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import joblib

# ================ 自定义 Boruta Transformer ================
class BorutaSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators='auto', max_iter=100,
                 random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs                  # 仍保留给 RandomForest

    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=self.n_jobs,               # 并行开在这里
            class_weight='balanced'
        )
        self.boruta_ = BorutaPy(
            estimator=rf,
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0                         # <─ ✅ 删除 n_jobs
        )
        self.boruta_.fit(np.asarray(X), np.asarray(y))
        return self

    def transform(self, X):
        return self.boruta_.transform(np.asarray(X))

    def get_support(self):
        return self.boruta_.support_

# ================ 1. 读取特征 & 标签 ================
df_pos = pd.read_csv(POS_FEATURE_CSV)
df_neg = pd.read_csv(NEG_FEATURE_CSV)
df_pos['label'] = 1
df_neg['label'] = 0

df_all = pd.concat([df_pos, df_neg], ignore_index=True)

# 提取特征列（去掉 label）
feature_cols = [c for c in df_all.columns if c not in ('label',)]
X = df_all[feature_cols].values
y = df_all['label'].values

print(f"样本总数: {len(y)} | 特征维度: {X.shape[1]}")

# ================ 2. 交叉验证 & 下采样 ================
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=RANDOM_SEED)
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_SEED)

# ================ 3. 构建 Pipeline ================
pipe = ImbPipeline([
    ('undersample', rus),
    ('scaler', MinMaxScaler()),
    ('feat_sel', BorutaSelector(random_state=RANDOM_SEED, max_iter=100, n_jobs=N_JOBS)),
    ('clf', RandomForestClassifier(random_state=RANDOM_SEED))
])

# ================ 4. 超参数搜索 ================
param_dist = {
    'feat_sel__max_iter': [50, 100, 150],
    'feat_sel__n_estimators': ['auto', 200, 500],
    'clf__n_estimators': [100, 300, 500],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_leaf': [1, 2, 5]
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=30,
    cv=skf,
    scoring='roc_auc',
    n_jobs=N_JOBS,
    verbose=2,
    random_state=RANDOM_SEED,
    refit=True
)

# ================ 5. 训练 & CV 结果 ================
search.fit(X, y)
print("\n=== 最佳模型参数 ===")
print(search.best_params_)
print(f"最佳 ROC AUC: {search.best_score_:.4f}")

# 交叉验证多指标
metrics = cross_val_score(search.best_estimator_, X, y, cv=skf,
                          scoring='f1', n_jobs=N_JOBS)
print(f"F1 (6-fold CV) : {metrics.mean():.4f} ± {metrics.std():.4f}")

# 保留特征数量
selector = search.best_estimator_.named_steps['feat_sel']
n_feat_kept = selector.transform(X).shape[1]
print(f"Boruta 保留下来的特征数: {n_feat_kept}")

# ================ 6. CV 预测 & 评估 =================
y_prob = cross_val_predict(search.best_estimator_, X, y, cv=skf,
                           method='predict_proba', n_jobs=N_JOBS)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)   # 默认阈值 0.5，可自行调整

cm = confusion_matrix(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)

print("\nConfusion Matrix:\n", cm)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print("\nClassification Report:\n",
      classification_report(y, y_pred, digits=4))

# ================ 7. 可视化混淆矩阵 =================
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=search.best_estimator_.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Boruta RF)")
plt.show()

# ================ 8. 保存模型 =================
joblib.dump(search.best_estimator_, MODEL_OUT)
print(f"模型已保存：{MODEL_OUT}")

# ================ 9. 保存交叉验证预测结果 ================
# 重新读取肽序列列
seq_pos = pd.read_csv(POS_SEQ_CSV, usecols=['peptide_sequence'])['peptide_sequence']
seq_neg = pd.read_csv(NEG_SEQ_CSV, usecols=['peptide_sequence'])['peptide_sequence']
seqs = pd.concat([seq_pos, seq_neg], ignore_index=True)
assert len(seqs) == len(y)

results_df = pd.DataFrame({
    'peptide_sequence': seqs,
    'true_label': y,
    'pred_label': y_pred,
    'pred_prob': y_prob
})
results_df.to_csv(PRED_OUT, index=False)
print(f"交叉验证预测结果已保存：{PRED_OUT}")
