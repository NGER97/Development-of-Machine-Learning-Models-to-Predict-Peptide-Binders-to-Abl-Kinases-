import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report


from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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
    

    
# 1. 加载模型
model_path = r"Yuhao\models\rf_boruta_model.pkl"
model = joblib.load(model_path)

# 2. 读取测试集特征
test_path = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_generalization_test_ALLfeatures.csv"
df_test = pd.read_csv(test_path)

# 假设 df_test 中包含标识列 (如 'Protein' 或 'peptide_sequence') 和所有特征列。
# 这里只保留特征列用于预测，自动排除非特征列：
exclude = ['Protein', 'peptide_sequence']  # 如有其它非特征列也加到这里
feature_cols = [c for c in df_test.columns if c not in exclude]

X_test = df_test[feature_cols].values

# 3. 预测概率 & 按阈值生成标签
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.58
y_pred = (y_prob >= threshold).astype(int)

# 4. 因为这一测试集全是负例，我们用全 0 的真标签来计算
y_true = np.zeros_like(y_pred)

# 5. 计算混淆矩阵和 MCC
cm = confusion_matrix(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print("Confusion Matrix (threshold=0.6):")
print(cm)
print(f"MCC: {mcc:.4f}")

# 如需详细报告（注意没有正例时 recall/precision 可能为 0）
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

# 6. 把预测结果写回到 CSV
df_test['pred_prob']  = y_prob
df_test['pred_label'] = y_pred
out_path = test_path.replace(".csv", "_with_preds.csv")
df_test.to_csv(out_path, index=False)
print(f"\n预测结果已保存至：{out_path}")
