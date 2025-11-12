"""
Baseline RMSE 与 R²（均值预测）
Author: Yuhao Rao
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# ===== 数据路径 =====
FEATURE_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score_protBERT.parquet"
RAW_CSV      = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_ER&Score.csv"

# ===== 读取 score =====
# score 只在 RAW_CSV 里，直接读一列即可
df_raw = pd.read_csv(RAW_CSV, usecols=["score"])
y = df_raw["score"].values          # ndarray，范围 0–1

# ===== 基线 = 全局均值预测 =====
y_pred_baseline = np.full_like(y, fill_value=y.mean())

# ===== 评估 =====
baseline_rmse = np.sqrt(mean_squared_error(y, y_pred_baseline))
baseline_r2   = r2_score(y, y_pred_baseline)      # 均值预测的 R² 理论上为 0

print(f"Baseline RMSE  (mean predictor): {baseline_rmse:.4f}")
print(f"Baseline R²    (mean predictor): {baseline_r2:.4f}")
