# ===============================================================
#  evaluate_generalization.py
#  使用已训练好的 RF 模型评估新数据集
# ===============================================================
import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- 1. 路径 ----------
MODEL_PATH = r"D:\Me\IMB\Data\Yuhao\models\rf_R1toR4_significant.pkl"
FEAT_PATH  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\regression\BB_add_pseudocount_remove_r5_ALLfeatures.csv"
SCORE_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_remove_r5.csv"
OUT_PRED   = r"D:\Me\IMB\Data\Yuhao\models\rf_predictions_generalization_test.csv"

# ---------- 2. 读取模型 ----------
bundle = joblib.load(MODEL_PATH)
if isinstance(bundle, dict):                 # 我们训练阶段 dump 了 dict
    model     = bundle["model"]
    feat_cols = bundle["feat_cols"]
else:                                        # 若只保存了 estimator
    model     = bundle
    feat_cols = None                         # 稍后重新推断

print(f"[INFO] 模型加载完成：{model.__class__.__name__}")

# ---------- 3. 读取特征 & score ----------
df_feat  = pd.read_csv(FEAT_PATH)
df_score = pd.read_csv(SCORE_PATH, usecols=["score"])

if len(df_feat) != len(df_score):
    raise ValueError("行数不一致，无法按顺序合并！")

df_feat["score"] = df_score["score"].values

# ---------- 4. 构造特征矩阵 ----------
drop_cols_default = {"Protein", "protein", "Score", "score", "class"}

# 如果训练时未保存 feat_cols，则依照默认规则自动推断
if feat_cols is None:
    feat_cols = [c for c in df_feat.columns if c not in drop_cols_default]
    print(f"[WARN] feat_cols 未从模型文件恢复，按默认规则推断共 {len(feat_cols)} 列。")

# 检查缺失列；若缺失用 0 填充
missing = [c for c in feat_cols if c not in df_feat.columns]
if missing:
    print(">>> 评估集缺少以下列，将以 0 填充：", missing)
    for c in missing:
        df_feat[c] = 0.0

# 保证列顺序与训练时一致，只留需要的列
X_test = df_feat.reindex(columns=feat_cols).astype(np.float32).values
y_true = df_feat["score"].values

# ---------- 5. 预测 & 评价 ----------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print("\n=== Generalization Test ===")
print(f"Samples          : {len(y_true)}")
print(f"RMSE             : {rmse:.4f}")
print(f"MAE              : {mae:.4f}")
print(f"R²               : {r2:.4f}")

# ---------- 6. 保存预测结果 ----------
out_df = pd.DataFrame({"true_score": y_true, "pred_score": y_pred})
os.makedirs(os.path.dirname(OUT_PRED), exist_ok=True)
out_df.to_csv(OUT_PRED, index=False)
print(f"\n[Save] 预测结果已写入 → {OUT_PRED}")
