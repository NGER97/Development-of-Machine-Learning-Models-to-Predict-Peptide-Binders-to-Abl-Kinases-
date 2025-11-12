import os
import numpy as np
import pandas as pd
import joblib

# ===== 配置路径 =====
model_path = r"D:\Me\IMB\Data\Yuhao\models\rf_classfication_iFeature_model.pkl"
feat_path  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\lib_expand_2_ALLfeatures.csv"
seq_path   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.LibraryDesign\lib_expand_2.csv"
out_path   = r"D:\Me\IMB\Data\Yuhao\models\predictions_lib_expand_2.csv"

# ===== 读取 =====
print("Loading model & data ...")
model   = joblib.load(model_path)
df_feat = pd.read_csv(feat_path)
df_seq  = pd.read_csv(seq_path)

# 规范序列表头为 peptide_sequence
if "peptide_sequence" not in df_seq.columns:
    # 若只有一列，则将其视为 peptide_sequence
    if df_seq.shape[1] == 1:
        df_seq.columns = ["peptide_sequence"]
    else:
        # 否则默认第一列为序列
        first_col = df_seq.columns[0]
        df_seq = df_seq.rename(columns={first_col: "peptide_sequence"})[["peptide_sequence"]]

# 如果特征表没有序列列，就按顺序对齐；有则做内连接以确保一致
if "peptide_sequence" in df_feat.columns:
    df_feat = df_feat.merge(df_seq[["peptide_sequence"]], on="peptide_sequence", how="inner")
else:
    if len(df_feat) != len(df_seq):
        raise ValueError("特征表与序列表行数不一致，且特征表缺少 peptide_sequence 列，无法按顺序对齐。")
    df_feat = df_feat.copy()
    df_feat["peptide_sequence"] = df_seq["peptide_sequence"].values

# ===== 选择特征列（优先用模型保存的列名）=====
meta_cols = {
    "peptide_sequence", "label", "y", "target", "score",
    "R1_counts","R2_counts","R3_counts","R4_counts","R5_counts",
    "R1_frequency","R2_frequency","R3_frequency","R4_frequency","R5_frequency"
}

def numeric_cols(df):
    return [c for c in df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

if hasattr(model, "feature_names_in_"):
    feat_cols = [c for c in model.feature_names_in_ if c in df_feat.columns]
    missing  = [c for c in model.feature_names_in_ if c not in df_feat.columns]
    if missing:
        raise ValueError(f"特征表缺少模型需要的列：{missing[:5]} ...（共{len(missing)}列）")
else:
    feat_cols = numeric_cols(df_feat)
    if not feat_cols:
        raise ValueError("未检测到可用的数值型特征列。")

X = df_feat[feat_cols].values.astype(np.float32)

# ===== 预测概率（取“正类”的概率）=====
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)
    # 选择正类列：优先找 1 / 'binder' / 'positive'；否则取最后一列（常见为正类）
    classes = getattr(model, "classes_", None)
    pos_idx = -1  # 缺省取最后一列
    if classes is not None and len(classes) == 2:
        classes = list(classes)
        for cand in (1, "binder", "positive", True):
            if cand in classes:
                pos_idx = classes.index(cand)
                break
    y_score = proba[:, pos_idx]
elif hasattr(model, "decision_function"):
    raw = model.decision_function(X).astype(float)
    # 将 margin 转成 0-1 概率感得分（Sigmoid）
    y_score = 1.0 / (1.0 + np.exp(-raw))
else:
    # 只能拿到 hard label，则转为 0/1 并作为分数；阈值预测同它一致
    pred = model.predict(X)
    y_score = (pred == 1).astype(float)

# True/False 标签（默认阈值 0.5）
y_pred = y_score >= 0.5

# ===== 输出 =====
df_out = pd.DataFrame({
    "peptide_sequence": df_feat["peptide_sequence"].values,
    "pred_label": y_pred,            # True / False
    "pred_score": y_score            # 概率/置信度分数
})

df_out.to_csv(out_path, index=False)
print(f"Saved: {out_path}  |  n={len(df_out)}")
print(df_out.head())
