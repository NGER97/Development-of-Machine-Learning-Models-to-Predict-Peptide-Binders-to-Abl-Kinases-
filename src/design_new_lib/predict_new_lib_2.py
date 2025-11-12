import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Optional

# ===== 路径配置 =====
model_path = r"D:\Me\IMB\Data\Yuhao\models\rf_classfication_iFeature_model.pkl"
feat_path  = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\lib_singlemut_ALL_triples_ALLfeatures.csv"
seq_path   = r"Yuhao\NGS\Abl_L1\06.LibraryDesign\triples_mut\lib_singlemut_ALL_triples.csv"
out_path   = r"D:\Me\IMB\Data\Yuhao\models\predictions_lib_singlemut_ALL_triples.csv"

CHUNK_SIZE = 100_000
THRESHOLD  = 0.5
FLOAT_FMT  = "%.6f"

print("[INFO] Loading model ...")
model = joblib.load(model_path)

def expected_n_features_from_pipeline(est) -> Optional[int]:
    """从 Pipeline/步骤里尽力取到期望特征数 n_features_in_。"""
    # 1) 直接取
    n = getattr(est, "n_features_in_", None)
    if isinstance(n, (int, np.integer)):
        return int(n)
    # 2) imblearn/sklearn Pipeline: 遍历 steps
    for attr in ("steps", "named_steps"):
        steps = getattr(est, attr, None)
        if steps:
            # steps 可能是 list[ (name, obj), ... ] 或 dict
            if isinstance(steps, dict):
                it = steps.items()
            else:
                it = steps
            # 尽量从前处理器（如 scaler）拿
            for name, obj in it:
                n = getattr(obj, "n_features_in_", None)
                if isinstance(n, (int, np.integer)):
                    return int(n)
    # 3) 尝试最后一步分类器
    last = None
    if hasattr(est, "named_steps"):
        try:
            last = list(est.named_steps.values())[-1]
        except Exception:
            last = None
    elif hasattr(est, "steps"):
        try:
            last = est.steps[-1][1]
        except Exception:
            last = None
    if last is not None:
        n = getattr(last, "n_features_in_", None)
        if isinstance(n, (int, np.integer)):
            return int(n)
    return None

def resolve_pos_index(est):
    """确定 predict_proba 的正类列索引。"""
    classes = getattr(est, "classes_", None)
    if classes is None and hasattr(est, "named_steps"):
        try:
            last = list(est.named_steps.values())[-1]
            classes = getattr(last, "classes_", None)
        except Exception:
            classes = None
    if classes is not None and len(classes) == 2:
        classes = list(classes)
        for v in (1, "binder", "positive", True):
            if v in classes:
                return classes.index(v)
    return -1

# ===== 准备列名 =====
feat_header = pd.read_csv(feat_path, nrows=0)
feat_all_cols = list(feat_header.columns)

seq_header = pd.read_csv(seq_path, nrows=0)
seq_col = "peptide_sequence" if "peptide_sequence" in seq_header.columns else seq_header.columns[0]

meta_cols = {
    "peptide_sequence", "y", "target", "score",
    "R1_counts","R2_counts","R3_counts","R4_counts","R5_counts",
    "R1_frequency","R2_frequency","R3_frequency","R4_frequency","R5_frequency"
    # 注意：此处**不**把 'class' 放进 meta，便于必要时当特征使用
}

expected_n = expected_n_features_from_pipeline(model)
print(f"[INFO] expected_n_features (from pipeline): {expected_n}")

# 优先：用模型记录的列名
if hasattr(model, "feature_names_in_"):
    feat_cols = [c for c in model.feature_names_in_ if c in feat_all_cols]
    missing  = [c for c in model.feature_names_in_ if c not in feat_all_cols]
    if missing:
        raise ValueError(f"[ERROR] 特征表缺少模型需要的列（示例）: {missing[:8]} ... 共 {len(missing)} 列")
else:
    # 回退策略：先不含 'class'
    feat_cols_no_class = [c for c in feat_all_cols if c not in meta_cols and c != "class"]
    # 如果存在 'class' 列，备用列表加上它
    feat_cols_with_class = [c for c in feat_all_cols if c not in meta_cols]  # 这里若有 'class' 会包含

    # 根据 expected_n 自动选择
    if expected_n is not None:
        if len(feat_cols_no_class) == expected_n:
            feat_cols = feat_cols_no_class
            print("[INFO] Using features WITHOUT 'class' (counts match).")
        elif len(feat_cols_with_class) == expected_n:
            feat_cols = feat_cols_with_class
            print("[INFO] Using features WITH 'class' (counts match).")
        elif (len(feat_cols_with_class) - 1 == len(feat_cols_no_class)
              and len(feat_cols_no_class) + 1 == expected_n
              and "class" in feat_all_cols):
            # 典型情形：训练时把 class 也当特征；现在自动补上
            feat_cols = feat_cols_with_class
            print("[INFO] Adjusted to include 'class' to match expected_n.")
        else:
            # 无法匹配，给出提示
            raise ValueError(
                f"[ERROR] 无法匹配期望特征数：expected={expected_n}, "
                f"no_class={len(feat_cols_no_class)}, with_class={len(feat_cols_with_class)}.\n"
                f"请检查训练时是否把 'class' 当作特征，或提供训练时的特征列清单。"
            )
    else:
        # 没有 expected_n，只能默认不用 class；如后续报特征数不匹配再提示
        feat_cols = feat_cols_no_class
        print("[WARN] 未能从模型推断 expected_n_features，默认不使用 'class'。")

pos_idx = resolve_pos_index(model)

# ===== 输出头 =====
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", newline="") as f:
    f.write("peptide_sequence,pred_label,pred_score\n")

print("[INFO] Start streaming predict ...")
feat_reader = pd.read_csv(feat_path, usecols=feat_cols, chunksize=CHUNK_SIZE, low_memory=True)
seq_reader  = pd.read_csv(seq_path,  usecols=[seq_col],   chunksize=CHUNK_SIZE, low_memory=True)

total = 0
for i, (feat_chunk, seq_chunk) in enumerate(zip(feat_reader, seq_reader), start=1):
    if len(feat_chunk) != len(seq_chunk):
        raise ValueError(f"[ERROR] 第{i}块行数不一致：features={len(feat_chunk)} vs seqs={len(seq_chunk)}")

    # 转 float32 以省内存
    X = feat_chunk.astype(np.float32, copy=False)

    # === 预测 ===
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        y_score = proba[:, pos_idx]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X).astype(float)
        y_score = 1.0 / (1.0 + np.exp(-raw))
    else:
        pred = model.predict(X)
        y_score = (pred == 1).astype(float)

    y_pred = y_score >= THRESHOLD

    out_chunk = pd.DataFrame({
        "peptide_sequence": seq_chunk[seq_col].astype(str).values,
        "pred_label": y_pred,
        "pred_score": y_score
    })
    out_chunk.to_csv(out_path, mode="a", header=False, index=False, float_format=FLOAT_FMT)

    total += len(out_chunk)
    if i % 10 == 0:
        print(f"[INFO] written {total:,} rows ...")

# 完整性检查
try:
    next(feat_reader)
    raise RuntimeError("[ERROR] 特征文件还有剩余分块，序列文件已耗尽（两文件行数不一致）。")
except StopIteration:
    pass
try:
    next(seq_reader)
    raise RuntimeError("[ERROR] 序列文件还有剩余分块，特征文件已耗尽（两文件行数不一致）。")
except StopIteration:
    pass

print(f"[DONE] Saved: {out_path}  |  total rows = {total:,}")
