import os
import pandas as pd

# ---- 路径与参数 ----
PRED_CSV = r"Yuhao\NGS\Abl_L1\06.LibraryDesign\pairs_singlemut\lib_singlemut_ALL_pairs.csv"
FEAT_CSV = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\lib_singlemut_ALL_pairs_ALLfeatures.csv"
OUT_CSV  = r"D:\Me\IMB\Data\Yuhao\models\predictions_lib_singlemut_ALL_pairs_ALLfeatures.csv"

THRESH   = 0.8
CHUNK    = 10_000      # 特征列很多，块开小一点更稳（可按内存调 5k–20k）
FLOAT_FMT = "%.6f"

# （可选）如果你只想挑部分特征，列在这里能大幅降内存/文件体积
SELECT_FEATURES = None
# 例如：SELECT_FEATURES = ["class","binary::f1","binary::f2","AAC::f1","DPC::f1"]

# ---- 准备输出表头：预测三列 + 特征列（去重 peptide_sequence）----
pred_cols = ["peptide_sequence", "pred_label", "pred_score"]

feat_header = pd.read_csv(FEAT_CSV, nrows=0)
feat_all_cols = list(feat_header.columns)

if SELECT_FEATURES is None:
    feat_usecols = feat_all_cols
else:
    missing = [c for c in SELECT_FEATURES if c not in feat_all_cols]
    if missing:
        raise ValueError(f"这些特征列在特征表中不存在: {missing[:8]} ...")
    feat_usecols = SELECT_FEATURES

# 输出列顺序：预测三列 + (特征列去掉重复的 peptide_sequence)
feat_out_cols = [c for c in feat_usecols if c != "peptide_sequence"]
out_header = pred_cols + feat_out_cols

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    f.write(",".join(out_header) + "\n")

# ---- 流式同步读取、筛选并拼接 ----
pred_reader = pd.read_csv(PRED_CSV, usecols=pred_cols, chunksize=CHUNK, low_memory=True)
feat_reader = pd.read_csv(FEAT_CSV,  usecols=feat_usecols, chunksize=CHUNK, low_memory=True)

total = kept = 0
for i, (pred_chunk, feat_chunk) in enumerate(zip(pred_reader, feat_reader), start=1):
    if len(pred_chunk) != len(feat_chunk):
        raise ValueError(f"[ERROR] 第{i}块行数不一致：pred={len(pred_chunk)} vs feat={len(feat_chunk)}")

    total += len(pred_chunk)

    # 确保数值类型
    pred_chunk["pred_score"] = pd.to_numeric(pred_chunk["pred_score"], errors="coerce")
    mask = pred_chunk["pred_score"] > THRESH

    if mask.any():
        # 选出对应行（按**同一行位次**对齐）
        pred_sel = pred_chunk.loc[mask, pred_cols]
        # 特征侧去掉 peptide_sequence，避免重复列（保留预测侧的序列列）
        feat_sel = feat_chunk.loc[mask, feat_out_cols]

        out = pd.concat([pred_sel.reset_index(drop=True),
                         feat_sel.reset_index(drop=True)], axis=1)
        out.to_csv(OUT_CSV, mode="a", header=False, index=False, float_format=FLOAT_FMT)
        kept += len(out)

    if i % 10 == 0:
        print(f"[INFO] processed {total:,} rows, kept {kept:,} (> {THRESH})")

# 完整性检查：若两文件总行数不同，zip 会提前结束
try:
    next(pred_reader)
    raise RuntimeError("[ERROR] 预测文件还有剩余分块，特征文件已耗尽（两文件行数不一致）。")
except StopIteration:
    pass
try:
    next(feat_reader)
    raise RuntimeError("[ERROR] 特征文件还有剩余分块，预测文件已耗尽（两文件行数不一致）。")
except StopIteration:
    pass

print(f"[DONE] total={total:,}, kept={kept:,}  →  {OUT_CSV}")
