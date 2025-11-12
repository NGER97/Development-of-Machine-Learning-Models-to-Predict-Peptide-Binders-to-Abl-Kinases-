# -*- coding: utf-8 -*-
"""
PCA 对比：两份训练集 vs 大规模预测集（分块、增量PCA、密度+散点叠加）
稳健版：
- 自动嗅探分隔符(, / \t / ; / |) 与编码(BOM处理)
- 列名规范化(NFKC、去零宽、压缩冒号、去 f 序号前导零)
- 以首文件表头为基准选择特征(按前缀)，其余文件对齐；缺列统一剔除
- 预测集严格水库采样
依赖：pandas, numpy, scikit-learn, matplotlib
"""

import re
import csv
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from pathlib import Path

# -------------------- 路径与配置 --------------------
f_train1 = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_monotonic_decrease_R1toR4_ALLfeatures.csv"
f_train2 = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_R1toR3_filtered_ALLfeatures.csv"
f_pred   = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\lib_expand_2_ALLfeatures.csv"

# 仅把这些前缀开头的列当作特征
FEATURE_PREFIXES = [
    "binary::",
    "AAC::",
    "DPC::",
    "CKSAAGP_gap3::",
    "PAAC_l5_w0.05::",
    # 如有其它族（ProtBERT:: 等），直接在此追加
]

# 明确排除（即使前缀匹配也不纳入）
EXCLUDE_COLS_RAW = {
    'class','Class',
    'seq','sequence','Sequence','SEQ',
    'label','Label','target','Target','y','Y',
    'set','Set','source','Source',
    'id','ID','Id',
    'Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','index','Index'
}

CHUNK_SIZE = 20000
PRED_SAMPLE_MAX = 200_000
N_COMPONENTS = 2

OUT_PNG     = "pca_all.png"
OUT_PARQUET = "pca_coords_sampled.parquet"
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
# ----------------------------------------------------

# ========= CSV 读入参数嗅探 =========
def detect_csv_format(fp, sniff_bytes=65536):
    """
    简单嗅探分隔符与编码（优先 UTF-8-SIG；失败退回 latin-1），
    分隔符在 [',', '\\t', ';', '|'] 中选“第一行计数最多”的那个。
    """
    # 读原始字节
    with open(fp, 'rb') as f:
        raw = f.read(sniff_bytes)

    # 尝试 UTF-8-SIG（可自动去BOM）
    try:
        text = raw.decode('utf-8-sig')
        enc  = 'utf-8-sig'
    except UnicodeDecodeError:
        text = raw.decode('latin-1')
        enc  = 'latin-1'

    first_line = text.splitlines()[0] if text else ""
    cand = [',', '\t', ';', '|']
    counts = {c: first_line.count(c) for c in cand}
    sep = max(counts, key=counts.get) if any(counts.values()) else ','

    # python 引擎 + on_bad_lines='skip' 更稳（慢一点但安全）
    kwargs = dict(sep=sep, engine='python', encoding=enc, on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL)
    print(f"[Info] 读取设置 {Path(fp).name}: sep='{sep}', encoding={enc}, header_tokens={counts.get(sep,0)}")
    return kwargs

CSV_READ_KW = {}
def get_csv_kw(fp):
    if fp not in CSV_READ_KW:
        CSV_READ_KW[fp] = detect_csv_format(fp)
    return CSV_READ_KW[fp]

# ========= 列名规范化（用于容错映射） =========
_fnum_pat = re.compile(r'^(.*::f)(\d+)$')

def norm_name(col: str) -> str:
    s = unicodedata.normalize('NFKC', str(col)).strip()
    s = s.replace('\u200b', '')               # 零宽空格
    s = s.replace('：', ':')                   # 全角冒号 -> 半角
    s = re.sub(r':{2,}', '::', s)             # 多冒号压缩成 '::'
    s = re.sub(r'(?<=\w):(f\d+)$', r'::\1', s) # 单冒号 -> '::'（仅末尾 fN 场景）
    m = _fnum_pat.match(s)                    # f001 -> f1
    if m:
        s = m.group(1) + str(int(m.group(2)))
    return s

def header(fp):
    kw = get_csv_kw(fp)
    cols = pd.read_csv(fp, nrows=0, low_memory=False, **kw).columns.tolist()
    return cols

def starts_with_any(s: str, prefixes) -> bool:
    return any(s.startswith(p) for p in prefixes)

# ========= 基于“首文件表头”的特征列选择与对齐 =========
def build_base_feature_list(base_fp, prefixes, exclude_raw):
    cols = header(base_fp)
    exclude_norm = {norm_name(x) for x in exclude_raw}
    base_feat = []
    for c in cols:
        cn = norm_name(c)
        if cn in exclude_norm:
            continue
        if starts_with_any(cn, prefixes):
            base_feat.append(c)   # 保留原列名 & 首文件顺序
    if not base_feat:
        raise ValueError("在首文件中未匹配到任何特征列，请检查 FEATURE_PREFIXES/EXCLUDE_COLS_RAW。")
    print(f"[Info] 基准特征列数: {len(base_feat)}；前10列: {base_feat[:10]}")
    return base_feat

def align_usecols_to_other_file(base_usecols, other_fp):
    """
    将“基准文件”的 usecols（原名与顺序）映射到 other_fp：
      1) 若列名 exact match，直接使用；
      2) 否则用规范化名匹配（norm_name），成功则用对方原列名；
      3) 若仍找不到，记为 missing，最后从所有文件的 usecols 中统一剔除。
    """
    other_cols = header(other_fp)
    other_set  = set(other_cols)
    other_norm_map = {}
    for oc in other_cols:
        other_norm_map.setdefault(norm_name(oc), oc)

    mapped = []
    missing = set()
    for bc in base_usecols:
        if bc in other_set:
            mapped.append(bc)
        else:
            bn = norm_name(bc)
            oc = other_norm_map.get(bn, None)
            if oc is not None:
                mapped.append(oc)
            else:
                mapped.append(None)
                missing.add(bc)

    if missing:
        miss_show = list(missing)[:10]
        print(f"[Warn] {other_fp} 缺失 {len(missing)} 列（示例）: {miss_show}")
    return mapped, missing

def make_fully_aligned_usecols(base_fp, other_fps, prefixes, exclude_raw):
    base_usecols = build_base_feature_list(base_fp, prefixes, exclude_raw)

    usecols_map = {base_fp: base_usecols[:]}

    all_missing = set()
    temp_maps = {}
    for fp in other_fps:
        mapped, missing = align_usecols_to_other_file(base_usecols, fp)
        temp_maps[fp] = mapped
        all_missing |= missing

    if all_missing:
        print(f"[Info] 有 {len(all_missing)} 个基准列在至少一个文件缺失，将统一剔除。")
        base_filtered = [c for c in base_usecols if c not in all_missing]
        usecols_map[base_fp] = base_filtered

        for fp in other_fps:
            mapped = temp_maps[fp]
            mapped_filtered = [mc for bc, mc in zip(base_usecols, mapped) if (bc not in all_missing and mc is not None)]
            usecols_map[fp] = mapped_filtered
    else:
        for fp in other_fps:
            usecols_map[fp] = temp_maps[fp]

    lengths = [len(v) for v in usecols_map.values()]
    if len(set(lengths)) != 1 or lengths[0] == 0:
        # 这里给出更直观的诊断
        raise RuntimeError(
            f"对齐后的列数不一致或为0：{lengths}\n"
            f"请确认 {Path(f_pred).name} 是否为同一版本特征导出（分隔符/编码/前缀）。"
        )
    print(f"[Info] 对齐完成；每个文件特征列数：{lengths[0]}")
    return usecols_map

# ========= 数据流 & 变换 =========
def stream_numeric_chunks(fp, usecols, chunksize=CHUNK_SIZE):
    kw = get_csv_kw(fp)
    for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False, **kw):
        for c in chunk.columns:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        X = chunk.fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        yield X

def partial_fit_scaler_on_files(scaler, files, usecols_map):
    for fp in files:
        for X in stream_numeric_chunks(fp, usecols_map[fp]):
            if X.size == 0:
                continue
            scaler.partial_fit(X)

def partial_fit_ipca_on_files(ipca, scaler, files, usecols_map):
    for fp in files:
        for X in stream_numeric_chunks(fp, usecols_map[fp]):
            if X.size == 0:
                continue
            Xs = scaler.transform(X)
            ipca.partial_fit(Xs)

def transform_full(ipca, scaler, fp, usecols):
    outs = []
    for X in stream_numeric_chunks(fp, usecols):
        if X.size == 0:
            continue
        Z = ipca.transform(scaler.transform(X))
        outs.append(Z)
    return np.concatenate(outs, axis=0) if outs else np.empty((0, 2))

def reservoir_sample_transform(ipca, scaler, fp, usecols, k_max):
    if not k_max or k_max <= 0:
        return transform_full(ipca, scaler, fp, usecols)

    S = None
    s_size = 0
    n_seen = 0
    for X in stream_numeric_chunks(fp, usecols):
        if X.size == 0:
            continue
        Z = ipca.transform(scaler.transform(X))
        m = Z.shape[0]
        for i in range(m):
            n_seen += 1
            zi = Z[i]
            if s_size < k_max:
                if S is None:
                    S = np.empty((k_max, 2), dtype=np.float64)
                S[s_size] = zi
                s_size += 1
            else:
                j = rng.integers(0, n_seen)
                if j < k_max:
                    S[j] = zi
    if S is None:
        return np.empty((0, 2))
    return S[:s_size].copy()

# ========= 主流程 =========
def main():
    files = [f_train1, f_train2, f_pred]

    # 打印每个文件的嗅探结果 & 列数，帮助诊断
    for fp in files:
        _ = get_csv_kw(fp)
        cols = header(fp)
        print(f"[Info] {Path(fp).name} 列计数: {len(cols)}（前5列: {cols[:5]}）")

    # 1) 以首文件为基准，对齐三份文件的特征列（顺序完全一致）
    usecols_map = make_fully_aligned_usecols(
        base_fp=f_train1,
        other_fps=[f_train2, f_pred],
        prefixes=FEATURE_PREFIXES,
        exclude_raw=EXCLUDE_COLS_RAW
    )

    # 2) 在两份训练集上拟合标准化
    scaler = StandardScaler(with_mean=True, with_std=True)
    partial_fit_scaler_on_files(scaler, [f_train1, f_train2], usecols_map)
    print("[Info] StandardScaler 拟合完成。")

    # 3) 在训练集上增量拟合 PCA
    ipca = IncrementalPCA(n_components=N_COMPONENTS)
    partial_fit_ipca_on_files(ipca, scaler, [f_train1, f_train2], usecols_map)
    print("[Info] IncrementalPCA 拟合完成。")

    # 4) 变换：训练集全量，预测集水库采样
    Z_train1 = transform_full(ipca, scaler, f_train1, usecols_map[f_train1])
    Z_train2 = transform_full(ipca, scaler, f_train2, usecols_map[f_train2])
    Z_pred   = reservoir_sample_transform(ipca, scaler, f_pred,  usecols_map[f_pred], k_max=PRED_SAMPLE_MAX)

    print(f"[Info] 降维完成：train1={Z_train1.shape}, train2={Z_train2.shape}, pred(sampled)={Z_pred.shape}")

    # 5) 导出 + 绘图
    df_t1 = pd.DataFrame(Z_train1, columns=["PC1", "PC2"]); df_t1["source"] = "Train_R1toR4"
    df_t2 = pd.DataFrame(Z_train2, columns=["PC1", "PC2"]); df_t2["source"] = "Train_R1toR3_filtered"
    df_pr = pd.DataFrame(Z_pred,   columns=["PC1", "PC2"]); df_pr["source"] = "Lib_Expand_2 (sampled)"
    df_all_small = pd.concat([df_t1, df_t2, df_pr], axis=0, ignore_index=True)
    df_all_small.to_parquet(OUT_PARQUET, index=False)
    print(f"[Info] 已导出降维结果采样表：{OUT_PARQUET}")

    plt.figure(figsize=(10, 8), dpi=150)
    if len(df_pr) > 0:
        hb = plt.hexbin(df_pr["PC1"], df_pr["PC2"], gridsize=80, bins='log', mincnt=1, linewidths=0.0, alpha=0.9)
        cbar = plt.colorbar(hb)
        cbar.set_label("Pred density (log)")
    if len(df_t1) > 0:
        plt.scatter(df_t1["PC1"], df_t1["PC2"], s=6, alpha=0.7, label="Train_R1toR4")
    if len(df_t2) > 0:
        plt.scatter(df_t2["PC1"], df_t2["PC2"], s=6, alpha=0.7, label="Train_R1toR3_filtered")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA: Training vs Expanded Library (sampled)")
    plt.legend(loc="best"); plt.tight_layout(); plt.savefig(OUT_PNG)
    print(f"[Info] 图像已保存：{OUT_PNG}")


if __name__ == "__main__":
    main()
