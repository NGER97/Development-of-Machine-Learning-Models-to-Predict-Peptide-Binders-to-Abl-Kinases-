# -*- coding: utf-8 -*-
"""
PCA 对比：两份训练集 vs 大规模预测集（分块、增量PCA、密度+散点叠加）
特性：
- 以首个训练文件的表头为“基准顺序”，仅挑选指定前缀的特征列
- 其它文件按同名或“规范化名”对齐（把 ::123 规范为 ::f123，去前导零、全角、零宽空格等）
- 自动嗅探 CSV 分隔符/编码；分块 StandardScaler.partial_fit + IncrementalPCA.partial_fit
- 预测集严格水库采样（无偏）控制绘图点数
- 导出 parquet（若无 pyarrow/fastparquet 则回退 CSV）
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
    # 若有其它族（如 "ProtBERT::"），直接在此追加
]

# 明确排除（即使前缀匹配也不纳入）
EXCLUDE_COLS_RAW = {
    'class','Class','peptide_sequence',  # 预测集里有序列列，排除
    'seq','sequence','Sequence','SEQ',
    'label','Label','target','Target','y','Y',
    'set','Set','source','Source',
    'id','ID','Id',
    'Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','index','Index'
}

CHUNK_SIZE = 20000           # 分块大小（内存紧张可调小）
PRED_SAMPLE_MAX = 200_000    # 预测集水库采样上限
N_COMPONENTS = 2             # PCA 维度

OUT_PNG     = "pca_all.png"
OUT_TABLE_PQ = "pca_coords_sampled.parquet"
OUT_TABLE_CSV = "pca_coords_sampled.csv"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
# ----------------------------------------------------

# ========= CSV 读参嗅探 =========
def detect_csv_format(fp, sniff_bytes=65536):
    with open(fp, 'rb') as f:
        raw = f.read(sniff_bytes)
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
    kwargs = dict(sep=sep, engine='python', encoding=enc, on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL)
    print(f"[Info] 读取设置 {Path(fp).name}: sep='{sep}', encoding={enc}, header_tokens={counts.get(sep,0)}")
    return kwargs

CSV_READ_KW = {}
def get_csv_kw(fp):
    if fp not in CSV_READ_KW:
        CSV_READ_KW[fp] = detect_csv_format(fp)
    return CSV_READ_KW[fp]

# ========= 列名规范化（把 ::123 → ::f123，去前导零/全角/零宽空格/多冒号等） =========
_pat_fnum      = re.compile(r'^(.*::f)(\d+)$')   # ...::f123
_pat_num_nof   = re.compile(r'^(.*::)(\d+)$')    # ...::123   ← 预测集可能是这种

def norm_name(col: str) -> str:
    s = unicodedata.normalize('NFKC', str(col)).strip()
    s = s.replace('\u200b', '')               # 去零宽
    s = s.replace('：', ':')                   # 全角冒号 -> 半角
    s = re.sub(r':{2,}', '::', s)             # 多冒号压缩
    s = re.sub(r'(?<=\w):(f?\d+)$', r'::\1', s)  # 单冒号 -> 双冒号（末尾 fN 或 N）
    # 先把 ...::123 统一成 ...::f123（去前导零）
    m2 = _pat_num_nof.match(s)
    if m2:
        s = m2.group(1) + 'f' + str(int(m2.group(2)))
    # 再把 ...::f001 → ...::f1
    m1 = _pat_fnum.match(s)
    if m1:
        s = m1.group(1) + str(int(m1.group(2)))
    return s

# —— 修正 header()：去掉 low_memory=False ——
def header(fp):
    kw = get_csv_kw(fp).copy()
    # python 引擎不支持 low_memory，稳妥起见先删掉这个键
    kw.pop('low_memory', None)
    cols = pd.read_csv(fp, nrows=0, **kw).columns.tolist()
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
            base_feat.append(c)   # 保留首文件“原名+顺序”
    if not base_feat:
        raise ValueError("在首文件中未匹配到任何特征列，请检查 FEATURE_PREFIXES/EXCLUDE_COLS_RAW。")
    print(f"[Info] 基准特征列数: {len(base_feat)}；前10列: {base_feat[:10]}")
    return base_feat

def align_usecols_to_other_file(base_usecols, other_fp):
    other_cols = header(other_fp)
    other_set  = set(other_cols)
    other_norm_map = {}
    for oc in other_cols:
        other_norm_map.setdefault(norm_name(oc), oc)

    mapped = []
    missing = set()
    for bc in base_usecols:
        if bc in other_set:
            mapped.append(bc)                    # 完全同名
        else:
            bn = norm_name(bc)                   # 规范名匹配
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
        raise RuntimeError(
            f"对齐后的列数不一致或为0：{lengths}\n"
            f"请确认三个文件确实为同一套特征（或调整 FEATURE_PREFIXES）。"
        )
    print(f"[Info] 对齐完成；每个文件特征列数：{lengths[0]}")
    return usecols_map

# ========= 数据流 & 变换 =========
# —— 修正 stream_numeric_chunks()：去掉 low_memory=False ——
def stream_numeric_chunks(fp, usecols, chunksize=CHUNK_SIZE):
    kw = get_csv_kw(fp).copy()
    kw.pop('low_memory', None)
    for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, **kw):
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
                j = rng.integers(0, n_seen)  # 均匀抽取 [0, n_seen-1]
                if j < k_max:
                    S[j] = zi
    if S is None:
        return np.empty((0, 2))
    return S[:s_size].copy()

# ========= 主流程 =========
def main():
    files = [f_train1, f_train2, f_pred]

    # 打印嗅探信息 + 列数
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

    # 5) 导出 + 绘图（parquet 优先，失败回退 CSV）
    df_t1 = pd.DataFrame(Z_train1, columns=["PC1", "PC2"]); df_t1["source"] = "Train_R1toR4"
    df_t2 = pd.DataFrame(Z_train2, columns=["PC1", "PC2"]); df_t2["source"] = "Train_R1toR3_filtered"
    df_pr = pd.DataFrame(Z_pred,   columns=["PC1", "PC2"]); df_pr["source"] = "Lib_Expand_2 (sampled)"
    df_all_small = pd.concat([df_t1, df_t2, df_pr], axis=0, ignore_index=True)

    try:
        df_all_small.to_parquet(OUT_TABLE_PQ, index=False)
        print(f"[Info] 已导出降维结果：{OUT_TABLE_PQ}")
    except Exception as e:
        df_all_small.to_csv(OUT_TABLE_CSV, index=False)
        print(f"[Warn] 保存 parquet 失败({e})，已回退保存 CSV：{OUT_TABLE_CSV}")

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
