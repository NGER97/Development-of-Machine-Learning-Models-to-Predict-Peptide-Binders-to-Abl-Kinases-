# -*- coding: utf-8 -*-
"""
PCA（训练集专用）：对两份训练数据做 PCA 并在一张图中对比
- 以首个训练文件表头为基准，依据前缀选择特征列
- 另一文件按同名或“规范化名”对齐（::123 → ::f123，去前导零/全角/零宽空格/多冒号等）
- 分隔符/编码自动嗅探；分块 StandardScaler.partial_fit + IncrementalPCA.partial_fit
- 可选对训练集做无偏水库采样（默认关闭）
- 输出：pca_train_only.png + pca_train_only.parquet + pca_train_only.csv（同时导出）
- 现在会把每条样本的 peptide 序列（若找到列）写入导出表（列名统一为 peptide_sequence）
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

# 特征列前缀白名单（只选这些前缀开头的列）
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
    'class','Class','peptide_sequence',
    'seq','sequence','Sequence','SEQ','peptide','Peptide',
    'label','Label','target','Target','y','Y',
    'set','Set','source','Source',
    'id','ID','Id',
    'Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','index','Index'
}

CHUNK_SIZE = 20000            # 分块大小（内存紧张可再调小）
N_COMPONENTS = 2              # PCA 维度

# （可选）训练集采样上限；默认 None 表示全量绘图/导出
TRAIN1_SAMPLE_MAX = None      # 例如 100_000
TRAIN2_SAMPLE_MAX = None

# 输出文件
OUT_PNG        = "pca_train_only.png"
OUT_TABLE_PQ   = "pca_train_only.parquet"
OUT_TABLE_CSV  = "pca_train_only.csv"

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
_pat_num_nof   = re.compile(r'^(.*::)(\d+)$')    # ...::123

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

def header(fp):
    kw = get_csv_kw(fp).copy()
    kw.pop('low_memory', None)                     # python 引擎不支持 low_memory
    cols = pd.read_csv(fp, nrows=0, **kw).columns.tolist()
    return cols

def starts_with_any(s: str, prefixes) -> bool:
    return any(s.startswith(p) for p in prefixes)

# ========= 依据首文件选择特征 & 对齐第二个文件 =========
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

    mapped, missing = [], set()
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
        print(f"[Warn] {other_fp} 缺失 {len(missing)} 列（示例）: {list(missing)[:10]}")
    return mapped, missing

def make_fully_aligned_usecols(base_fp, other_fp, prefixes, exclude_raw):
    base_usecols = build_base_feature_list(base_fp, prefixes, exclude_raw)
    mapped, missing = align_usecols_to_other_file(base_usecols, other_fp)

    if missing:
        print(f"[Info] 有 {len(missing)} 个基准列在第二个文件缺失，将统一剔除。")
        base_filtered = [c for c in base_usecols if c not in missing]
        other_filtered = [mc for bc, mc in zip(base_usecols, mapped) if (bc not in missing and mc is not None)]
    else:
        base_filtered = base_usecols
        other_filtered = mapped

    if len(base_filtered) == 0 or len(base_filtered) != len(other_filtered):
        raise RuntimeError(f"对齐后的列数异常：base={len(base_filtered)}, other={len(other_filtered)}")
    print(f"[Info] 对齐完成；每个训练集特征列数：{len(base_filtered)}")
    return {base_fp: base_filtered, other_fp: other_filtered}

# ========= 序列列名自动识别 =========
SEQ_CANDIDATES_LOWER = [
    "peptide_sequence", "sequence", "seq", "peptide"
]

def get_seq_col(fp):
    """在文件表头中查找序列列名（大小写不敏感），找不到则返回 None"""
    cols = header(fp)
    lower_map = {c.lower(): c for c in cols}
    for key in SEQ_CANDIDATES_LOWER:
        if key in lower_map:
            return lower_map[key]
    print(f"[Warn] {Path(fp).name} 未找到序列列（尝试了 {SEQ_CANDIDATES_LOWER}），将不输出序列。")
    return None

# ========= 数据流 & 变换 =========
def stream_numeric_chunks(fp, usecols, seq_col=None, chunksize=CHUNK_SIZE):
    """读取特征矩阵（必选）+ 序列列（可选），逐块返回
    Yields:
        (X, seq_list) 若 seq_col 为 None，则 seq_list 为空列表
    """
    kw = get_csv_kw(fp).copy()
    kw.pop('low_memory', None)
    read_cols = list(usecols)
    if seq_col is not None and seq_col not in read_cols:
        read_cols.append(seq_col)

    for chunk in pd.read_csv(fp, usecols=read_cols, chunksize=chunksize, **kw):
        # 特征部分转数值
        feat_chunk = chunk[usecols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X = feat_chunk.to_numpy(dtype=np.float64, copy=False)

        # 序列部分（若有）
        if seq_col is not None and seq_col in chunk.columns:
            seq_list = chunk[seq_col].astype(str).tolist()
        else:
            seq_list = []
        yield X, seq_list

def partial_fit_scaler_on_files(scaler, files, usecols_map):
    for fp in files:
        for X, _ in stream_numeric_chunks(fp, usecols_map[fp]):
            if X.size == 0:
                continue
            scaler.partial_fit(X)

def partial_fit_ipca_on_files(ipca, scaler, files, usecols_map):
    for fp in files:
        for X, _ in stream_numeric_chunks(fp, usecols_map[fp]):
            if X.size == 0:
                continue
            Xs = scaler.transform(X)
            ipca.partial_fit(Xs)

def reservoir_sample_transform(ipca, scaler, fp, usecols, k_max, seq_col=None):
    """返回 (Z, seqs)，其中 seqs 为与 Z 对应的序列列表
       若 k_max 为 None/<=0 则全量返回；否则进行水库采样
    """
    # ——全量路径——
    if not k_max or k_max <= 0:
        Z_out = []
        S_out = []  # sequences
        for X, seqs in stream_numeric_chunks(fp, usecols, seq_col=seq_col):
            if X.size == 0:
                continue
            Z = ipca.transform(scaler.transform(X))
            Z_out.append(Z)
            if seq_col is not None:
                S_out.extend(seqs)
        if Z_out:
            Z_cat = np.concatenate(Z_out, axis=0)
            return Z_cat, (S_out if seq_col is not None else [])
        else:
            return np.empty((0, 2)), ([] if seq_col is not None else [])

    # ——水库采样路径——
    S = None                      # 坐标水库
    S_seq = None                  # 序列水库
    s_size = 0
    n_seen = 0

    for X, seqs in stream_numeric_chunks(fp, usecols, seq_col=seq_col):
        if X.size == 0:
            continue
        Z = ipca.transform(scaler.transform(X))
        m = Z.shape[0]
        for i in range(m):
            n_seen += 1
            zi = Z[i]
            si = (seqs[i] if (seq_col is not None and i < len(seqs)) else None)

            if s_size < k_max:
                if S is None:
                    S = np.empty((k_max, 2), dtype=np.float64)
                    S_seq = [None] * k_max
                S[s_size] = zi
                if seq_col is not None:
                    S_seq[s_size] = si
                s_size += 1
            else:
                j = rng.integers(0, n_seen)
                if j < k_max:
                    S[j] = zi
                    if seq_col is not None:
                        S_seq[j] = si

    if S is None:
        return np.empty((0, 2)), ([] if seq_col is not None else [])
    return S[:s_size].copy(), (S_seq[:s_size] if seq_col is not None else [])

# ========= 主流程 =========
def main():
    files = [f_train1, f_train2]

    # 打印嗅探信息 + 列数
    for fp in files:
        _ = get_csv_kw(fp)
        cols = header(fp)
        print(f"[Info] {Path(fp).name} 列计数: {len(cols)}（前5列: {cols[:5]}）")

    # 1) 以首文件为基准，对齐两份训练集的特征列（顺序完全一致）
    usecols_map = make_fully_aligned_usecols(
        base_fp=f_train1,
        other_fp=f_train2,
        prefixes=FEATURE_PREFIXES,
        exclude_raw=EXCLUDE_COLS_RAW
    )

    # 新增：识别序列列名（两文件可能不同）
    seq_col_map = {fp: get_seq_col(fp) for fp in files}
    print("[Info] 序列列映射：", {Path(k).name: v for k, v in seq_col_map.items()})

    # 2) 在两份训练集上拟合标准化
    scaler = StandardScaler(with_mean=True, with_std=True)
    partial_fit_scaler_on_files(scaler, files, usecols_map)
    print("[Info] StandardScaler 拟合完成。")

    # 3) 在训练集上增量拟合 PCA
    ipca = IncrementalPCA(n_components=N_COMPONENTS)
    partial_fit_ipca_on_files(ipca, scaler, files, usecols_map)
    print("[Info] IncrementalPCA 拟合完成。")

    # 4) 变换（按需采样），同时拿到序列
    Z_train1, seqs1 = reservoir_sample_transform(
        ipca, scaler, f_train1, usecols_map[f_train1],
        k_max=TRAIN1_SAMPLE_MAX, seq_col=seq_col_map[f_train1]
    )
    Z_train2, seqs2 = reservoir_sample_transform(
        ipca, scaler, f_train2, usecols_map[f_train2],
        k_max=TRAIN2_SAMPLE_MAX, seq_col=seq_col_map[f_train2]
    )
    print(f"[Info] 降维完成：train1={Z_train1.shape}, train2={Z_train2.shape}")

    # 5) 导出 + 绘图（同时导出 Parquet 与 CSV）
    df_t1 = pd.DataFrame(Z_train1, columns=["PC1", "PC2"])
    df_t1["source"] = "Train_R1toR4"
    if seq_col_map[f_train1] is not None and len(seqs1) == len(df_t1):
        df_t1["peptide_sequence"] = seqs1

    df_t2 = pd.DataFrame(Z_train2, columns=["PC1", "PC2"])
    df_t2["source"] = "Train_R1toR3_filtered"
    if seq_col_map[f_train2] is not None and len(seqs2) == len(df_t2):
        df_t2["peptide_sequence"] = seqs2

    df_all = pd.concat([df_t1, df_t2], axis=0, ignore_index=True)

    # ——Parquet——
    try:
        # 如需压缩可加：compression="snappy"
        df_all.to_parquet(OUT_TABLE_PQ, index=False)
        print(f"[Info] 已导出降维结果（Parquet）：{OUT_TABLE_PQ}")
    except Exception as e:
        print(f"[Warn] 保存 Parquet 失败：{e}")

    # ——CSV（总是导出）——
    try:
        # 大文件可改为 .csv.gz 并加 compression="gzip"；可加 float_format 控制精度
        df_all.to_csv(OUT_TABLE_CSV, index=False)
        print(f"[Info] 已导出降维结果（CSV）：{OUT_TABLE_CSV}")
    except Exception as e:
        print(f"[Error] 保存 CSV 失败：{e}")

    # 绘图：两份训练集散点，默认不同颜色
    plt.figure(figsize=(10, 8), dpi=150)
    if len(df_t1) > 0:
        plt.scatter(df_t1["PC1"], df_t1["PC2"], s=6, alpha=0.7, label="Train_R1toR4")
    if len(df_t2) > 0:
        plt.scatter(df_t2["PC1"], df_t2["PC2"], s=6, alpha=0.7, label="Train_R1toR3_filtered")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA: Training sets only")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f("[Info] 图像已保存：{OUT_PNG}"))

if __name__ == "__main__":
    main()
