# -*- coding: utf-8 -*-
"""
PCA（训练集 + triples with scores）— Hexbin 热图版
- 训练集：拟合 StandardScaler + IncrementalPCA（分块增量）
- triples：与预测分数按行对齐，流式变换+水库采样，保持(Z, score)配对
- 背景：hexbin 显示“平均预测分数”
- 覆盖：两份训练集散点（不同颜色，matplotlib 默认）
"""

import os, re, csv, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# -------------------- 路径（按你的提供） --------------------
f_train1 = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_monotonic_decrease_R1toR4_ALLfeatures.csv"
f_train2 = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_R1toR3_filtered_ALLfeatures.csv"

feat_path = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\lib_singlemut_ALL_triples_ALLfeatures.csv"
seq_path  = r"Yuhao\NGS\Abl_L1\06.LibraryDesign\triples_mut\lib_singlemut_ALL_triples.csv"
pred_path = r"D:\Me\IMB\Data\Yuhao\models\predictions_lib_singlemut_ALL_triples.csv"

# -------------------- 配置 --------------------
FEATURE_PREFIXES = ["binary::", "AAC::", "DPC::", "CKSAAGP_gap3::", "PAAC_l5_w0.05::"]
EXCLUDE_COLS = {
    'class','Class','peptide_sequence',
    'seq','sequence','Sequence','SEQ',
    'label','Label','target','Target','y','Y',
    'set','Set','source','Source',
    'id','ID','Id',
    'Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','index','Index'
}
CHUNK_SIZE = 20000
N_COMPONENTS = 2

# triples 采样上限（越大越细致，内存越高；保持(Z,score)配对）
TRIPLES_SAMPLE_MAX = 300_000

# hexbin 参数
HEX_GRIDSIZE = 120
HEX_MINCNT   = 10

# 输出
OUT_PNG        = "pca_train_triples_hexbin.png"
OUT_TABLE_PQ   = "pca_coords_train_triples_hexbin.parquet"
OUT_TABLE_CSV  = "pca_coords_train_triples_hexbin.csv"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ========= CSV 读参嗅探 =========
def detect_csv_format(fp, sniff_bytes=65536):
    with open(fp, 'rb') as f:
        raw = f.read(sniff_bytes)
    try:
        text = raw.decode('utf-8-sig'); enc = 'utf-8-sig'
    except UnicodeDecodeError:
        text = raw.decode('latin-1');  enc = 'latin-1'
    first = text.splitlines()[0] if text else ""
    cand = [',','\t',';','|']
    counts = {c:first.count(c) for c in cand}
    sep = max(counts, key=counts.get) if any(counts.values()) else ','
    kw = dict(sep=sep, engine='python', encoding=enc, on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL)
    print(f"[Info] 读取设置 {Path(fp).name}: sep='{sep}', enc={enc}, header_tokens={counts.get(sep,0)}")
    return kw

CSV_KW = {}
def get_kw(fp):
    if fp not in CSV_KW: CSV_KW[fp]=detect_csv_format(fp)
    return CSV_KW[fp]

# ========= 列名规范化（::N → ::fN；去前导零/全角/零宽空格/多冒号）=========
_pat_fnum    = re.compile(r'^(.*::f)(\d+)$')
_pat_no_fnum = re.compile(r'^(.*::)(\d+)$')
def norm_name(col: str) -> str:
    s = unicodedata.normalize('NFKC', str(col)).strip()
    s = s.replace('\u200b','').replace('：',':')
    s = re.sub(r':{2,}','::',s)
    s = re.sub(r'(?<=\w):(f?\d+)$', r'::\1', s)
    m2 = _pat_no_fnum.match(s)
    if m2: s = m2.group(1)+'f'+str(int(m2.group(2)))
    m1 = _pat_fnum.match(s)
    if m1: s = m1.group(1)+str(int(m1.group(2)))
    return s

def header(fp):
    kw = get_kw(fp).copy(); kw.pop('low_memory',None)
    return pd.read_csv(fp, nrows=0, **kw).columns.tolist()

def starts_with_any(s, prefixes):
    return any(s.startswith(p) for p in prefixes)

# ========= 依据训练1选择特征 & 对齐另外两个文件 =========
def build_base_feature_list(base_fp):
    cols = header(base_fp)
    exn = {norm_name(x) for x in EXCLUDE_COLS}
    base = [c for c in cols if starts_with_any(norm_name(c), FEATURE_PREFIXES) and norm_name(c) not in exn]
    if not base: raise ValueError("首训练文件未匹配到特征列，请检查前缀/排除列。")
    print(f"[Info] 基准特征列数: {len(base)}；前10列: {base[:10]}")
    return base

def align_usecols_to_other_file(base_usecols, other_fp):
    ocols = header(other_fp); oset=set(ocols)
    onorm = {}
    for oc in ocols:
        onorm.setdefault(norm_name(oc), oc)
    mapped=[]; missing=set()
    for bc in base_usecols:
        if bc in oset: mapped.append(bc)
        else:
            oc = onorm.get(norm_name(bc))
            if oc is None: mapped.append(None); missing.add(bc)
            else: mapped.append(oc)
    if missing: print(f"[Warn] {other_fp} 缺失 {len(missing)} 列（示例）: {list(missing)[:10]}")
    return mapped, missing

def make_usecols_map(base_fp, other_fps):
    base = build_base_feature_list(base_fp)
    usecols = {base_fp: base[:]}

    all_miss=set(); temp={}
    for fp in other_fps:
        m, miss = align_usecols_to_other_file(base, fp)
        temp[fp]=m; all_miss|=miss
    if all_miss:
        print(f"[Info] 有 {len(all_miss)} 个基准列在至少一个文件缺失，统一剔除。")
        base_f = [c for c in base if c not in all_miss]; usecols[base_fp]=base_f
        for fp in other_fps:
            mapped_f = [mc for bc, mc in zip(base, temp[fp]) if (bc not in all_miss and mc is not None)]
            usecols[fp]=mapped_f
    else:
        for fp in other_fps: usecols[fp]=temp[fp]

    lens = [len(v) for v in usecols.values()]
    if len(set(lens))!=1 or lens[0]==0:
        raise RuntimeError(f"对齐后的特征列数异常：{lens}")
    print(f"[Info] 对齐完成；每个文件特征列数：{lens[0]}")
    return usecols

# ========= 流式读 & 变换 =========
def stream_numeric_chunks(fp, usecols):
    kw = get_kw(fp).copy(); kw.pop('low_memory',None)
    for chunk in pd.read_csv(fp, usecols=usecols, chunksize=CHUNK_SIZE, **kw):
        for c in chunk.columns:
            chunk[c]=pd.to_numeric(chunk[c], errors='coerce')
        X = chunk.fillna(0.0).to_numpy(np.float64, copy=False)
        yield X

def partial_fit_scaler(scaler, files, usecols_map):
    for fp in files:
        for X in stream_numeric_chunks(fp, usecols_map[fp]):
            if X.size: scaler.partial_fit(X)

def partial_fit_ipca(ipca, scaler, files, usecols_map):
    for fp in files:
        for X in stream_numeric_chunks(fp, usecols_map[fp]):
            if not X.size: continue
            ipca.partial_fit(scaler.transform(X))

def transform_all(ipca, scaler, fp, usecols, sample_max=None):
    zs=[]
    if not sample_max or sample_max<=0:
        for X in stream_numeric_chunks(fp, usecols):
            if not X.size: continue
            zs.append(ipca.transform(scaler.transform(X)))
        return np.concatenate(zs, axis=0) if zs else np.empty((0,2))
    # 否则采样（仅坐标，无配对）
    S=None; s_size=0; n_seen=0
    for X in stream_numeric_chunks(fp, usecols):
        if not X.size: continue
        Z = ipca.transform(scaler.transform(X))
        for i in range(Z.shape[0]):
            n_seen+=1
            if s_size<sample_max:
                if S is None: S=np.empty((sample_max,2), dtype=np.float64)
                S[s_size]=Z[i]; s_size+=1
            else:
                j = rng.integers(0, n_seen)
                if j<sample_max: S[j]=Z[i]
    return (S[:s_size].copy() if S is not None else np.empty((0,2)))

# ========= 找预测分数列 =========
def find_score_col(pred_path):
    kw = get_kw(pred_path).copy(); kw.pop('low_memory',None)
    hdr = pd.read_csv(pred_path, nrows=0, **kw).columns.tolist()
    # 优先包含这些关键词的数值列
    sample = pd.read_csv(pred_path, nrows=1000, **kw)
    cand = [c for c in sample.columns if any(k in c.lower() for k in ('pred','score','prob')) and pd.api.types.is_numeric_dtype(sample[c])]
    if cand: return cand[0]
    num = [c for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c])]
    if num: return num[0]
    raise ValueError("预测文件里找不到数值型分数列。")

# ========= triples 变换 + 与分数配对的水库采样 =========
def transform_triples_with_scores(ipca, scaler, feat_fp, usecols, pred_fp, score_col, sample_max):
    """同步流式读取 features 和 scores，保持配对后做严格水库采样"""
    if not sample_max or sample_max<=0:
        # 全量返回（可能很大）
        Z_all = transform_all(ipca, scaler, feat_fp, usecols, sample_max=None)
        kwp = get_kw(pred_fp).copy(); kwp.pop('low_memory',None)
        score_all = pd.read_csv(pred_fp, usecols=[score_col], **kwp)[score_col].to_numpy()
        n = min(len(score_all), Z_all.shape[0])
        return Z_all[:n], score_all[:n]

    # 配对水库采样
    kwf = get_kw(feat_fp).copy(); kwf.pop('low_memory',None)
    kwp = get_kw(pred_fp).copy(); kwp.pop('low_memory',None)

    feat_reader = pd.read_csv(feat_fp, usecols=usecols, chunksize=CHUNK_SIZE, **kwf)
    pred_reader = pd.read_csv(pred_fp, usecols=[score_col], chunksize=CHUNK_SIZE, **kwp)

    # 预测分数的缓冲
    pred_buf = np.empty((0,), dtype=np.float64)

    def need_scores(n):
        nonlocal pred_buf
        while pred_buf.shape[0] < n:
            try:
                ch = next(pred_reader)
            except StopIteration:
                break
            s = pd.to_numeric(ch[score_col], errors='coerce').fillna(0.0).to_numpy(np.float64, copy=False)
            pred_buf = np.concatenate([pred_buf, s], axis=0)
        if pred_buf.shape[0] < n:
            raise RuntimeError("预测分数行数不足，无法与特征按行对齐。")

        out = pred_buf[:n].copy()
        pred_buf = pred_buf[n:]
        return out

    S = None                       # (k,2)
    Sco = None                     # (k,)
    s_size = 0
    n_seen = 0

    for ch in feat_reader:
        # 特征 -> 标准化 -> PCA
        for c in ch.columns:
            ch[c] = pd.to_numeric(ch[c], errors='coerce')
        X = ch.fillna(0.0).to_numpy(np.float64, copy=False)
        Z = ipca.transform(scaler.transform(X))      # (m,2)
        scores = need_scores(Z.shape[0])             # (m,)

        for i in range(Z.shape[0]):
            n_seen += 1
            zi = Z[i]; si = scores[i]
            if s_size < sample_max:
                if S is None:
                    S = np.empty((sample_max,2), dtype=np.float64)
                    Sco = np.empty((sample_max,), dtype=np.float64)
                S[s_size] = zi
                Sco[s_size] = si
                s_size += 1
            else:
                j = rng.integers(0, n_seen)
                if j < sample_max:
                    S[j] = zi
                    Sco[j] = si

    if S is None:
        return np.empty((0,2)), np.empty((0,))
    return S[:s_size].copy(), Sco[:s_size].copy()

# ========= 主流程 =========
def main():
    # 嗅探+简报
    for fp in [f_train1, f_train2, feat_path, pred_path]:
        if os.path.isfile(fp): _ = get_kw(fp)
        else: print(f"[Warn] 文件不存在：{fp}")

    # 对齐特征列
    usecols_map = make_usecols_map(f_train1, [f_train2, feat_path])

    # 拟合 scaler/ipca（仅训练集）
    scaler = StandardScaler(with_mean=True, with_std=True)
    partial_fit_scaler(scaler, [f_train1, f_train2], usecols_map)
    ipca = IncrementalPCA(n_components=N_COMPONENTS)
    partial_fit_ipca(ipca, scaler, [f_train1, f_train2], usecols_map)
    print("[Info] 标准化与IPCA拟合完成。")

    # 训练集投影（全量）
    Z_t1 = transform_all(ipca, scaler, f_train1, usecols_map[f_train1], sample_max=None)
    Z_t2 = transform_all(ipca, scaler, f_train2, usecols_map[f_train2], sample_max=None)
    print(f"[Info] 训练集降维：t1={Z_t1.shape}, t2={Z_t2.shape}")

    # triples 分数列
    score_col = find_score_col(pred_path)
    print(f"[Info] 预测分数列：{score_col}")

    # triples 投影 + 分数（配对采样）
    Z_tri, S_tri = transform_triples_with_scores(
        ipca, scaler, feat_path, usecols_map[feat_path], pred_path, score_col, TRIPLES_SAMPLE_MAX
    )
    print(f"[Info] Triples 降维（样本={Z_tri.shape[0]}），已与分数配对。")

    # 组织导出表
    df_t1 = pd.DataFrame(Z_t1, columns=['PC1','PC2']); df_t1['source']='Train_R1toR4'
    df_t2 = pd.DataFrame(Z_t2, columns=['PC1','PC2']); df_t2['source']='Train_R1toR3_filtered'
    df_tr = pd.DataFrame(Z_tri, columns=['PC1','PC2']); df_tr['source']='Triples'; df_tr['score']=S_tri
    df_all = pd.concat([df_t1, df_t2, df_tr], ignore_index=True)

    try:
        df_all.to_parquet(OUT_TABLE_PQ, index=False)
        print(f"[Info] 已导出：{OUT_TABLE_PQ}")
    except Exception as e:
        df_all.to_csv(OUT_TABLE_CSV, index=False)
        print(f"[Warn] 保存 parquet 失败({e})，已回退 CSV：{OUT_TABLE_CSV}")

    # ---------- 绘图：hexbin 背景 + 训练集散点 ----------
    plt.figure(figsize=(10, 8), dpi=150)

    if len(df_tr)>0:
        hb = plt.hexbin(
            df_tr['PC1'], df_tr['PC2'],
            C=df_tr['score'],
            reduce_C_function=np.mean,
            gridsize=HEX_GRIDSIZE, mincnt=HEX_MINCNT
        )
        cbar = plt.colorbar(hb)
        cbar.set_label("Mean prediction score")

    if len(df_t1)>0:
        plt.scatter(df_t1['PC1'], df_t1['PC2'], s=6, alpha=0.7, label='Train_R1toR4')
    if len(df_t2)>0:
        plt.scatter(df_t2['PC1'], df_t2['PC2'], s=6, alpha=0.7, label='Train_R1toR3_filtered')

    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA: Train sets + Triples (hexbin = mean prediction score)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f"[Info] 图像已保存：{OUT_PNG}")

if __name__ == "__main__":
    main()
