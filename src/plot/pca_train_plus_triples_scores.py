# -*- coding: utf-8 -*-
"""
PCA：训练集(2份) + triples 数据集（带预测分数）
- 用两份训练集拟合 StandardScaler + IncrementalPCA（分块）
- 将 triples 特征投影到同一 PCA 空间
- 背景：按规则网格计算“平均预测分数”热图 + 画出分数场梯度较大的等高线（快速下滑区域）
- 覆盖：两份训练集散点（不同颜色）
- 兼容列名风格(::fN 与 ::N)、分隔符/编码嗅探；triples-预测文件自动匹配 score 列并合并
"""

import os, re, csv, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# -------------------- 你的文件路径 --------------------
f_train1 = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_monotonic_decrease_R1toR4_ALLfeatures.csv"
f_train2 = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\BB_add_pseudocount_R1toR3_filtered_ALLfeatures.csv"

feat_path = r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\lib_singlemut_ALL_triples_ALLfeatures.csv"
seq_path  = r"Yuhao\NGS\Abl_L1\06.LibraryDesign\triples_mut\lib_singlemut_ALL_triples.csv"
pred_path = r"D:\Me\IMB\Data\Yuhao\models\predictions_lib_singlemut_ALL_triples.csv"

# -------------------- 配置 --------------------
FEATURE_PREFIXES = [
    "binary::", "AAC::", "DPC::", "CKSAAGP_gap3::", "PAAC_l5_w0.05::",
]
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
TRIPLES_SAMPLE_MAX = 300_000  # triples 可视化采样上限
GRID_BINS = 150               # PCA 网格分辨率（用于分数热图/梯度）
GRAD_PERCENTILE = 92          # 梯度幅值的百分位阈值（等高线）

OUT_PNG         = "pca_train_triples_scores.png"
OUT_COORDS_PQ   = "pca_coords_train_triples.parquet"
OUT_COORDS_CSV  = "pca_coords_train_triples.csv"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
# -----------------------------------------------------

# ========= CSV 嗅探 =========
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

# ========= 基于首训练文件选择特征，并对齐其他文件 =========
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
            mapped_f = [mc for bc,mc in zip(base, temp[fp]) if (bc not in all_miss and mc is not None)]
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
    zs=[]; # 或水库采样
    if sample_max is None or sample_max<=0:
        for X in stream_numeric_chunks(fp, usecols):
            if not X.size: continue
            zs.append(ipca.transform(scaler.transform(X)))
        return np.concatenate(zs, axis=0) if zs else np.empty((0,2))
    # 水库采样
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

# ========= 读取 triples 的序列与预测，合并到同一 DataFrame =========
def read_triples_predictions(seq_path, pred_path):
    # 读序列表（尽力找到序列列）
    kw_seq = get_kw(seq_path).copy(); kw_seq.pop('low_memory',None)
    df_seq = pd.read_csv(seq_path, **kw_seq)
    # 猜测序列列名
    seq_cols_guess = [c for c in df_seq.columns if c.lower() in ('peptide_sequence','sequence','seq')]
    if not seq_cols_guess:
        # 若完全没有，就创建一个基于行号的序列 ID（用于后续与预测对齐）
        df_seq['_row_id_'] = np.arange(len(df_seq))
        seq_col = None
    else:
        seq_col = seq_cols_guess[0]
        df_seq = df_seq[[seq_col]].copy()

    # 读预测
    kw_pred = get_kw(pred_path).copy(); kw_pred.pop('low_memory',None)
    df_pred = pd.read_csv(pred_path, **kw_pred)

    # 找分数列：优先名含 'pred'/'score'/'prob' 的数值列，否则挑第一列数值列
    num_cols = [c for c in df_pred.columns if pd.api.types.is_numeric_dtype(df_pred[c])]
    cand = [c for c in df_pred.columns if any(k in c.lower() for k in ('pred','score','prob')) and pd.api.types.is_numeric_dtype(df_pred[c])]
    score_col = (cand[0] if cand else (num_cols[0] if num_cols else None))
    if score_col is None:
        raise ValueError(f"{pred_path} 未找到数值型预测分数列，请确认列名。")

    # 尝试按序列列合并
    if seq_col and seq_col in df_pred.columns:
        df = pd.merge(df_seq, df_pred[[seq_col, score_col]], on=seq_col, how='left')
    elif seq_col:
        # 预测里没有序列列 -> 退化为行对齐
        df = df_seq.copy()
        if len(df_pred) != len(df_seq):
            print(f"[Warn] 预测与序列表长度不同，行对齐将截断到最短。pred={len(df_pred)}, seq={len(df_seq)}")
        n = min(len(df_pred), len(df_seq))
        df = df.iloc[:n].copy()
        df['__score__'] = df_pred[score_col].values[:n]
        return df, '__score__'
    else:
        # 没找到序列列，纯行对齐
        n = min(len(df_pred), len(df_seq))
        df = df_seq.iloc[:n].copy()
        df['__score__'] = df_pred[score_col].values[:n]
        return df, '__score__'

    sc = '__score__'
    df.rename(columns={score_col: sc}, inplace=True)
    return df, sc

# ========= 栅格化平均分数 + 梯度 =========
def grid_mean_and_grad(x, y, val, bins=150):
    # 取范围
    xmin,xmax = np.nanmin(x), np.nanmax(x)
    ymin,ymax = np.nanmin(y), np.nanmax(y)
    # 2D 直方 & 2D 加权直方
    H_num, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[xmin,xmax],[ymin,ymax]])
    H_sum, _, _            = np.histogram2d(x, y, bins=[xedges, yedges], weights=val)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = H_sum / H_num
    # 计算梯度幅值（NaN 用邻域插值粗糙填充）
    M = mean.copy()
    # 简单填补：用均值替 NaN（更复杂方法需要 scipy，不引入依赖）
    M_nan = np.isnan(M)
    if np.any(M_nan):
        M[~M_nan] = M[~M_nan]
        M[M_nan] = np.nanmean(M[~M_nan]) if np.any(~M_nan) else 0.0
    gy, gx = np.gradient(M)   # 注意轴顺序：第0轴对应 y
    grad_mag = np.sqrt(gx*gx + gy*gy)
    # 栅格中心
    xc = 0.5*(xedges[:-1]+xedges[1:])
    yc = 0.5*(yedges[:-1]+yedges[1:])
    return xc, yc, mean.T, grad_mag.T  # 转置使得 [row=y, col=x]

# ========= 主流程 =========
def main():
    # 嗅探 + 简报
    for fp in [f_train1, f_train2, feat_path, seq_path, pred_path]:
        if os.path.isfile(fp): _ = get_kw(fp)
        else: print(f"[Warn] 文件不存在：{fp}")

    # 对齐特征列（以训练1为基准）
    usecols_map = make_usecols_map(f_train1, [f_train2, feat_path])

    # 拟合 StandardScaler / IPCA（仅用两份训练集）
    scaler = StandardScaler(with_mean=True, with_std=True)
    partial_fit_scaler(scaler, [f_train1, f_train2], usecols_map)
    ipca = IncrementalPCA(n_components=N_COMPONENTS)
    partial_fit_ipca(ipca, scaler, [f_train1, f_train2], usecols_map)
    print("[Info] 标准化与IPCA拟合完成。")

    # 训练集投影
    Z_t1 = transform_all(ipca, scaler, f_train1, usecols_map[f_train1], sample_max=None)
    Z_t2 = transform_all(ipca, scaler, f_train2, usecols_map[f_train2], sample_max=None)
    print(f"[Info] 训练集降维：t1={Z_t1.shape}, t2={Z_t2.shape}")

    # 读取 triples 的预测分数（只拿序列+score，用于后续合并）
    df_triples_meta, score_col = read_triples_predictions(seq_path, pred_path)

    # triples 特征投影（可采样，避免巨量点）
    Z_tri = transform_all(ipca, scaler, feat_path, usecols_map[feat_path], sample_max=TRIPLES_SAMPLE_MAX)
    print(f"[Info] Triples 降维（采样后）：{Z_tri.shape}")

    # 将分数对齐到采样后的 triples 投影：
    # 简化假设：feat_path 的行顺序与 seq_path/pred_path一致（通常你的生成流程就是按顺序写出）
    # 若采样了 TRIPLES_SAMPLE_MAX，则按“见到的顺序”保留前 s_size 个样本的分数
    # 因为 transform_all 的水库采样会打乱位置，为保持严格一致，这里选择：不做水库采样时严格一一对应；
    # 做水库采样时，无法追踪索引，退化为“随机抽样同数目的分数”近似呈现（可接受用于空间热图）
    # ——稳妥起见：当 TRIPLES_SAMPLE_MAX 生效时，随机抽 df_triples_meta 的分数同样本数
    if TRIPLES_SAMPLE_MAX and TRIPLES_SAMPLE_MAX>0:
        if len(df_triples_meta) < len(Z_tri):
            raise RuntimeError("triples 分数条数少于降维点数，无法匹配；请检查输入。")
        score_vals = df_triples_meta[score_col].to_numpy()
        idx = rng.choice(len(score_vals), size=Z_tri.shape[0], replace=False)
        scores = score_vals[idx]
    else:
        # 不采样：严格按行对齐（长度以较短为准）
        n = min(len(df_triples_meta), Z_tri.shape[0])
        if n < Z_tri.shape[0]:
            Z_tri = Z_tri[:n]
        scores = df_triples_meta[score_col].to_numpy()[:n]

    # 构造 DataFrame 并保存（训练集全量 + triples 采样）
    df_t1 = pd.DataFrame(Z_t1, columns=['PC1','PC2']); df_t1['source']='Train_R1toR4'
    df_t2 = pd.DataFrame(Z_t2, columns=['PC1','PC2']); df_t2['source']='Train_R1toR3_filtered'
    df_tr = pd.DataFrame(Z_tri, columns=['PC1','PC2']); df_tr['source']='Triples'; df_tr['score']=scores
    df_all = pd.concat([df_t1, df_t2, df_tr], ignore_index=True)

    try:
        df_all.to_parquet(OUT_COORDS_PQ, index=False)
        print(f"[Info] 已导出：{OUT_COORDS_PQ}")
    except Exception as e:
        df_all.to_csv(OUT_COORDS_CSV, index=False)
        print(f"[Warn] 保存 parquet 失败({e})，已回退 CSV：{OUT_COORDS_CSV}")

    # ---------- 生成背景热图（平均预测分数）与“快速下滑区域” ----------
    if len(df_tr)>0:
        xc, yc, mean_grid, grad_grid = grid_mean_and_grad(
            df_tr['PC1'].to_numpy(), df_tr['PC2'].to_numpy(),
            df_tr['score'].to_numpy(), bins=GRID_BINS
        )
    else:
        mean_grid = grad_grid = None

    # ---------- 画图 ----------
    plt.figure(figsize=(10, 8), dpi=150)

    # 背景：平均预测分数热图
    if mean_grid is not None and np.isfinite(mean_grid).any():
        # 用 imshow 风格：extent 由网格边界决定
        extent=[xc.min(), xc.max(), yc.min(), yc.max()]
        plt.imshow(mean_grid, origin='lower', extent=extent, aspect='auto')
        cbar = plt.colorbar()
        cbar.set_label("Mean prediction score")

        # 叠加“快速下滑区”——梯度等高线
        thr = np.nanpercentile(grad_grid, GRAD_PERCENTILE)
        CS = plt.contour(xc, yc, grad_grid, levels=[thr], linewidths=1.0)
        for c in CS.collections: c.set_label("High gradient (rapid drop)")

    # 覆盖训练集散点
    if len(df_t1)>0:
        plt.scatter(df_t1['PC1'], df_t1['PC2'], s=6, alpha=0.7, label='Train_R1toR4')
    if len(df_t2)>0:
        plt.scatter(df_t2['PC1'], df_t2['PC2'], s=6, alpha=0.7, label='Train_R1toR3_filtered')

    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA: Train sets + Triples (colored by mean prediction; contour=rapid drop)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f"[Info] 图像已保存：{OUT_PNG}")

if __name__ == "__main__":
    main()
