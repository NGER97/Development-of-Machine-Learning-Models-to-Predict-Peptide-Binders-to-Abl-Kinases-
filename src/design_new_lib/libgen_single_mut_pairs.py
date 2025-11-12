# libgen_single_mut_pairs.py
# 批量：对多个“两位点组合”生成单点突变文库（可选也生双突变）
# 直接在“全局变量”里改，不用命令行参数

from pathlib import Path
import pandas as pd
import itertools as it

# ========= 你只需要改这里 =========
# 1) 亲本序列（可多条）
PARENTS = [
    ("seq1", "YLHWDYVW"),
    # ("seq2", "ABCDEFGH"),
]

# 2) 氨基酸字母表
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# 3) 两位点组合列表（1-based）。例如只想试 (2,5) 和 (3,6)：
POSITION_SETS = None

# 3.1) 或者一键尝试所有 8 选 2 的组合（共 28 组）
AUTO_ALL_8C2 = True

# 4) 是否把亲本也写入（WT 行，position=0, mut_tag="WT"）
INCLUDE_PARENTS = True

# 5) 去重策略
DEDUP_CHILDREN_WITHIN_PAIR = False   # 单个组合内，按 child 去重
GLOBAL_DEDUP_CHILDREN = False        # 所有组合合并后，跨组合全局去重

# 6) 是否额外生成“双突变库”（同时改两个位点的aa组合）
GENERATE_DOUBLE_MUTS = False

# 7) 输出
OUTPUT_DIR = r"Yuhao/NGS/Abl_L1/06.LibraryDesign/pairs_singlemut"
FILENAME_PREFIX = "lib_singlemut"    # 文件名前缀
WRITE_COMBINED = True                # 是否输出合并后的总表
# ==================================


def _check_inputs():
    assert isinstance(PARENTS, (list, tuple)) and len(PARENTS) > 0, "PARENTS 不能为空"
    for pid, seq in PARENTS:
        if not pid:
            raise ValueError("parent_id 不能为空")
        s = str(seq).strip().upper()
        if len(s) != 8:
            raise ValueError(f"{pid} 的序列长度不是 8：{s} (len={len(s)})")
        bad = set(s) - set(ALPHABET)
        if bad:
            raise ValueError(f"{pid} 的序列含非字母表字符 {bad}；当前 ALPHABET={ALPHABET}")

def _norm_pairs(pairs):
    seen = set()
    out = []
    for a, b in pairs:
        if not (1 <= a <= 8 and 1 <= b <= 8) or a == b:
            raise ValueError(f"非法位点组合: {(a,b)}（需 1..8 且 a!=b）")
        t = tuple(sorted((int(a), int(b))))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _single_mut_rows(parent_id, parent_seq, pos_pair, alphabet):
    """只做单点突变（限制在 pos_pair 两个位点上）"""
    seq = parent_seq.strip().upper()
    rows = []

    if INCLUDE_PARENTS:
        rows.append({
            "parent_id": parent_id, "parent_seq": seq,
            "position": 0, "wt": "", "mut": "", "mut_tag": "WT",
            "peptide_sequence": seq, "pos_set_tag": f"{pos_pair[0]}-{pos_pair[1]}"
        })

    for pos1 in pos_pair:  # 两个位点各自做单点突变
        i = pos1 - 1
        wt = seq[i]
        for aa in alphabet:
            if aa == wt:
                continue
            child = seq[:i] + aa + seq[i+1:]
            rows.append({
                "parent_id": parent_id, "parent_seq": seq,
                "position": pos1, "wt": wt, "mut": aa,
                "mut_tag": f"{wt}{pos1}{aa}",
                "peptide_sequence": child,
                "pos_set_tag": f"{pos_pair[0]}-{pos_pair[1]}",
            })
    return rows

def _double_mut_rows(parent_id, parent_seq, pos_pair, alphabet):
    """可选：做双突变（两个位点同时变），不含 WT-WT；如需含WT行请用 INCLUDE_PARENTS=True"""
    seq = parent_seq.strip().upper()
    (p1, p2) = pos_pair
    i, j = p1 - 1, p2 - 1
    wt1, wt2 = seq[i], seq[j]
    rows = []
    for aa1 in alphabet:
        for aa2 in alphabet:
            if aa1 == wt1 and aa2 == wt2:
                continue  # 保留 WT 行由 INCLUDE_PARENTS 控制
            child = seq[:i] + aa1 + seq[i+1:j] + aa2 + seq[j+1:]
            rows.append({
                "parent_id": parent_id, "parent_seq": seq,
                "position": -1,                 # -1 表示“双突变”
                "wt": f"{wt1}{wt2}",
                "mut": f"{aa1}{aa2}",
                "mut_tag": f"{wt1}{p1}{aa1}_{wt2}{p2}{aa2}",
                "peptide_sequence": child,
                "pos_set_tag": f"{p1}-{p2}",
            })
    return rows

def _build_pairs():
    if AUTO_ALL_8C2:
        return _norm_pairs(list(it.combinations(range(1, 9), 2)))
    else:
        if not POSITION_SETS:
            raise ValueError("请设置 POSITION_SETS 或打开 AUTO_ALL_8C2")
        return _norm_pairs(POSITION_SETS)

def _write_df(df, path, info=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["parent_id", "parent_seq", "pos_set_tag", "position", "wt", "mut", "mut_tag", "peptide_sequence"]
    df = df[cols]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 写入 {path} {info}（{len(df)} 行）")

def main():
    _check_inputs()
    pairs = _build_pairs()
    out_dir = Path(OUTPUT_DIR)
    all_chunks = []

    print(f"[Info] 共有 {len(pairs)} 组两位点组合：{pairs[:6]}{' ...' if len(pairs)>6 else ''}")
    print(f"[Info] INCLUDE_PARENTS={INCLUDE_PARENTS}, 生成双突变={GENERATE_DOUBLE_MUTS}")

    for (a, b) in pairs:
        pair_tag = f"p{a}_{b}"
        rows = []
        for pid, s in PARENTS:
            rows.extend(_single_mut_rows(pid, s, (a, b), list(ALPHABET)))
            if GENERATE_DOUBLE_MUTS:
                rows.extend(_double_mut_rows(pid, s, (a, b), list(ALPHABET)))

        df = pd.DataFrame(rows)

        if DEDUP_CHILDREN_WITHIN_PAIR:
            before = len(df)
            df = df.drop_duplicates(subset=["peptide_sequence"], keep="first").reset_index(drop=True)
            print(f"[Info] 组 {pair_tag} 去重：{before} -> {len(df)}")

        # 每个组合单独输出一个文件
        per_path = out_dir / f"{FILENAME_PREFIX}_{pair_tag}.csv"
        _write_df(df, per_path, info=f"(组合 {a}-{b})")
        all_chunks.append(df)

    if WRITE_COMBINED:
        combo = pd.concat(all_chunks, axis=0, ignore_index=True)
        if GLOBAL_DEDUP_CHILDREN:
            before = len(combo)
            combo = combo.drop_duplicates(subset=["peptide_sequence"], keep="first").reset_index(drop=True)
            print(f"[Info] 合并后全局去重：{before} -> {len(combo)}")
        combo_path = out_dir / f"{FILENAME_PREFIX}_ALL_pairs.csv"
        _write_df(combo, combo_path, info="(全部组合合并)")

    # 简要规模提示（以“单点突变”为例）
    per_parent_single = (1 if INCLUDE_PARENTS else 0) + 2 * (len(ALPHABET) - 1)
    if GENERATE_DOUBLE_MUTS:
        per_parent_double = (len(ALPHABET) ** 2 - 1)  # 去掉 WT-WT
        print(f"[Info] 单个组合：每个亲本 ≈ 单突变 {per_parent_single} 行 + 双突变 {per_parent_double} 行")
    else:
        print(f"[Info] 单个组合：每个亲本 ≈ 单突变 {per_parent_single} 行")

if __name__ == "__main__":
    main()
