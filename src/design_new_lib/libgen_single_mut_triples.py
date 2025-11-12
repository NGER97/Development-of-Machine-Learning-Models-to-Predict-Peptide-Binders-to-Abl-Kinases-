# libgen_single_mut_triples.py
# 批量：对多个“三位点组合”生成单点突变文库（可选也生双突变、三突变）
# 不需要命令行参数，直接在顶部改配置

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

# 3) 三位点组合列表（1-based）。示例：只试 (2,5,7) 和 (3,6,8)
TRIPLE_POSITION_SETS = [(2, 5, 7), (3, 6, 8)]

# 3.1) 或者一键尝试所有 8 选 3 的组合（共 56 组）
AUTO_ALL_8C3 = True

# 4) 是否把亲本也写入（WT 行，position=0, mut_tag="WT"）
INCLUDE_PARENTS = True

# 5) 去重策略
DEDUP_CHILDREN_WITHIN_TRIPLE = False   # 单个三位点组合内，按 child 去重
GLOBAL_DEDUP_CHILDREN = False          # 所有组合合并后，跨组合全局去重

# 6) 是否额外生成“双突变、三突变库”
#    双突变：在该三位点中的任意一对位点同时变（两位都 != WT）
#    三突变：该三位点三位同时变（都 != WT）
GENERATE_DOUBLE_MUTS_WITHIN_TRIPLE = True
GENERATE_TRIPLE_MUTS = True

# 7) 输出
OUTPUT_DIR = r"Yuhao/NGS/Abl_L1/06.LibraryDesign/triples_mut"
FILENAME_PREFIX = "lib_singlemut"      # 文件名前缀（仅单突变也沿用这个前缀）
WRITE_COMBINED = True                  # 是否输出合并后的总表
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

def _norm_triples(triples):
    seen = set()
    out = []
    for a, b, c in triples:
        if not (1 <= a <= 8 and 1 <= b <= 8 and 1 <= c <= 8):
            raise ValueError(f"非法位点组合: {(a,b,c)}（需 1..8）")
        if len({a,b,c}) != 3:
            raise ValueError(f"位点不能重复: {(a,b,c)}")
        t = tuple(sorted((int(a), int(b), int(c))))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _single_mut_rows(parent_id, parent_seq, pos_triple, alphabet):
    """只做单点突变（限制在 pos_triple 三个位点上分别改）"""
    seq = parent_seq.strip().upper()
    rows = []
    tag = f"{pos_triple[0]}-{pos_triple[1]}-{pos_triple[2]}"

    if INCLUDE_PARENTS:
        rows.append({
            "parent_id": parent_id, "parent_seq": seq,
            "pos_set_tag": tag,
            "position": 0,          # 0 表示 WT
            "wt": "", "mut": "", "mut_tag": "WT",
            "peptide_sequence": seq,
            "scope": "wt"
        })

    for pos1 in pos_triple:  # 三个位点各自做单点突变
        i = pos1 - 1
        wt = seq[i]
        for aa in alphabet:
            if aa == wt:
                continue
            child = seq[:i] + aa + seq[i+1:]
            rows.append({
                "parent_id": parent_id, "parent_seq": seq,
                "pos_set_tag": tag,
                "position": pos1,  # 单点突变：记录真实位点
                "wt": wt, "mut": aa,
                "mut_tag": f"{wt}{pos1}{aa}",
                "peptide_sequence": child,
                "scope": "single"
            })
    return rows

def _double_mut_rows_within_triple(parent_id, parent_seq, pos_triple, alphabet):
    """在三个位点中的任意两位同时变（两位都 != WT）"""
    seq = parent_seq.strip().upper()
    tag = f"{pos_triple[0]}-{pos_triple[1]}-{pos_triple[2]}"
    rows = []
    for (p1, p2) in it.combinations(pos_triple, 2):
        i, j = p1 - 1, p2 - 1
        wt1, wt2 = seq[i], seq[j]
        for aa1 in alphabet:
            if aa1 == wt1: 
                continue
            for aa2 in alphabet:
                if aa2 == wt2:
                    continue
                child = seq[:i] + aa1 + seq[i+1:j] + aa2 + seq[j+1:]
                rows.append({
                    "parent_id": parent_id, "parent_seq": seq,
                    "pos_set_tag": tag,
                    "position": -1,   # -1 统一标记“双突变”
                    "wt": f"{wt1}{wt2}",
                    "mut": f"{aa1}{aa2}",
                    "mut_tag": f"{wt1}{p1}{aa1}_{wt2}{p2}{aa2}",
                    "peptide_sequence": child,
                    "scope": "double"
                })
    return rows

def _triple_mut_rows(parent_id, parent_seq, pos_triple, alphabet):
    """该三个位点三位同时变（都 != WT）"""
    seq = parent_seq.strip().upper()
    tag = f"{pos_triple[0]}-{pos_triple[1]}-{pos_triple[2]}"
    (p1, p2, p3) = pos_triple
    i, j, k = p1 - 1, p2 - 1, p3 - 1
    wt1, wt2, wt3 = seq[i], seq[j], seq[k]
    rows = []
    for aa1 in alphabet:
        if aa1 == wt1: 
            continue
        for aa2 in alphabet:
            if aa2 == wt2:
                continue
            for aa3 in alphabet:
                if aa3 == wt3:
                    continue
                child = seq[:i] + aa1 + seq[i+1:j] + aa2 + seq[j+1:k] + aa3 + seq[k+1:]
                rows.append({
                    "parent_id": parent_id, "parent_seq": seq,
                    "pos_set_tag": tag,
                    "position": -2,   # -2 统一标记“三突变”
                    "wt": f"{wt1}{wt2}{wt3}",
                    "mut": f"{aa1}{aa2}{aa3}",
                    "mut_tag": f"{wt1}{p1}{aa1}_{wt2}{p2}{aa2}_{wt3}{p3}{aa3}",
                    "peptide_sequence": child,
                    "scope": "triple"
                })
    return rows

def _build_triples():
    if AUTO_ALL_8C3:
        return _norm_triples(list(it.combinations(range(1, 9), 3)))
    else:
        if not TRIPLE_POSITION_SETS:
            raise ValueError("请设置 TRIPLE_POSITION_SETS 或打开 AUTO_ALL_8C3")
        return _norm_triples(TRIPLE_POSITION_SETS)

def _write_df(df, path, info=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["parent_id", "parent_seq", "pos_set_tag", "position", "scope", "wt", "mut", "mut_tag", "peptide_sequence"]
    df = df[cols]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 写入 {path} {info}（{len(df)} 行）")

def main():
    _check_inputs()
    triples = _build_triples()
    out_dir = Path(OUTPUT_DIR)
    all_chunks = []

    print(f"[Info] 共有 {len(triples)} 组三位点组合：{triples[:6]}{' ...' if len(triples)>6 else ''}")
    print(f"[Info] INCLUDE_PARENTS={INCLUDE_PARENTS}, 双突变={GENERATE_DOUBLE_MUTS_WITHIN_TRIPLE}, 三突变={GENERATE_TRIPLE_MUTS}")

    for t in triples:
        triple_tag = f"p{t[0]}_{t[1]}_{t[2]}"
        rows = []
        for pid, s in PARENTS:
            rows.extend(_single_mut_rows(pid, s, t, list(ALPHABET)))
            if GENERATE_DOUBLE_MUTS_WITHIN_TRIPLE:
                rows.extend(_double_mut_rows_within_triple(pid, s, t, list(ALPHABET)))
            if GENERATE_TRIPLE_MUTS:
                rows.extend(_triple_mut_rows(pid, s, t, list(ALPHABET)))

        df = pd.DataFrame(rows)

        if DEDUP_CHILDREN_WITHIN_TRIPLE:
            before = len(df)
            df = df.drop_duplicates(subset=["peptide_sequence"], keep="first").reset_index(drop=True)
            print(f"[Info] 组 {triple_tag} 去重：{before} -> {len(df)}")

        per_path = out_dir / f"{FILENAME_PREFIX}_{triple_tag}.csv"
        _write_df(df, per_path, info=f"(组合 {t[0]}-{t[1]}-{t[2]})")
        all_chunks.append(df)

    if WRITE_COMBINED:
        combo = pd.concat(all_chunks, axis=0, ignore_index=True)
        if GLOBAL_DEDUP_CHILDREN:
            before = len(combo)
            combo = combo.drop_duplicates(subset=["peptide_sequence"], keep="first").reset_index(drop=True)
            print(f"[Info] 合并后全局去重：{before} -> {len(combo)}")
        combo_path = out_dir / f"{FILENAME_PREFIX}_ALL_triples.csv"
        _write_df(combo, combo_path, info="(全部组合合并)")

    # 规模提示
    A = len(ALPHABET)
    per_parent_single = (1 if INCLUDE_PARENTS else 0) + 3 * (A - 1)
    msg = f"[Info] 单个三位点组合：每个亲本 ≈ 单突变 {per_parent_single} 行"
    if GENERATE_DOUBLE_MUTS_WITHIN_TRIPLE:
        per_parent_double = 3 * ((A - 1) ** 2)  # C(3,2) * (A-1)^2
        msg += f" + 双突变 {per_parent_double} 行"
    if GENERATE_TRIPLE_MUTS:
        per_parent_triple = (A - 1) ** 3
        msg += f" + 三突变 {per_parent_triple} 行"
    print(msg)

if __name__ == "__main__":
    main()
