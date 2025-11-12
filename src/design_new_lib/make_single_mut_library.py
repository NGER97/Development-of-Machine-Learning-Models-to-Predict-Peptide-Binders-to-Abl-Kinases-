# libgen_single_mut.py
# 生成 8-mer 单点突变文库（一次只改一个位点）
# ✅ 全局变量可改，无需命令行参数
# 输出列：parent_id, parent_seq, position(1-based), wt, mut, mut_tag, peptide_sequence

from pathlib import Path
import pandas as pd

# ========= 你只需要改这里 =========
# 1) 直接在这里列出你的亲本序列（按顺序编号）。可以只放一条。
PARENTS = [
    # ("可选自定义ID", "序列")
    ("seq1", "YLHWDYVW"),
    # ("seq2", "ABCDEFGH"),
]

# 2) 要突变到的氨基酸字母表（默认 20 标准氨基酸）
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# 3) 指定要突变的位点（1-based）。None 表示全长 1..8 都突变
POSITIONS = None  # 例如只改 2 和 5 位： [2, 5]

# 4) 是否将亲本序列也写入结果（方便后续算 Δscore）
INCLUDE_PARENTS = True

# 5) 输出路径
OUTPUT_CSV = r"Yuhao/NGS/Abl_L1/06.LibraryDesign/lib_singlemut.csv"

# 6) 如多个亲本导致产生相同子序列，是否去重（默认不去重，便于追溯 parent）
DEDUP_CHILDREN = False
# ==================================


def _check_inputs():
    assert isinstance(PARENTS, (list, tuple)) and len(PARENTS) > 0, "PARENTS 不能为空"
    for pid, seq in PARENTS:
        if not pid:
            raise ValueError("parent_id 不能为空")
        if not isinstance(seq, str):
            raise ValueError(f"{pid} 的序列不是字符串")
        s = seq.strip().upper()
        if len(s) != 8:
            raise ValueError(f"{pid} 的序列长度不是 8：{s} (len={len(s)})")
        bad = set(s) - set(ALPHABET)
        if bad:
            raise ValueError(f"{pid} 的序列含非字母表字符 {bad}；当前 ALPHABET={ALPHABET}")
    if POSITIONS is not None:
        if any((p < 1 or p > 8) for p in POSITIONS):
            raise ValueError("POSITIONS 需为 1..8 的整数列表")


def _single_mutants(parent_id: str, parent_seq: str, positions, alphabet):
    seq = parent_seq.strip().upper()
    rows = []

    # 可选：把亲本也加入，便于后续对比
    if INCLUDE_PARENTS:
        rows.append({
            "parent_id": parent_id,
            "parent_seq": seq,
            "position": 0,          # 0 表示 WT，不是实际位点
            "wt": "",
            "mut": "",
            "mut_tag": "WT",
            "peptide_sequence": seq
        })

    pos_list = positions if positions is not None else range(1, 9)  # 1..8
    for pos1 in pos_list:
        i = pos1 - 1  # 0-based
        wt = seq[i]
        for aa in alphabet:
            if aa == wt:
                continue
            child = seq[:i] + aa + seq[i+1:]
            rows.append({
                "parent_id": parent_id,
                "parent_seq": seq,
                "position": pos1,     # 1-based
                "wt": wt,
                "mut": aa,
                "mut_tag": f"{wt}{pos1}{aa}",
                "peptide_sequence": child
            })
    return rows


def main():
    _check_inputs()

    all_rows = []
    for (pid, s) in PARENTS:
        all_rows.extend(_single_mutants(pid, s, POSITIONS, list(ALPHABET)))

    df = pd.DataFrame(all_rows)

    # 可选去重（基于生成后的 child 序列去重）
    if DEDUP_CHILDREN:
        before = len(df)
        df = df.drop_duplicates(subset=["peptide_sequence"], keep="first").reset_index(drop=True)
        print(f"[Info] 去重：{before} -> {len(df)}")

    # 统一列顺序
    cols = ["parent_id", "parent_seq", "position", "wt", "mut", "mut_tag", "peptide_sequence"]
    df = df[cols]

    # 输出
    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 统计信息
    parents_n = len(PARENTS)
    pos_n = len(POSITIONS) if POSITIONS is not None else 8
    per_parent = (1 if INCLUDE_PARENTS else 0) + pos_n * (len(ALPHABET) - 1)
    print(f"[OK] 写入: {out_path}")
    print(f"[OK] 亲本数: {parents_n}，位点数: {pos_n}，字母表: {len(ALPHABET)}")
    print(f"[OK] 每个亲本生成: {per_parent} 条；总计: {len(df)} 条")

if __name__ == "__main__":
    main()
