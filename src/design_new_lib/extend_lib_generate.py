# === 扩展库 + 2/5位完全放开（流式写出） ===
from itertools import product
import csv, os
from math import prod

# 基础序列（长度=8）
BASE_SEQ = "YLHWDYVW"
OUT_CSV  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\06.LibraryDesign\lib_expand_2.csv"

assert len(BASE_SEQ) == 8, "BASE_SEQ 必须是 8 个氨基酸"

# 20种标准氨基酸（完全随机/放开）
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# 扩展集合（并集方案）+ 2/5位完全放开
opts = {
    1: list("FYWH"),                   # 扩展：F/Y/W/H
    2: AA20,                           # 完全放开
    3: ['H','N','Q','S','T','D','E'],  # 扩展
    4: list("FYWH"),                   # 扩展
    5: AA20,                           # 完全放开
    6: list("FYWH"),                   # 扩展
    7: ['V','A','I','L','S','T'],      # 扩展
    8: list("FYWH"),                   # 扩展
}

var_positions = sorted(opts.keys())  # [1,2,3,4,5,6,7,8]
choice_lists  = [opts[p] for p in var_positions]

# 计算总组合数
total = 1
for p in var_positions:
    total *= len(opts[p])
print(f"预计生成序列数：{total:,}")  # 4,300,800

# 确保输出目录存在
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# 流式写出，避免占用大量内存
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["peptide_sequence"])  # 只写序列列

    seq_list = list(BASE_SEQ)
    count = 0

    # 通过笛卡尔积穷举所有组合
    for combo in product(*choice_lists):
        # 将变异位点写回序列（1-indexed -> 0-indexed）
        for pos, aa in zip(var_positions, combo):
            seq_list[pos-1] = aa
        writer.writerow(["".join(seq_list)])
        count += 1
        if count % 100000 == 0:
            print(f"已写出：{count:,}/{total:,}")

print(f"完成：{OUT_CSV}  |  共 {count:,} 条")
