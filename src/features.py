#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate AAC, DPC, CKSAAGP(gap=3), PAAC(lamada=4, weight=0.05) features
并横向合并为一张表
----------------------------------------------------------------------
Yuhao, 2025-06-24
"""
import os
import pandas as pd
from pathlib import Path
from subprocess import run, CalledProcessError

# ───────────────────────────【路径配置】─────────────────────────────
ilearn_root      = r"D:/Me/IMB/Data/iLearn"                 # 含 descproteins/
desc_dir         = os.path.join(ilearn_root, "descproteins") # 底层脚本目录
input_csv        = r"Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_R1toR4_significant.csv"
output_dir       = r"D:/Me/IMB/Data/Yuhao/NGS/Abl L1/03.GenerateFeatures"
os.makedirs(output_dir, exist_ok=True)

# ───────────────────────────【CSV → FASTA】─────────────────────────
df = pd.read_csv(input_csv)
seqs = df.iloc[:, 0].astype(str).str.replace(r"[^ACDEFGHIKLMNPQRSTVWY]", "", regex=True)  # 仅保留20种AA
fasta_path = Path(output_dir) / "tmp_sequences.fasta"
with open(fasta_path, "w") as fh:
    for i, s in enumerate(seqs, 1):
        fh.write(f">pep{i}|0|training\n{s}\n")
print("FASTA written:", fasta_path)

# ───────────────────────────【特征脚本及参数】───────────────────────
descriptors = [
    # (脚本文件, 生成文件名(不带后缀), 额外 CLI 参数列表)
    ("AAC.py",     "AAC",     []),
    ("DPC.py",     "DPC",     []),
    ("CKSAAGP.py", "CKSAAGP", ["--gap", "3"]),
    ("PAAC.py",    "PAAC",    ["--lamada", "4", "--weight", "0.05"]),
]

def call_descriptor(pyfile: str, tag: str, extra_args: list):
    script = os.path.join(desc_dir, pyfile)
    out_csv = os.path.join(output_dir, f"{tag}.csv")
    cmd = [
        "python", script,
        "--file", str(fasta_path),
        "--format", "csv",
        "--out",   out_csv,
        *extra_args
    ]
    print("Running:", " ".join(cmd))
    try:
        run(cmd, check=True)
    except CalledProcessError as e:
        raise RuntimeError(f"{tag} generation failed: {e}") from None
    if not os.path.isfile(out_csv):
        raise FileNotFoundError(f"{out_csv} not generated.")
    return out_csv

generated_files = [call_descriptor(*desc) for desc in descriptors]

# ───────────────────────────【读取并合并】───────────────────────────
merged_parts = []
for idx, (tag, csv_path) in enumerate(zip([d[1] for d in descriptors], generated_files)):
    df_feat = pd.read_csv(csv_path, index_col=0).drop(columns=["label"], errors="ignore")
    # 为避免列冲突，加前缀
    df_feat.columns = [f"{tag}_{i}" for i in range(len(df_feat.columns))]
    if idx != 0:                       # 仅保留第一张表的索引列
        df_feat = df_feat.iloc[:, 1:]
    merged_parts.append(df_feat)

final_df = pd.concat(merged_parts, axis=1).reset_index(names=["Protein"])
final_out = os.path.join(output_dir, "BB_R1toR4_significant_features_all.csv")
final_df.to_csv(final_out, index=False)
print("All features merged to:", final_out)
