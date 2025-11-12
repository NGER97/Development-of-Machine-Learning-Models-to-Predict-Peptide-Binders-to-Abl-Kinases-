#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV ▶ FASTA ▶ 多种 iLearn 描述子（Binary, AAC, …）
"""

import os
import subprocess
import sys
import pandas as pd

# ---------------------------------------------------------------
CONFIG = {
    "ILEARN_ROOT": r"D:/Me/IMB/Data/iLearn/",
    "INPUT_CSV": r"Yuhao\NGS\Abl_L1\06.LibraryDesign\triples_mut\lib_singlemut_ALL_triples.csv",
    "FASTA_DIR": r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification\lib_singlemut_ALL_triples.fa",
    "FEATURE_DIR": r"Yuhao\NGS\Abl L1\03.GenerateFeatures\classification",
    "SEQ_COLUMN": "peptide_sequence",
    # 这里按需增删描述子就行
    "DESCRIPTORS": [
        # 调用 iLearn-protein-basic.py（默认脚本）的例子
        {"name": "binary", "script": "iLearn-protein-basic.py", "args": ["--method", "binary"]},
        {"name": "AAC",    "script": "iLearn-protein-basic.py", "args": ["--method", "AAC"]},
        {"name": "DPC",    "script": "iLearn-protein-basic.py", "args": ["--method", "DPC"]},
        # 需要专用脚本 + 额外参数（gap=3）的 CKSAAGP
        {"name": "CKSAAGP_gap3",
         "script": os.path.join("descproteins", "CKSAAGP.py"),
         "args": ["--gap", "3"]},
        {"name": "PAAC_l5_w0.05",
         "script": os.path.join("descproteins", "PAAC.py"),
         "args": ["--lamada", "5", "--weight", "0.05"]},
    ]
}
# ---------------------------------------------------------------

def csv_to_fasta(csv_path: str, fasta_path: str, seq_col: str) -> None:
    df = pd.read_csv(csv_path)
    if seq_col not in df.columns:
        raise ValueError(f"列 '{seq_col}' 不存在于 {csv_path}")

    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as fh:
        for idx, seq in enumerate(df[seq_col], 1):
            seq = str(seq).strip().upper()
            if seq:
                fh.write(f">pep{idx}|0|training\n{seq}\n")
    print(f"[INFO] FASTA 写入 {fasta_path}（{idx} 条序列）")

def run_descriptor(fasta_path: str, ilearn_root: str, out_dir: str, desc_cfg: dict) -> str:
    """
    根据 desc_cfg 调用对应脚本生成特征。
    desc_cfg 需包含:
        name   : 输出文件用名
        script : 相对 ilearn_root 的脚本路径
        args   : 需附加到命令行的参数列表
    """
    script_abspath = os.path.join(ilearn_root, desc_cfg["script"])
    if not os.path.isfile(script_abspath):
        raise FileNotFoundError(f"脚本不存在: {script_abspath}")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(fasta_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{desc_cfg['name']}.csv")

    cmd = [
        sys.executable,
        script_abspath,
        "--file", fasta_path,
        "--format", "csv",    # 大部分脚本通用；若某脚本不支持可在 args 覆盖
        "--out", out_path,
        *desc_cfg["args"]     # 追加额外参数
    ]

    print("[INFO]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print("[stdout]\n", res.stdout)
    if res.stderr:
        print("[stderr]\n", res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"{desc_cfg['name']} 运行失败，退出码 {res.returncode}")

    print(f"[SUCCESS] 特征 {desc_cfg['name']} 写入 {out_path}")
    return out_path

def merge_feature_files(feature_dir: str, base_name: str,
                        descriptors: list) -> str:
    """
    读取 <base_name>_<desc['name']>.csv（无表头），
    在行索引上 concat，自动给列加前缀，写出 *_ALLfeatures.csv
    """
    merged_parts = []
    keep_class = True   # 仅首文件保留 class

    for desc in descriptors:
        path = os.path.join(feature_dir, f"{base_name}_{desc['name']}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到特征文件: {path}")

        # header=None → 不把首行当表头
        df = pd.read_csv(path, header=None)

        # 拆分 class 与特征
        y = df.iloc[:, 0]                # 第一列 = class
        X = df.iloc[:, 1:]               # 其余列 = 特征

        # 给特征列加前缀，避免重名
        X.columns = [f"{desc['name']}::f{i+1}" for i in range(X.shape[1])]

        # 组合
        if keep_class:
            part = pd.concat([y.rename("class"), X], axis=1)
            keep_class = False           # 之后的文件不再保留 class
        else:
            part = X

        merged_parts.append(part)

    merged_df = pd.concat(merged_parts, axis=1)

    out_path = os.path.join(feature_dir, f"{base_name}_ALLfeatures.csv")
    merged_df.to_csv(out_path, index=False)
    print(f"[MERGE] 已生成总特征表: {out_path}")
    return out_path

def main():
    csv_path = CONFIG["INPUT_CSV"]
    fasta_name = os.path.splitext(os.path.basename(csv_path))[0] + ".fa"
    fasta_path = os.path.join(CONFIG["FASTA_DIR"], fasta_name)

    # Step 1: CSV → FASTA
    if not os.path.exists(fasta_path):
        csv_to_fasta(csv_path, fasta_path, CONFIG["SEQ_COLUMN"])
    else:
        print(f"[INFO] FASTA 已存在：{fasta_path}")

    # Step 2: FASTA → 各种描述子
    for desc in CONFIG["DESCRIPTORS"]:
        run_descriptor(
            fasta_path=fasta_path,
            ilearn_root=CONFIG["ILEARN_ROOT"],
            out_dir=CONFIG["FEATURE_DIR"],
            desc_cfg=desc
        )

    # Step 3: 合并所有特征
    base_name = os.path.splitext(os.path.basename(CONFIG["INPUT_CSV"]))[0]
    merge_feature_files(
        feature_dir=CONFIG["FEATURE_DIR"],
        base_name=base_name,
        descriptors=CONFIG["DESCRIPTORS"]
    )

if __name__ == "__main__":
    main()
