#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert peptide sequences stored in a CSV file into a FASTA file
compatible with iLearn.

Usage (run directly):
    python csv2fasta.py

You may modify the three path variables below as needed, or supply
your own via command-line arguments (see argparse section).

Author: ChatGPT
"""

import os
import argparse
import pandas as pd

# ------------------------------------------------------------------
# Default paths ―― adjust if necessary
ilearn_master_folder = r"D:/Me/IMB/Data/iLearn/"  # currently unused here
input_csv            = r"Yuhao\NGS\Abl L1\06.LibraryDesign\lib_expand.csv"
output_dir           = r"D:/Me/IMB/Data/Yuhao/NGS/Abl L1/03.GenerateFeatures/classification/"
label = 1
# ------------------------------------------------------------------

def csv_to_fasta(csv_path: str, fasta_path: str, seq_col: str = "peptide_sequence") -> None:
    """
    Read sequences from a CSV and write them to FASTA.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    fasta_path : str
        Path where the output FASTA file will be written.
    seq_col : str, default "peptide_sequence"
        Column that contains peptide sequences.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found in {csv_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)

    # Write FASTA
    with open(fasta_path, "w", encoding="utf-8") as fasta:
        for idx, seq in enumerate(df[seq_col], start=1):
            seq = str(seq).strip().upper()
            if not seq:
                continue  # skip empty entries
            fasta.write(f">pep{idx}|{label}|training\n{seq}\n")

    print(f"FASTA written to: {fasta_path} (total sequences: {idx})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CSV containing peptide sequences to FASTA.")
    parser.add_argument(
        "-i", "--input", default=input_csv,
        help="Path to the input CSV file (default: preset path).")
    parser.add_argument(
        "-o", "--outdir", default=output_dir,
        help="Directory to save the generated FASTA file (default: preset path).")
    parser.add_argument(
        "-c", "--column", default="peptide_sequence",
        help="Name of the column containing sequences (default: 'peptide_sequence').")
    args = parser.parse_args()

    # Derive output FASTA filename
    csv_basename = os.path.splitext(os.path.basename(args.input))[0]
    fasta_path = os.path.join(args.outdir, f"{csv_basename}.fa")

    csv_to_fasta(args.input, fasta_path, args.column)


if __name__ == "__main__":
    main()
