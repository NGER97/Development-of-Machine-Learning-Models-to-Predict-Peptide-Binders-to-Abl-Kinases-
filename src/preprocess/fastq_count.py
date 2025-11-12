"""
Created by Yuhao Rao

"""
r"""
fastq_count.py

Description:
    This script searches for files matching "*_L2_merged.fasta" in the specified input directory,
    parses the FASTA sequences to find a specific peptide motif using a regular expression,
    counts the occurrences of the motif captured as 8 variable amino acids between "CGAIYAA" and "GC",
    calculates the frequency of each motif occurrence, and then saves the results in a CSV file
    with the file name pattern *_counted.csv.

Usage:
    Default input directory: D:\Me\IMB\Data\Yuhao\NGS\Abl L1\01.RawData
    Default output directory: D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing
    File pattern: *_L2_merged.fasta

    Run the script with:
        python fastq_count.py
"""

import os
import glob
import re
import csv
from Bio import SeqIO

# Global motif pattern: captures 8 amino acids between 'CGAIYAA' and 'GC'
MOTIF_PATTERN = re.compile(r'CGAIYAA([A-Z]{8})GC')
# Define the input and output directories (update paths if necessary)
input_dir = r'D:\Me\IMB\Data\Yuhao\NGS\Abl L1\01.RawData'
output_dir = r'D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing'
os.makedirs(output_dir, exist_ok=True)

def find_motif(protein_seq):
    """
    Searches for the target motif in the protein sequence.

    Parameters:
        protein_seq (str): The translated protein sequence.

    Returns:
        str or None: The captured 8 amino acid segment if found, otherwise None.
    """
    match = MOTIF_PATTERN.search(str(protein_seq))
    return match.group(1) if match else None

def process_fasta_file(input_file, output_dir):
    """
    Processes a single FASTA file to count the occurrences of the target motif,
    and exports the results to a CSV file.

    Parameters:
        input_file (str): Path to the input FASTA file.
        output_dir (str): Directory where the output CSV file will be saved.
    """
    # Construct the output CSV file name based on the input file name
    base_name = os.path.basename(input_file)
    file_root, _ = os.path.splitext(base_name)
    output_csv = os.path.join(output_dir, f"{file_root}_counted.csv")

    peptide_counts = {}
    total_count = 0

    # Iterate over each record in the FASTA file
    for record in SeqIO.parse(input_file, "fasta"):
        dna_seq = record.seq
        motif_found = None

        # Try all three reading frames
        for frame in range(3):
            protein_seq = dna_seq[frame:].translate(to_stop=False)
            motif = find_motif(protein_seq)
            if motif:
                motif_found = motif
                break  # Exit once the motif is found

        if motif_found:
            peptide_counts[motif_found] = peptide_counts.get(motif_found, 0) + 1
            total_count += 1

    # Write the results to a CSV file with columns: peptide_sequence, counts, frequency
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['peptide_sequence', 'counts', 'frequency'])
        for peptide, count in peptide_counts.items():
            frequency = (count / total_count * 100) if total_count > 0 else 0
            writer.writerow([peptide, count, frequency])

    print(f"CSV file has been written: {output_csv}")

def main():
    # Find all FASTA files matching the pattern "*_L2_merged.fasta" in the input directory
    fasta_files = glob.glob(os.path.join(input_dir, '*_L2_merged.fasta'))

    if not fasta_files:
        print("No files matching '*_L2_merged.fasta' were found in the input directory.")
        return

    # Process each FASTA file
    for input_file in fasta_files:
        process_fasta_file(input_file, output_dir)

if __name__ == '__main__':
    main()
