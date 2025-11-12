import os
import glob
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Specify the input folder containing FASTQ files (replace with your folder path)
input_dir = r'D:\Me\IMB\Data\Yuhao\NGS\Abl L1\01.RawData'
# Specify the output folder where the merged FASTA files will be saved
output_dir = r'D:\Me\IMB\Data\Yuhao\NGS\Abl L1\01.RawData'
os.makedirs(output_dir, exist_ok=True)

# Find all forward files ending with '_1.fq'
forward_files = glob.glob(os.path.join(input_dir, '*_1.fq'))

# Loop over each forward file and find its corresponding reverse file
for fwd_file in forward_files:
    # Determine the reverse file by replacing '_1.fq' with '_2.fq'
    rev_file = fwd_file.replace('_1.fq', '_2.fq')
    if not os.path.exists(rev_file):
        print(f"Reverse file not found for {fwd_file}, skipping this pair.")
        continue

    merged_records = []

    # Open the forward and reverse FASTQ files using SeqIO
    fwd_records = SeqIO.parse(fwd_file, "fastq")
    rev_records = SeqIO.parse(rev_file, "fastq")

    # Iterate through both files simultaneously
    for fwd_rec, rev_rec in zip(fwd_records, rev_records):
        # Optionally, check if the IDs match (assuming they do)
        # Get the reverse complement of the reverse read's sequence
        rev_seq_rc = rev_rec.seq.reverse_complement()
        # Merge the sequences by concatenation (you may insert a linker if needed)
        merged_seq = fwd_rec.seq + rev_seq_rc
        # Create a new SeqRecord for the merged sequence using the forward read's ID
        merged_rec = SeqRecord(merged_seq, id=fwd_rec.id, description="merged")
        merged_records.append(merged_rec)

    # Generate the output file name: use the forward file's base name (without '_1.fq') and add '_merged.fasta'
    base_name = os.path.basename(fwd_file)
    common_prefix = base_name.replace('_1.fq', '')
    output_file = os.path.join(output_dir, common_prefix + "_merged.fasta")

    # Write the merged records to the output file in FASTA format
    SeqIO.write(merged_records, output_file, "fasta")
    print(f"Merged file written: {output_file}")
