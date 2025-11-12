import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Specify the input directory
input_dir = r'D:\Me\IMB\Data\Yuhao\NGS\2023-Nov\02.DataPreprocessing'

# Collect all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

# ========== 1. Define bins for 'counts' ==========
bin_labels_counts = [
    "counts=1",
    "1<counts<=10",
    "10<counts<=100",
    "100<counts<=1000",
    "counts>1000"
]

def get_counts_bin_label(count):
    """Return the bin label for a given count value."""
    if count == 1:
        return bin_labels_counts[0]
    elif 1 < count <= 10:
        return bin_labels_counts[1]
    elif 10 < count <= 100:
        return bin_labels_counts[2]
    elif 100 < count <= 1000:
        return bin_labels_counts[3]
    else:
        return bin_labels_counts[4]

# ========== 2. Define bins for 'frequency' ==========
# Based on your requirement: <0.001, 0.001-0.01, 0.01-0.1, 0.1-1, 1-10, >10
bin_labels_freq = [
    "<0.001",
    "0.001-0.01",
    "0.01-0.1",
    "0.1-1",
    "1-10",
    ">10"
]

def get_frequency_bin_label(freq):
    """Return the bin label for a given frequency value."""
    if freq < 0.001:
        return bin_labels_freq[0]
    elif 0.001 <= freq < 0.01:
        return bin_labels_freq[1]
    elif 0.01 <= freq < 0.1:
        return bin_labels_freq[2]
    elif 0.1 <= freq < 1:
        return bin_labels_freq[3]
    elif 1 <= freq < 10:
        return bin_labels_freq[4]
    else:
        return bin_labels_freq[5]

for inputfile in csv_files:
    # Read the CSV file
    df = pd.read_csv(inputfile)

    # Make sure the file contains the required columns: "counts" and "frequency"
    required_columns = {"counts", "frequency"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        print(f"The file '{inputfile}' is missing the following columns: {missing_cols}. Please check your data.")
        continue

    # ========== A. Bin by 'counts' and compute the proportion of sequences ==========
    df['counts_bin'] = df['counts'].apply(get_counts_bin_label)
    total_sequences = len(df)
    grouped_sequence_counts = df.groupby('counts_bin').size()
    sequence_proportions = grouped_sequence_counts / total_sequences
    sequence_proportions = sequence_proportions.reindex(bin_labels_counts, fill_value=0)

    # ========== B. Bin by 'frequency' and compute the proportion of sequences ==========
    df['freq_bin'] = df['frequency'].apply(get_frequency_bin_label)
    grouped_freq_counts = df.groupby('freq_bin').size()
    freq_proportions = grouped_freq_counts / total_sequences
    freq_proportions = freq_proportions.reindex(bin_labels_freq, fill_value=0)

    # Extract the base name of the file (without extension) to use in plot titles and filenames
    base_name = os.path.splitext(os.path.basename(inputfile))[0]

    # ========== 3. Plot counts distribution (proportion of sequences) ==========
    plt.bar(sequence_proportions.index, sequence_proportions.values)
    plt.xlabel("Count Bins")
    plt.ylabel("Proportion of Sequences")
    plt.title(f"Proportion of Sequences by Count Bin (Source: {base_name})")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_png_1 = os.path.join(input_dir, f"{base_name}_sequence_proportion.png")
    plt.savefig(out_png_1)
    plt.close()

    # ========== 4. Plot frequency distribution (proportion of sequences) ==========
    plt.bar(freq_proportions.index, freq_proportions.values)
    plt.xlabel("Frequency Bins")
    plt.ylabel("Proportion of Sequences")
    plt.title(f"Frequency Distribution in Sequences (Source: {base_name})")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_png_2 = os.path.join(input_dir, f"{base_name}_frequency_distribution.png")
    plt.savefig(out_png_2)
    plt.close()

    print(f"Finished processing file: {inputfile}\n"
          f"  - Counts distribution chart saved as: {out_png_1}\n"
          f"  - Frequency distribution chart saved as: {out_png_2}\n")
