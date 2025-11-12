import os
import glob
import pandas as pd

# Specify the directory containing your CSV files
input_dir = r'D:\Me\IMB\Data\Yuhao\NGS\2023-Nov\02.DataPreprocessing'

# Define the threshold to filter
threshold = 0.01

# Collect all CSV files in the input directory that match the pattern "*_merged_counted.csv"
csv_files = glob.glob(os.path.join(input_dir, '*_merged_counted.csv'))

for inputfile in csv_files:
    # Read the CSV file
    df = pd.read_csv(inputfile)

    # Ensure that your CSV contains the necessary columns: e.g., "counts" and "frequency"
    required_cols = {"counts", "frequency"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"Skipping '{inputfile}' because it is missing columns: {missing_cols}")
        continue

    # Filter out rows where frequency < threshold
    df_filtered = df[df['frequency'] >= threshold].copy()

    # Remove the sequence 'YLHWDYVW'
    df_filtered = df_filtered[df_filtered['peptide_sequence'] != 'YLHWDYVW']

    # Recalculate frequency relative to the remaining rows
    # Here, we assume frequency = counts / sum_of_counts in the filtered dataset
    total_counts_after_filter = df_filtered['counts'].sum()
    df_filtered['frequency'] = df_filtered['counts'] / total_counts_after_filter

    # Construct output filename: replace '_merged_counted.csv' with '_filtered.csv'
    base_name = os.path.basename(inputfile)
    # e.g., if base_name is "Sample_merged_counted.csv", output should be "Sample_filtered.csv"
    output_name = base_name.replace('_merged_counted.csv', f'_filtered{threshold}.csv')
    output_path = os.path.join(input_dir, output_name)

    # Save the filtered DataFrame to a new CSV
    df_filtered.to_csv(output_path, index=False)
    print(f"Filtered file saved as: {output_path}")
