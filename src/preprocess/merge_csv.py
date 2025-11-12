import os
import glob
import pandas as pd
import re
from functools import reduce

# Set the input directory containing the CSV files
input_dir = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing"

# Pattern to match the CSV files: e.g., BB_R1_L2_merged_counted.csv, BB_R2_L2_merged_counted.csv, etc.
file_pattern = os.path.join(input_dir, "BB_R*_L2_merged_counted.csv")
files = glob.glob(file_pattern)

# List to store each file's DataFrame
dataframes = []

# Regular expression to extract the prefix (e.g., R1, R2, etc.) from the file name
file_regex = re.compile(r"BB_(R\d+)_L2_merged_counted\.csv")

for file in files:
    base_name = os.path.basename(file)
    match = file_regex.search(base_name)
    if match:
        r_prefix = match.group(1)  # e.g., "R1"
    else:
        print(f"File naming did not match expected pattern: {file}")
        continue

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    # Rename the columns "frequency" and "counts" by adding the R* prefix
    df.rename(columns={"frequency": f"{r_prefix}_frequency", "counts": f"{r_prefix}_counts"}, inplace=True)
    # Keep only the necessary columns: peptide_sequence, and the renamed counts and frequency
    df = df[["peptide_sequence", f"{r_prefix}_counts", f"{r_prefix}_frequency"]]
    dataframes.append(df)

if dataframes:
    # Merge all DataFrames on 'peptide_sequence' using an outer join
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="peptide_sequence", how="outer"), dataframes)
    # Replace missing values (for sequences that did not occur in some files) with 0
    merged_df.fillna(0, inplace=True)
    
    # Optionally, sort the merged DataFrame by the peptide_sequence
    merged_df.sort_values(by="peptide_sequence", inplace=True)
    
    # Specify the output file for the merged result; here saving it in the same input directory
    output_file = os.path.join(input_dir, "BB_Allrounds.csv")
    merged_df.to_csv(output_file, index=False)
    print("Merged CSV file created:", output_file)
else:
    print("No CSV files found matching the pattern.")
