import pandas as pd

threshold = 0

# Define the input and output CSV file paths
input_csv = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_ER&Score.csv"
output_csv = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_Score_sorted.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Filter rows where score > threshold and sort them by score in descending order
df_filtered = df[df["score"] > threshold].sort_values("score", ascending=False)

# Save the filtered and sorted DataFrame to a new CSV file
df_filtered.to_csv(output_csv, index=False)

print(f"Filtered and sorted CSV saved to: {output_csv}")
