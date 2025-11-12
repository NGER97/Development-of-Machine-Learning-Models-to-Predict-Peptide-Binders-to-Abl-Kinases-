import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

# Path to your merged CSV file (update this to your actual file path)
csv_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_Allrounds.csv"

# Load the merged CSV into a DataFrame
merged_df = pd.read_csv(csv_path)

# Identify all replicate frequency and counts columns.
# We assume columns are named like "R1_frequency" and "R1_counts" (for R1, R2, …).
replicate_freq_cols = [col for col in merged_df.columns if col.endswith('_frequency') and col.startswith('R')]
replicates = sorted({col.split('_')[0] for col in replicate_freq_cols})  # e.g., ['R1', 'R2', 'R3', 'R4', 'R5']

# Define frequency bins (values assumed to be in percentages) and labels.
bins = [0, 0.01, 0.1, 1, 10, 100]
labels = ["<0.01", "0.01-0.1", "0.1-1", "1-10", "10-100"]

# Dictionary to store binned count proportions for each replicate
bin_summary = {}

for rep in replicates:
    freq_col = f"{rep}_frequency"
    counts_col = f"{rep}_counts"
    
    # Get the frequency and counts series for the replicate
    freq_series = merged_df[freq_col]
    counts_series = merged_df[counts_col]
    
    # Bin the frequency values according to the defined bins.
    # Use include_lowest=True to ensure values equal to 0 fall into the first bin.
    binned = pd.cut(freq_series, bins=bins, labels=labels, right=True, include_lowest=True)
    
    # Create a temporary DataFrame combining the binned frequency and the counts.
    temp_df = pd.DataFrame({
        "bin": binned,
        "counts": counts_series
    })
    
    # Sum the counts for each frequency bin.
    bin_sum = temp_df.groupby("bin")["counts"].sum()
    
    # Total counts for this replicate
    total_counts = counts_series.sum()
    
    # Compute the proportion of total counts per bin.
    proportions = (bin_sum / total_counts).sort_index()
    
    # Ensure all bins are present (fill missing bins with 0).
    proportions = proportions.reindex(labels, fill_value=0)
    
    # Store the proportions for the current replicate.
    bin_summary[rep] = proportions

# Convert the summary dictionary into a DataFrame.
# The DataFrame will have replicate names as columns in the dictionary; here we transpose so rows = replicates.
summary_df = pd.DataFrame(bin_summary).T

# Optional: sort replicates (if needed)
summary_df = summary_df.sort_index()

# Plot a stacked bar chart.
fig, ax = plt.subplots(figsize=(8, 6))
summary_df.plot(kind='bar', stacked=True, ax=ax, colormap="tab20")

# Set labels and title.
ax.set_xlabel("Rounds")
ax.set_ylabel("Proportion of Total Counts")
ax.set_title("Distribution of Total Counts by Frequency Bins")

# Format the y-axis as percentages.
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Place legend outside the plot.
plt.legend(title="Frequency bins", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save the figure to a file.
plt.savefig("Distribution of Total Counts by Frequency Bins.png", dpi=300)

# Show the plot.
plt.show()