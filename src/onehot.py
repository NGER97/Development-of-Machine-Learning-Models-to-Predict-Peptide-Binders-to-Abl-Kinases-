import os
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ========== 1. Set paths and parameters ==========
# Folder containing counted files
counted_folder = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing"
# Output folder for one-hot encoded data and plots
output_folder = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\05.Onehot"

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Frequency threshold for grouping in plots (not used in continuous color approach)
freq_threshold = 3.0

# Define the 20 common amino acids (adjust or add as needed)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
mapping = {aa: idx for idx, aa in enumerate(amino_acids)}

# ========== 2. Define one-hot encoding function ==========
def one_hot_encode(seq, max_len, mapping):
    """
    One-hot encodes a single peptide sequence.
    - seq: peptide sequence as a string
    - max_len: maximum sequence length in the file (to standardize vector lengths)
    - mapping: dictionary mapping amino acid characters to indices
    Returns a flattened 1D vector of length max_len * len(mapping).
    """
    # Create a zero matrix of size (max_len, len(mapping))
    encoding = np.zeros((max_len, len(mapping)), dtype=int)
    # Set the corresponding position to 1 for each character in the sequence
    for i, aa in enumerate(seq):
        if i >= max_len:
            break  # Prevent exceeding the maximum length (rarely happens)
        if aa in mapping:
            encoding[i, mapping[aa]] = 1
    # Flatten the matrix into a 1D vector and return it
    return encoding.flatten()

# ========== 3. Process counted files ==========
# Match all files ending with _filtered.csv
counted_files = glob.glob(os.path.join(counted_folder, "*_ER&Score.csv"))

for cfile in counted_files:
    base_name = os.path.basename(cfile).replace(".csv", "")
    print(f"Processing file: {base_name}")

    # Read the counted file (which should include 'peptide_sequence', 'counts', 'frequency', etc.)
    df = pd.read_csv(cfile)
    if "peptide_sequence" not in df.columns:
        print(f"[Warning] File {base_name} is missing the 'peptide_sequence' column, skipping.")
        continue

    # Determine the maximum peptide sequence length for one-hot encoding
    max_len = df['peptide_sequence'].apply(lambda x: len(str(x))).max()
    print(f"Maximum sequence length: {max_len}")

    # Perform one-hot encoding for each peptide sequence
    df['onehot'] = df['peptide_sequence'].apply(lambda x: one_hot_encode(str(x), max_len, mapping))

    # Expand the 'onehot' column (list/array) into separate columns
    onehot_df = pd.DataFrame(df['onehot'].tolist(), index=df.index)
    # Concatenate the original data with the one-hot encoded data
    df_encoded = pd.concat([df, onehot_df], axis=1)

    # ========== 4. PCA Dimensionality Reduction ==========
    # Use the one-hot encoded columns for PCA
    feature_cols = onehot_df.columns.tolist()  # All one-hot encoded columns
    X = df_encoded[feature_cols].values

    # Standardize the data (optional: for 0/1 data, standardization gives each column zero mean and unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA to extract the first two principal components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_encoded['PC1'] = pca_result[:, 0]
    df_encoded['PC2'] = pca_result[:, 1]

    # Output the PCA explained variance ratio for reference
    print(f"PCA explained variance ratio for {base_name}: {pca.explained_variance_ratio_}")

    # If 'frequency' is not present, assign NaN so the scatter plot will default to gray
    if 'score' not in df_encoded.columns:
        df_encoded['score'] = np.nan

    # ========== 5. Save the one-hot encoded data ==========
    output_csv_path = os.path.join(output_folder, f"{base_name}_onehot_encoded.csv")
    df_encoded.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"One-hot encoded CSV exported: {output_csv_path}")

    # ========== 6. Plot PCA Scatter Plot ==========
    plt.figure(figsize=(8, 6))

    if not df_encoded['score'].isnull().all():
        # Use a continuous colormap based on frequency values if valid frequency data exists
        # Normalize the frequency values
        norm = plt.Normalize(df_encoded['score'].min(), df_encoded['score'].max())
        # Choose a prominent colormap (Oranges_r) so that higher frequency appears lighter
        cmap = plt.cm.plasma_r
        # Plot the scatter using frequency values for color mapping
        sc = plt.scatter(
            df_encoded['PC1'],
            df_encoded['PC2'],
            c=df_encoded['score'],
            cmap=cmap,
            norm=norm,
            alpha=0.6,
            s=10 
        )
        # Add a colorbar to indicate the frequency-to-color mapping
        cbar = plt.colorbar(sc)
        cbar.set_label('score')
    else:
        # If there is no valid frequency data, plot all points in gray
        plt.scatter(df_encoded['PC1'], df_encoded['PC2'], color='gray', alpha=0.6, label='NA')
        plt.legend(title='score')

    plt.title(f"PCA of One-Hot Encoded Peptide Sequences - {base_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    output_fig_path = os.path.join(output_folder, f"{base_name}_onehot_pca_plot.png")
    plt.savefig(output_fig_path, dpi=300)
    plt.close()
    print(f"PCA scatter plot exported: {output_fig_path}\n")

print("All files have been processed!")
