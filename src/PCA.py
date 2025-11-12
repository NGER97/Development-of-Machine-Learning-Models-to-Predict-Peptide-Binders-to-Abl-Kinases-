import os
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Set file paths and threshold parameters

# 1.1 Folder containing counted.csv files (includes files like BB_ER&Score.csv)
score_folder = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing"

# 1.2 Folder containing features.csv files (includes corresponding -final.csv files)
features_folder = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures"

# 1.3 Output folder for PCA results
output_folder = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\04.PCA"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Threshold for frequency, can be customized
freq_threshold = 3.0  # Example: if >= 3.0 is considered "High", otherwise "Low"

# 2. Collect counted files

# Matches all files ending with _filtered.csv
counted_files = glob.glob(os.path.join(score_folder, "*_ER&Score.csv"))

# 3. Loop through each counted file

for cfile in counted_files:
    cfilename = os.path.basename(cfile)  
    # cfilename example: BB_L2_filtered.csv
    
    # Generate the features filename by replacing ".csv" with "-final.csv"
    base_name = cfilename.replace(".csv", "")
    features_filename = base_name + "-final.csv"
    ffile = os.path.join(features_folder, features_filename)

    # Check if the features file exists
    if not os.path.exists(ffile):
        print(f"[Warning] Could not find the features file {features_filename} for {cfilename}. Skipping...")
        continue

    # 3.1 Read data

    # The counted file should contain columns like peptide_sequence, counts, frequency, etc.
    df_counted = pd.read_csv(cfile)
    # The features file contains the corresponding feature data
    df_features = pd.read_csv(ffile)

    # If the number of rows does not match, print a warning
    if len(df_counted) != len(df_features):
        print(f"[Warning] The files {cfilename} and {features_filename} do not have matching rows. Skipping...")
        continue

    # 3.2 Merge the data directly by rows
    # Here we assume the two files' rows correspond exactly, so we simply use pd.concat
    df_merged = pd.concat([df_counted, df_features], axis=1)

    # 3.3 Feature selection and preprocessing
    # Assume these columns do not need PCA: peptide_sequence, counts, frequency, Protein (if it exists)
    non_feature_cols = ['peptide_sequence','R1_frequency' ,'R2_frequency' ,'R3_frequency' ,'R4_frequency' ,'R5_frequency', 'ER', 'Protein']
    feature_cols = [col for col in df_merged.columns if col not in non_feature_cols]

    if len(feature_cols) == 0:
        print(f"[Warning] No numeric features available for PCA in file {cfilename}. Skipping...")
        continue

    X = df_merged[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3.4 Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_merged['PC1'] = pca_result[:, 0]
    df_merged['PC2'] = pca_result[:, 1]

    # Check PCA explained variance ratio
    var_ratio = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio for file {cfilename}:", var_ratio)

    # 3.5 Group by frequency threshold
    df_merged['freq_category'] = np.where(
        df_merged['score'] >= freq_threshold,
        'High',
        'Low'
    )

    # 3.6 Save CSV with PCA results
    output_csv_path = os.path.join(output_folder, f"{base_name}_pca_result.csv")
    df_merged.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"PCA result CSV exported: {output_csv_path}")

# 3.7 绘制 PCA 散点图（使用连续 colormap 表示 frequency 的深浅）
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    plt.figure(figsize=(8, 6))

# 假设 df_merged 中有 'frequency' 列，数值越高代表频率越高
# 使用 plt.Normalize 标准化 frequency 的取值范围
    norm = plt.Normalize(df_merged['score'].min(), df_merged['score'].max())

# 选择一个单色调的 colormap，比如 Blues_r (反转的 Blues)
# 这样，频率越高（normalized 值高）对应的颜色就越浅
    cmap = plt.cm.plasma_r

# 用连续颜色绘制散点图
    sc = plt.scatter(
        df_merged['PC1'],
        df_merged['PC2'],
        c=df_merged['score'],  # 依据频率值上色
        cmap=cmap,
        norm=norm,
        alpha=0.6
    )

    plt.title(f"PCA (PC1 vs PC2) - {base_name}")
    plt.xlabel('PC1')
    plt.ylabel('PC2')

# 添加颜色条以显示 frequency 对应的颜色深浅
    cbar = plt.colorbar(sc)
    cbar.set_label('Frequency')

    plt.tight_layout()

    output_fig_path = os.path.join(output_folder, f"{base_name}_pca_plot.png")
    plt.savefig(output_fig_path, dpi=300)
    plt.close()
    print(f"PCA scatter plot exported: {output_fig_path}")      

    print("Processing complete!")
