import pandas as pd

# 定义全局变量，用于增加到每个counts中的伪读数
PSEUDO_COUNT = 1  # 你可以根据需要修改这个值

# Path to your merged CSV file
csv_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_remove.csv"
# Output path for the new CSV
output_path = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_remove_add_pseudocount.csv"


def process_csv(input_file: str, output_file: str):
    """
    读取 CSV 文件，对 counts 列加上伪读数，并更新 frequency 列，
    最后保存修改后的 CSV 文件到 output_file.
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 找出所有 counts 列并增加伪读数
    for col in df.columns:
        if col.endswith('_counts'):
            df[col] = df[col] + PSEUDO_COUNT

    # 根据更新后的 counts 重新计算 frequency 列
    for col in df.columns:
        if col.endswith('_frequency'):
            # 得到对应的 counts 列名称，假设 counts 列的名称为 "R#_counts"
            counts_col = col.replace('frequency', 'counts')
            # 计算该轮总的 counts，注意此处是对所有序列在该 round 的 counts 求和
            total_counts = df[counts_col].sum()
            # 更新频率：每个序列的 counts 占总 counts 的比例
            df[col] = df[counts_col] / total_counts

    # 保存结果到新的 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"处理后的数据已保存到 {output_file}")

if __name__ == "__main__": 
    process_csv(csv_path, output_path)
