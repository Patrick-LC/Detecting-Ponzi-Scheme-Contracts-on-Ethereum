import pandas as pd

dataset_path = '/Users/lc/Graduation Thesis/etherscan/transaction_dataset.csv'
data = pd.read_csv(dataset_path)

# 遍历数据集的每一行
for _, row in data.iterrows():
    # 遍历每一列（除了索引和Address列）
    for col in data.columns[2:]:
        try:
            # 尝试将值转换为浮点数
            float_value = float(row[col])
        except ValueError:
            # 转换失败，值无法转换为浮点数
            print(f"Column '{col}', value '{row[col]}' cannot be converted to float.")

