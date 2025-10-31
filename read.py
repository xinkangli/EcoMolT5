import pandas as pd

# 指定合并后的CSV文件路径
file_path = 'pretrain_datasets/merge_split0.csv'

# 加载CSV文件
df = pd.read_csv(file_path)

# 打印出前三条记录
print("First 3 rows of the merged CSV file:")
print(df.head(3))
