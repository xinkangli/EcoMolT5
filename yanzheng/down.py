from datasets import load_dataset

# 以 'bace' 任务为例
dataset = load_dataset("haitengzhao/molecule_property_instruction", split="bace")

# 查看前5个样本
for i in range(5):
    print(dataset[i])
