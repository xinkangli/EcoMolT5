from datasets import load_from_disk

def check_datasets(train_file, test_file):
    try:
        # 加载训练集和测试集
        train_dataset = load_from_disk(train_file)
        test_dataset = load_from_disk(test_file)
        
        print("训练集加载成功！")
        print("测试集加载成功！")
        
        # 查看训练集内容
        print("训练集内容预览：")
        print(train_dataset)
        
        # 查看测试集内容
        print("测试集内容预览：")
        print(test_dataset)
        
        # 查看训练集的列信息
        print("训练集列信息：")
        print(train_dataset.features)
        
        # 查看测试集的列信息
        print("测试集列信息：")
        print(test_dataset.features)
        
        # 如果需要，查看训练集前5条数据
        print("训练集前5条数据：")
        print(train_dataset[:5])
        
        # 查看测试集前5条数据
        print("测试集前5条数据：")
        print(test_dataset[:5])
        
    except Exception as e:
        print(f"加载数据集时出现错误：{e}")

# 加载并检查训练集和测试集
check_datasets('./gimlet_data/chembl_pretraining/', './gimlet_data/chembl_zero_shot/')
