from transformers import AutoModelForSeq2SeqLM, AutoConfig
import os
def check_model(model_path):
    try:
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"模型路径 {model_path} 不存在！")
            return
        
        # 检查是否有模型文件（例如 pytorch_model.bin）和配置文件
        model_file = os.path.join(model_path, 'pytorch_model.bin')
        config_file = os.path.join(model_path, 'config.json')
        
        if not os.path.isfile(model_file):
            print(f"模型文件 {model_file} 不存在！")
            return
        
        if not os.path.isfile(config_file):
            print(f"配置文件 {config_file} 不存在！")
            return
        
        # 尝试加载配置
        config = AutoConfig.from_pretrained(model_path)
        print(f"成功加载模型配置：{config}")
        
        # 使用 AutoModelForSeq2SeqLM 来加载 T5 模型
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)
        print("模型加载成功！")
        
    except Exception as e:
        print(f"加载模型时出现错误：{e}")

# 检查模型加载
check_model('./gimlet_model/')
