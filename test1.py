import logging
import os
from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,  # 使用 Seq2SeqLM 来加载 T5 模型
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_from_disk, DatasetDict
import torch
import faulthandler

# 启用 faulthandler 来捕获崩溃信息
faulthandler.enable()

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./gimlet_model/")
    tokenizer_name: str = field(default="./gimlet_model/")  # 修改为根文件夹路径
    transformer_backbone: str = field(default='')

@dataclass
class DataTrainingArguments:
    train_file: str = field(default="./gimlet_data/chembl_pretraining/")
    validation_file: str = field(default="./gimlet_data/chembl_zero_shot/")

def main():
    # 参数解析
    model_args = ModelArguments()
    data_args = DataTrainingArguments()

    set_seed(42)

    # 加载数据集
    raw_datasets = {}
    raw_datasets['train'] = load_from_disk(data_args.train_file)  # 使用 load_from_disk 加载训练集
    raw_datasets['validation'] = load_from_disk(data_args.validation_file)  # 使用 load_from_disk 加载验证集
    raw_datasets = DatasetDict(raw_datasets)

    # 加载tokenizer（使用本地文件夹路径加载）
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, local_files_only=True)
    print("Tokenizer 加载成功！")

    # 加载模型
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config)
    print("模型加载成功！")

    # 数据处理
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=4,  # 缩小 batch_size
        per_device_eval_batch_size=4,   # 缩小 batch_size
        num_train_epochs=3,
        logging_dir='./logs',
    )

    # 创建 Trainer 实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
