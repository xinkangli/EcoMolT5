from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 下载模型和 tokenizer
model_name = "haitengzhao/gimlet"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存模型和 tokenizer 到本地
model.save_pretrained("./gimlet_model")
tokenizer.save_pretrained("./gimlet_model")
