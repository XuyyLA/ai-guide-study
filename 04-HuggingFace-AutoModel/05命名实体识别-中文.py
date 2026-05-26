"""示例 5：命名实体识别 - 中文版 (AutoModelForTokenClassification)

使用中文 BERT 模型识别人名、地名、组织名等实体。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "uer/roberta-base-finetuned-cluener2020-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

label_list = model.config.id2label

input_text = "张三在北京大学计算机系学习，他计划去上海的人工智能公司工作。"

inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
pred_labels = [label_list[prediction.item()] for prediction in predictions[0]]

for token, label in zip(tokens, pred_labels):
    print(f"{token}: {label}")
