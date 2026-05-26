"""示例 5：命名实体识别 (AutoModelForTokenClassification)

使用 BERT 模型识别文本中的人名、地名、组织名等实体。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

label_list = model.config.id2label

input_text = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."

inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
pred_labels = [label_list[prediction.item()] for prediction in predictions[0]]

for token, label in zip(tokens, pred_labels):
    print(f"{token}: {label}")
