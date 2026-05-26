"""示例 6：文本分类 (AutoModelForSequenceClassification)

使用 DistilBERT 模型进行情感分析。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

input_text = "I love using transformers library!"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probabilities = F.softmax(logits, dim=1)

labels = ['Negative', 'Positive']
prediction = torch.argmax(probabilities, dim=1)
predicted_label = labels[prediction]

print(f"文本: {input_text}")
print(f"情感预测: {predicted_label}")
