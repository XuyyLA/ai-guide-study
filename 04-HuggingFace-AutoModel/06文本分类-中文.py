"""示例 6：文本分类 - 中文版 (AutoModelForSequenceClassification)

使用中文模型进行情感分析。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "uer/roberta-base-finetuned-jd-binary-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

examples = [
    "这个产品质量很好，非常满意！",
    "太差了，完全不能用，浪费钱。",
    "一般般吧，没有想象中那么好。",
]

labels = ['Negative', 'Positive']

for text in examples:
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    predicted_label = labels[prediction]
    confidence = probabilities[0][prediction].item()

    print(f"文本: {text}")
    print(f"情感: {predicted_label} (置信度: {confidence:.2%})\n")
