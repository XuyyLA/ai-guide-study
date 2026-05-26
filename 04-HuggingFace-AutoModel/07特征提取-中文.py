"""示例 7：特征提取 - 中文版 (AutoModel)

使用中文 BERT 模型提取文本的隐藏状态向量。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

input_text = "深度学习是人工智能的核心技术。"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(f"输入文本: {input_text}")
print(f"Last hidden state shape: {last_hidden_states.shape}")
print(f"向量维度: {last_hidden_states.shape[-1]}，每个 token 对应一个 {last_hidden_states.shape[-1]} 维向量")
