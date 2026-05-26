"""示例 2：填空任务 - 中文版 (AutoModelForMaskedLM)

使用中文 BERT 模型预测被遮蔽的词。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 中文填空示例
examples = [
    "中国的首都是[MASK]。",
    "李白是唐朝著名的[MASK]人。",
    "地球围绕[MASK]转。",
]

for text in examples:
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    masked_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # 取 top-3 预测
    top_k = 3
    top_k_ids = predictions[0, masked_index].topk(top_k, dim=-1).indices[0]
    top_k_tokens = [tokenizer.decode([tid.item()]) for tid in top_k_ids]

    print(f"输入: {text}")
    print(f"预测 top-{top_k}: {', '.join(top_k_tokens)}\n")
