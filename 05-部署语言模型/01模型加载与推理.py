"""示例 9：部署语言模型 — 模型加载与推理

使用 distilgpt2（GPT-2 蒸馏版，约 8820 万参数）进行文本生成推理。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 自动选择设备：GPU > MPS (Apple Silicon) > CPU
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
model.to(device)
print(f"模型已加载到设备: {device}")

model.eval()

input_text = "你好，你是谁？"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型生成的文本：")
print(generated_text)
