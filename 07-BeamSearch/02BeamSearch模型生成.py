"""Beam Search 模型生成：使用 distilgpt2 对比不同束宽的输出

束宽越大，生成文本越稳定但多样性越低。
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

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
model.to(device)
model.eval()

input_text = "Hello GPT"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
attention_mask = torch.ones_like(inputs).to(device)

# ── 对比不同束宽 ──────────────────────────────────────────
beam_widths = [1, 3, 5]

for beam_width in beam_widths:
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=beam_width,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"束宽 {beam_width} 的生成结果：")
    print(generated_text)
    print('-' * 50)
