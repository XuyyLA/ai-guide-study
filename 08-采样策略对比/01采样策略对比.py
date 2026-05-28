"""采样策略对比：Top-K vs Top-P 采样

对比不同采样策略对生成文本的影响：
- 贪心搜索（num_beams=1, do_sample=False）
- Top-K 采样：只从概率最高的 K 个 token 中采样
- Top-P（核采样）：从累积概率达到 P 的最小 token 集合中采样
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

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

input_text = "The future of artificial intelligence is"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
attention_mask = torch.ones_like(inputs).to(device)

# ── 1. 贪心搜索（确定性输出） ──────────────────────────────
print("=" * 60)
print("1. 贪心搜索（每步选概率最高的 token）")
print("=" * 60)
with torch.no_grad():
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ── 2. Top-K 采样 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Top-K 采样（只从概率最高的 K 个 token 中采样）")
print("=" * 60)
top_k_values = [10, 50, 100]

for top_k in top_k_values:
    print(f"\n--- top_k = {top_k} ---")
    for i in range(3):
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=60,
                do_sample=True,
                top_k=top_k,
                top_p=1.0,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"  生成 {i+1}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# ── 3. Top-P（核采样） ────────────────────────────────────
print("\n" + "=" * 60)
print("3. Top-P 核采样（从累积概率达到 P 的最小集合中采样）")
print("=" * 60)
top_p_values = [0.5, 0.9, 0.95]

for top_p in top_p_values:
    print(f"\n--- top_p = {top_p} ---")
    for i in range(3):
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=60,
                do_sample=True,
                top_k=0,
                top_p=top_p,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"  生成 {i+1}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# ── 4. Top-K + Top-P 组合 ─────────────────────────────────
print("\n" + "=" * 60)
print("4. Top-K + Top-P 组合（先 Top-K 过滤，再 Top-P 过滤）")
print("=" * 60)
combos = [(50, 0.9), (50, 0.95), (100, 0.9)]

for top_k, top_p in combos:
    print(f"\n--- top_k={top_k}, top_p={top_p} ---")
    for i in range(3):
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=60,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"  生成 {i+1}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
