"""Temperature 对生成的影响

Temperature 控制概率分布的"尖锐程度"：
- temperature → 0：分布越尖锐，趋向贪心搜索（确定性输出）
- temperature = 1：原始概率分布
- temperature → ∞：分布越均匀，随机性越大

创意任务（写诗、故事）适合高 temperature，事实任务（翻译、摘要）适合低 temperature。
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

# ── 1. Temperature 对同一 prompt 的影响 ────────────────────
input_text = "In the year 2050, humanity will"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
attention_mask = torch.ones_like(inputs).to(device)

print("=" * 60)
print("Temperature 对比（同一 prompt，不同 temperature）")
print("=" * 60)

temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

for temp in temperatures:
    print(f"\n--- temperature = {temp} ---")
    for i in range(3):
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=60,
                do_sample=True,
                temperature=temp,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"  生成 {i+1}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# ── 2. 不同任务适合的 temperature ──────────────────────────
print("\n" + "=" * 60)
print("不同任务适合的 Temperature")
print("=" * 60)

tasks = [
    ("事实性任务（低 temperature）", "The capital of France is", 0.2),
    ("创意性任务（高 temperature）", "Once upon a time in a galaxy far away", 1.5),
    ("平衡任务（中 temperature）", "The meaning of life is", 0.7),
]

for task_name, prompt, temp in tasks:
    print(f"\n--- {task_name} (temperature={temp}) ---")
    task_inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    task_mask = torch.ones_like(task_inputs).to(device)
    for i in range(3):
        with torch.no_grad():
            outputs = model.generate(
                task_inputs,
                attention_mask=task_mask,
                max_length=60,
                do_sample=True,
                temperature=temp,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"  生成 {i+1}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# ── 3. 可视化：Temperature 如何改变概率分布 ────────────────
print("\n" + "=" * 60)
print("Temperature 对概率分布的影响（取前 10 个 token）")
print("=" * 60)

with torch.no_grad():
    logits = model(inputs).logits[0, -1, :]  # 最后一个位置的 logits

for temp in [0.1, 0.5, 1.0, 2.0]:
    scaled_logits = logits / temp
    probs = torch.softmax(scaled_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)

    print(f"\ntemperature = {temp}:")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx.item()])
        bar = "█" * int(prob.item() * 100)
        print(f"  {token:>6s}: {prob.item():.4f} {bar}")
