"""微调后推理：加载 LoRA checkpoint，测试写唐诗能力

有 GPU 时使用 4-bit 量化加载，无 GPU 时使用 float16 加载。
需要先运行 02微调训练.py 生成 checkpoint。
数据集准备：git clone https://github.com/CheeEn-Yu/GenAI-Hw5.git
"""
import os
import sys

if sys.platform == 'win32' and not getattr(sys, 'flags', None).utf8_mode:
    os.execv(sys.executable, [sys.executable, '-X', 'utf8'] + sys.argv)

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import json
import gc
import logging
import warnings
warnings.filterwarnings("ignore")

import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)

# ── 模型与路径 ────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "你是一个乐于助人的助手且擅长写唐诗。"
ckpt_dir = "./exp1"
output_dir = "./output"
cache_dir = "./cache"

# ── 查找 checkpoint ──────────────────────────────────────
ckpts = []
for ckpt in os.listdir(ckpt_dir):
    if ckpt.startswith("checkpoint-"):
        ckpts.append(ckpt)

ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[-1]))
print("所有可用的 checkpoints：")
print(" id: checkpoint 名称")
for (i, ckpt) in enumerate(ckpts):
    print(f"{i:>3}: {ckpt}")

# 使用最后一个 checkpoint
id_of_ckpt_to_use = -1
ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])
print(f"\n使用 checkpoint: {ckpt_name}")

# ── 解码参数 ──────────────────────────────────────────────
max_len = 128
temperature = 0.1
top_p = 0.3
no_repeat_ngram_size = 3

# ── 加载模型 ──────────────────────────────────────────────
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=nf4_config
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        device_map={'': 0},
        cache_dir=cache_dir
    )

    model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})
else:
    print("未检测到 CUDA，使用 float16 加载")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(model, ckpt_name)
    model.to(device)

# ── 评估函数 ──────────────────────────────────────────────
def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n{input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )

    for s in generation_output.sequences:
        output = tokenizer.decode(s[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if verbose:
            print(output)

    return output


# ── 生成测试结果 ──────────────────────────────────────────
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    no_repeat_ngram_size=no_repeat_ngram_size,
    pad_token_id=tokenizer.eos_token_id
)

test_data_path = "GenAI-Hw5/Tang_testing_data.json"
output_path = os.path.join(output_dir, "results.txt")

with open(test_data_path, "r", encoding="utf-8") as f:
    test_datas = json.load(f)

with open(output_path, "w", encoding="utf-8") as f:
    for (i, test_data) in enumerate(test_datas):
        predict = evaluate(test_data["instruction"], generation_config, max_len, test_data["input"], verbose=False)
        f.write(f"{i+1}. " + test_data["input"] + predict + "\n")
        print(f"{i+1}. " + test_data["input"] + predict)

# ── 微调前后对比 ──────────────────────────────────────────
print("\n" + "=" * 80)
print("微调后对比测试：")
print("=" * 80)

test_tang_list = [
    '相见时难别亦难，东风无力百花残。',
    '重帷深下莫愁堂，卧后清宵细细长。',
    '芳辰追逸趣，禁苑信多奇。'
]

demo_after_finetune = []
for tang in test_tang_list:
    demo_after_finetune.append(
        f'模型输入:\n以下是一首唐诗的第一句话，请用你的知识判断并完成整首诗。{tang}\n\n模型输出:\n' +
        evaluate('以下是一首唐诗的第一句话，请用你的知识判断并完成整首诗。', generation_config, max_len, tang, verbose=False)
    )

for idx in range(len(demo_after_finetune)):
    print(f"Example {idx + 1}:")
    print(demo_after_finetune[idx])
    print("-" * 80)
