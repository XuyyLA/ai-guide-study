"""微调前推理：加载 Qwen2.5-0.5B-Instruct 模型，测试写唐诗能力

有 GPU 时使用 4-bit 量化加载，无 GPU 时使用 float16 加载。
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
import logging
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)

# ── 模型选择 ──────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# ── 加载模型 ──────────────────────────────────────────────
cache_dir = "./cache"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )
else:
    print("未检测到 CUDA，使用 float16 加载")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.to(device)

logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

# ── 推理参数 ──────────────────────────────────────────────
max_len = 128
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
)


# ── 评估函数 ──────────────────────────────────────────────
SYSTEM_PROMPT = "你是一个乐于助人的助手且擅长写唐诗。"

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


# ── 微调前测试 ────────────────────────────────────────────
test_tang_list = [
    '相见时难别亦难，东风无力百花残。',
    '重帷深下莫愁堂，卧后清宵细细长。',
    '芳辰追逸趣，禁苑信多奇。'
]

demo_before_finetune = []
for tang in test_tang_list:
    demo_before_finetune.append(
        f'模型输入:\n以下是一首唐诗的第一句话，请用你的知识判断并完成整首诗。{tang}\n\n模型输出:\n' +
        evaluate('以下是一首唐诗的第一句话，请用你的知识判断并完成整首诗。', generation_config, max_len, tang, verbose=False)
    )

for idx in range(len(demo_before_finetune)):
    print(f"Example {idx + 1}:")
    print(demo_before_finetune[idx])
    print("-" * 80)
