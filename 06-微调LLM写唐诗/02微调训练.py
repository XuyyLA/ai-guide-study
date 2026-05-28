"""LoRA 微调 Qwen2.5-0.5B-Instruct：让它学会写唐诗

有 GPU 时使用 4-bit 量化 + LoRA + bf16，无 GPU 时使用 float16 + LoRA。
数据集准备：git clone https://github.com/CheeEn-Yu/GenAI-Hw5.git
"""
import os
import sys

# Windows 中文系统需 UTF-8 模式，否则 trl 读取 jinja 模板报 GBK 解码错误
if sys.platform == 'win32' and not getattr(sys, 'flags', None).utf8_mode:
    os.execv(sys.executable, [sys.executable, '-X', 'utf8'] + sys.argv)

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import sys
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ── 模型选择 ──────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "你是一个乐于助人的助手且擅长写唐诗。"

# ── 超参数（建议尝试调整）──────────────────────────────────
num_train_data = 1040   # 训练数据量，最大 5000
num_epoch = 1           # 训练 Epoch 数
LEARNING_RATE = 3e-4    # 学习率

# ── 超参数（一般不需要调整）────────────────────────────────
output_dir = "./output"
ckpt_dir = "./exp1"
cache_dir = "./cache"
from_ckpt = False
ckpt_name = None
dataset_dir = "./GenAI-Hw5/Tang_training_data.json"
logging_steps = 20
save_steps = 65
save_total_limit = 3
report_to = "none"
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

# ── 加载模型 ──────────────────────────────────────────────
use_cuda = torch.cuda.is_available()

if use_cuda:
    from peft import prepare_model_for_kbit_training
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
    model = prepare_model_for_kbit_training(model)
    use_bf16 = True
else:
    print("未检测到 CUDA，使用 float16 加载 + LoRA（CPU 训练较慢）")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    use_bf16 = False

logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)


# ── 数据预处理函数 ────────────────────────────────────────
def generate_training_data(data_point):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{data_point['instruction']}\n{data_point['input']}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    full_text = prompt + data_point["output"]
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=CUTOFF_LEN,
    )

    full_len = len(full["input_ids"])
    completion_mask = [0] * prompt_len + [1] * (full_len - prompt_len)

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "completion_mask": completion_mask,
    }


# ── 开始微调 ──────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

if from_ckpt:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ckpt_name)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

tokenizer.pad_token_id = 0

# 加载训练数据
with open(dataset_dir, "r", encoding="utf-8") as f:
    data_json = json.load(f)
with open("tmp_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data_json[:num_train_data], f, indent=2, ensure_ascii=False)

data = load_dataset('json', data_files="tmp_dataset.json", download_mode="force_redownload")

if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_training_data)
    val_data = train_val["test"].shuffle().map(generate_training_data)
else:
    train_data = data['train'].shuffle().map(generate_training_data)
    val_data = None

# SFTTrainer 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=SFTConfig(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        bf16=use_bf16,
        fp16=not use_bf16 and use_cuda,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=ckpt_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,
        report_to=report_to,
    ),
    data_collator=DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, completion_only_loss=True),
)

model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

trainer.train()
model.save_pretrained(ckpt_dir)

print("\n 如果上方有关于缺少键的警告，请忽略 :)")
