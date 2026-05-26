"""示例 4：问答系统 - 中文版 (AutoModelForQuestionAnswering)

使用中文预训练模型进行抽取式问答。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 使用中文 BERT 在 CMRC2018 数据集上微调的问答模型
model_name = "uer/roberta-base-chinese-extractive-qa"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 中文问答示例
examples = [
    {
        "context": "故宫位于北京市中心，是中国明清两代的皇家宫殿，旧称紫禁城。故宫占地面积72万平方米，建筑面积约15万平方米，有大小宫殿七十多座，房屋九千余间。",
        "question": "故宫的旧称是什么？",
    },
    {
        "context": "深度学习是机器学习的一个分支，它试图使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。深度学习的核心是神经网络。",
        "question": "深度学习的核心是什么？",
    },
    {
        "context": "李白，字太白，号青莲居士，是唐代伟大的浪漫主义诗人，被后人誉为诗仙。与杜甫并称为李杜，其人爽朗大方，爱饮酒作诗，喜交友。",
        "question": "李白被后人誉为什么？",
    },
]

for i, ex in enumerate(examples, 1):
    inputs = tokenizer(ex["question"], ex["context"], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    print(f"示例 {i}:")
    print(f"  问题: {ex['question']}")
    print(f"  答案: {answer}\n")
