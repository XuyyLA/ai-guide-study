"""示例 3：序列到序列任务 (AutoModelForSeq2SeqLM)

使用 Helsinki-NLP 模型进行中英互译。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载英→中和中→英两个模型
en2zh_name = "Helsinki-NLP/opus-mt-en-zh"
zh2en_name = "Helsinki-NLP/opus-mt-zh-en"

print("正在加载英→中模型...")
en2zh_tokenizer = AutoTokenizer.from_pretrained(en2zh_name)
en2zh_model = AutoModelForSeq2SeqLM.from_pretrained(en2zh_name)

print("正在加载中→英模型...")
zh2en_tokenizer = AutoTokenizer.from_pretrained(zh2en_name)
zh2en_model = AutoModelForSeq2SeqLM.from_pretrained(zh2en_name)


def translate(text, direction="en2zh"):
    """翻译文本

    Args:
        text: 待翻译文本
        direction: "en2zh" 英译中 或 "zh2en" 中译英
    """
    if direction == "en2zh":
        tokenizer, model = en2zh_tokenizer, en2zh_model
    else:
        tokenizer, model = zh2en_tokenizer, zh2en_model

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 英译中
en_text = "Hello, how are you? I love learning about artificial intelligence."
zh_result = translate(en_text, "en2zh")
print(f"英→中: {en_text}\n      → {zh_result}\n")

# 中译英
zh_text = "今天天气真好，我想去公园散步。"
en_result = translate(zh_text, "zh2en")
print(f"中→英: {zh_text}\n      → {en_result}")
