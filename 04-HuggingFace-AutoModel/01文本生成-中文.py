"""示例 1：文本生成 - 中文版 (AutoModelForCausalLM)

使用中文 GPT-2 模型进行文本生成。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "uer/gpt2-chinese-cluecorpussmall"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "人工智能的未来发展方向是"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.95, temperature=0.7)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
