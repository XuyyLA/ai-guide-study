"""示例 1：文本生成 (AutoModelForCausalLM)

使用 GPT-2 模型进行文本生成。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "今天周一，我八点起床去"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.95, temperature=0.7)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
