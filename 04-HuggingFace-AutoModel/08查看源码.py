"""查看 AutoModel 源码

使用 inspect 库查看 AutoModelForQuestionAnswering 的 __init__ 和 forward 方法。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import inspect
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

print("=" * 60)
print("__init__ 方法:")
print("=" * 60)
init_code = inspect.getsource(model.__init__)
print(init_code)

print("=" * 60)
print("forward 方法:")
print("=" * 60)
forward_code = inspect.getsource(model.forward)
print(forward_code)
