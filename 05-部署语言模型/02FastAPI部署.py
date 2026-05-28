"""示例 9：部署语言模型 — FastAPI 服务

使用 FastAPI 将 distilgpt2 封装为 API 服务。

运行方式：
    uvicorn 09部署语言模型-fastapi:app --host 0.0.0.0 --port 8000

交互文档：http://localhost:8000/docs
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class PromptRequest(BaseModel):
    prompt: str


app = FastAPI()

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.post("/generate")
def generate_text(request: PromptRequest):
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
