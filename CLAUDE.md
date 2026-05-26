# AI Guide Study

## 项目概述

学习 AI 开发相关内容的代码练习项目，包含 LLM API 调用、Gradio 应用搭建、Prompt 优化、HuggingFace 模型使用等。

## 网络代理

本机代理端口为 **7897**，HF 缓存路径为 **C:/hf_cache**（避免中文路径导致 sentencepiece 报错）：

```python
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'C:/hf_cache'
```

所有需要访问外网的 Python 脚本都应在导入第三方库前设置这三个环境变量。

## 目录结构

- `01-LLM基础对话/` - 智谱 API 基础调用（单轮、多轮、流式）
- `02-Gradio应用搭建/` - Gradio 构建 AI 应用（摘要、角色扮演、定制化任务）
- `03-Prompt优化解题/` - 自定义 Prompt 评估数学题准确率
- `04-HuggingFace-AutoModel/` - HuggingFace AutoModel 各任务示例
- `学习归档/` - 每次对话的详细记录，按日期命名

## Python 环境

使用 Python 3.11（`C:/Users/共享/AppData/Local/Programs/Python/Python311/python.exe`），已在 `.vscode/settings.json` 中配置。
