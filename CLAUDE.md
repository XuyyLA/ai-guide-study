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
- `05-部署语言模型/` - 本地部署 distilgpt2 并封装为 API 服务（FastAPI / Flask）
- `06-微调LLM写唐诗/` - LoRA 微调 Qwen2.5-0.5B-Instruct 让它学会写唐诗（4-bit 量化 + SFTTrainer）**⚠️ 待GPU环境运行**
- `07-BeamSearch/` - Beam Search 算法原理演示 + distilgpt2 不同束宽对比
- `学习归档/` - 每次对话的详细记录，按日期命名

## 项目目录与 Guide 文档对应关系

项目文件夹编号与参考文档（`D:\LA\资料\AI-Guide-and-Demos-zh_CN\Guide\`）的对应：

| 项目目录 | Guide 文档 | 说明 |
|---------|-----------|------|
| `01-LLM基础对话/` | `01. 初识 LLM API：环境配置与多轮对话演示.md` | 智谱 API 调用 |
| `02-Gradio应用搭建/` | `02. 简单入门：通过 API 与 Gradio 构建 AI 应用.md` | Gradio 应用 |
| `03-Prompt优化解题/` | `03. 进阶指南：自定义 Prompt 提升大模型解题能力.md` | Prompt 优化 |
| `04-HuggingFace-AutoModel/` | `05. 理解 Hugging Face 的 AutoModel 系列.md` | AutoModel 各任务 |
| `05-部署语言模型/` | `06. 开始实践：部署你的第一个语言模型.md` | 模型部署 |
| `06-微调LLM写唐诗/` | `08. 尝试微调 LLM：让它会写唐诗.md` | LoRA 微调 ⚠️ 待GPU |
| `07-BeamSearch/` | `09. 深入理解 Beam Search.md` | Beam Search 原理与对比 |

> 注意：Guide 文档编号 04（LoRA 认识）、07（模型参数与显存）未单独建目录，项目编号与 Guide 编号不是 1:1 对应。

## 模型选择规则

后续涉及开源 LLM 的练习统一使用 **Qwen** 系列模型，优先级：Qwen > DeepSeek > GLM > 其他。
- 默认模型：`Qwen/Qwen2.5-0.5B-Instruct`（CPU 可跑，7B 需 GPU）
- Prompt 格式：使用 Qwen 的 ChatML 格式（`tokenizer.apply_chat_template`），不使用 Llama 的 `[INST]` 格式
- 输出语言：简体中文

## Python 环境

使用 Python 3.11（`C:/Users/共享/AppData/Local/Programs/Python/Python311/python.exe`），已在 `.vscode/settings.json` 中配置。
