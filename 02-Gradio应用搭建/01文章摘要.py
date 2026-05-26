import json
import gradio as gr
from config import Config

client = Config.get_client()


def _msg(m):
    """从 ChatMessage 对象或 dict 中提取 role 和 content"""
    if isinstance(m, dict):
        return m['role'], m['content']
    return m.role, m.content


PROMPT_FOR_SUMMARIZATION = "请将以下文章概括成几句话。"


def reset():
    """清空对话记录"""
    return []


def interact_summarization(prompt, article, temp=1.0):
    """调用模型生成摘要

    Args:
        prompt: 用于摘要的提示词
        article: 需要摘要的文章内容
        temp: 模型温度，控制输出创造性（默认 1.0）

    Returns:
        对话记录，包含输入文本与模型输出
    """
    input_text = f"{prompt}\n{article}"

    response = client.chat.completions.create(
        model=Config.MODEL,
        messages=[{'role': 'user', 'content': input_text}],
        temperature=temp,
    )
    return [
        gr.ChatMessage(role="user", content=input_text),
        gr.ChatMessage(role="assistant", content=response.choices[0].message.content),
    ]


def export_summarization(chatbot, article):
    """导出摘要任务的对话记录和文章内容到 JSON 文件

    Args:
        chatbot: 模型对话记录
        article: 文章内容
    """
    messages = [{"role": r, "content": c} for r, c in (_msg(m) for m in chatbot)]
    target = {"chatbot": messages, "article": article}
    with open("files/part1.json", "w", encoding="utf-8") as file:
        json.dump(target, file, ensure_ascii=False, indent=4)


with gr.Blocks() as demo:
    gr.Markdown("# 文章摘要\n填写任何你喜欢的文章，让聊天机器人为你总结！")
    chatbot = gr.Chatbot()
    prompt_textbox = gr.Textbox(label="提示词", value=PROMPT_FOR_SUMMARIZATION, visible=False)
    article_textbox = gr.Textbox(label="文章", interactive=True, value="填充")

    with gr.Column():
        gr.Markdown("# 温度调节\n温度越高，响应越具创造性。")
        temperature_slider = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="温度")

    with gr.Row():
        send_button = gr.Button(value="发送")
        reset_button = gr.Button(value="重置")

    with gr.Column():
        gr.Markdown("# 保存结果\n点击导出按钮保存结果。")
        export_button = gr.Button(value="导出")

    send_button.click(interact_summarization,
                      inputs=[prompt_textbox, article_textbox, temperature_slider],
                      outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_summarization, inputs=[chatbot, article_textbox])

demo.launch(debug=True)
