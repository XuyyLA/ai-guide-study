import json
import gradio as gr
from config import Config

client = Config.get_client()


def _msg(m):
    """从 ChatMessage 对象或 dict 中提取 role 和 content"""
    if isinstance(m, dict):
        return m['role'], m['content']
    return m.role, m.content


CHATBOT_TASK = '小学数学老师（输入”开始”）'
PROMPT_FOR_TASK = "现在开始，你将扮演一个出小学数学题的老师，当我说开始时提供一个简单的数学题，接收到正确回答后进行下一题，否则给我答案"


def reset():
    """清空对话记录"""
    return []


def interact_customize(chatbot, prompt, user_input, temp=1.0):
    """调用模型处理定制化任务对话

    Args:
        chatbot: 历史对话记录
        prompt: 指定任务的提示词
        user_input: 当前用户输入
        temp: 模型温度参数（默认 1.0）

    Returns:
        更新后的对话记录
    """
    try:
        messages = []
        messages.append({'role': 'user', 'content': prompt})

        for m in chatbot:
            role, content = _msg(m)
            messages.append({'role': role, 'content': content})

        messages.append({'role': 'user', 'content': user_input})

        response = client.chat.completions.create(
            model=Config.MODEL,
            messages=messages,
            temperature=temp,
            max_tokens=200,
        )

        chatbot.append(gr.ChatMessage(role="user", content=user_input))
        chatbot.append(gr.ChatMessage(role="assistant", content=response.choices[0].message.content))

    except Exception as e:
        print(f"发生错误：{e}")
        chatbot.append(gr.ChatMessage(role="user", content=user_input))
        chatbot.append(gr.ChatMessage(role="assistant", content=f"抱歉，发生了错误：{e}"))

    return chatbot


def export_customized(chatbot, description):
    """导出定制化任务对话记录及任务描述到 JSON 文件

    Args:
        chatbot: 对话记录
        description: 任务描述
    """
    messages = [{"role": r, "content": c} for r, c in (_msg(m) for m in chatbot)]
    target = {"chatbot": messages, "description": description}
    with open("files/part3.json", "w", encoding="utf-8") as file:
        json.dump(target, file, ensure_ascii=False, indent=4)


with gr.Blocks() as demo:
    gr.Markdown("# 定制化任务\n聊天机器人可以执行某项任务，试着与它互动吧！")
    chatbot = gr.Chatbot()
    desc_textbox = gr.Textbox(label="任务描述", value=CHATBOT_TASK, interactive=False)
    prompt_textbox = gr.Textbox(label="提示词", value=PROMPT_FOR_TASK, visible=False)
    input_textbox = gr.Textbox(label="输入", value="")

    with gr.Column():
        gr.Markdown("# 温度调节\n温度越高响应越具创造性。")
        temperature_slider = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="温度")

    with gr.Row():
        send_button = gr.Button(value="发送")
        reset_button = gr.Button(value="重置")

    with gr.Column():
        gr.Markdown("# 保存结果\n点击导出按钮保存结果。")
        export_button = gr.Button(value="导出")

    send_button.click(interact_customize, inputs=[chatbot, prompt_textbox, input_textbox, temperature_slider], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_customized, inputs=[chatbot, desc_textbox])

demo.launch(debug=True)
