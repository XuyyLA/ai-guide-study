import json
import gradio as gr
from config import Config

client = Config.get_client()


def _msg(m):
    """从 ChatMessage 对象或 dict 中提取 role 和 content"""
    if isinstance(m, dict):
        return m['role'], m['content']
    return m.role, m.content


CHARACTER_FOR_CHATBOT = "面试官"
PROMPT_FOR_ROLEPLAY = "我需要你面试我有关AI的知识，仅提出问题"


def reset():
    """清空对话记录"""
    return []


def interact_roleplay(chatbot, user_input, temp=1.0):
    """处理角色扮演多轮对话，调用模型生成回复

    Args:
        chatbot: 对话历史记录（用户与模型回复）
        user_input: 当前用户输入
        temp: 模型温度参数（默认 1.0）

    Returns:
        更新后的对话记录
    """
    try:
        messages = []
        for m in chatbot:
            role, content = _msg(m)
            messages.append({'role': role, 'content': content})

        messages.append({'role': 'user', 'content': user_input})

        response = client.chat.completions.create(
            model=Config.MODEL,
            messages=messages,
            temperature=temp,
        )
        chatbot.append(gr.ChatMessage(role="user", content=user_input))
        chatbot.append(gr.ChatMessage(role="assistant", content=response.choices[0].message.content))

    except Exception as e:
        print(f"发生错误：{e}")
        chatbot.append(gr.ChatMessage(role="user", content=user_input))
        chatbot.append(gr.ChatMessage(role="assistant", content=f"抱歉，发生了错误：{e}"))

    return chatbot


def export_roleplay(chatbot, description):
    """导出角色扮演对话记录及任务描述到 JSON 文件

    Args:
        chatbot: 对话记录
        description: 任务描述
    """
    messages = [{"role": r, "content": c} for r, c in (_msg(m) for m in chatbot)]
    target = {"chatbot": messages, "description": description}
    with open("files/part2.json", "w", encoding="utf-8") as file:
        json.dump(target, file, ensure_ascii=False, indent=4)


first_dialogue = interact_roleplay([], PROMPT_FOR_ROLEPLAY)

with gr.Blocks() as demo:
    gr.Markdown("# 角色扮演\n与聊天机器人进行角色扮演互动！")
    chatbot = gr.Chatbot(value=first_dialogue)
    description_textbox = gr.Textbox(label="机器人扮演的角色", interactive=False, value=CHARACTER_FOR_CHATBOT)
    input_textbox = gr.Textbox(label="输入", value="")

    with gr.Column():
        gr.Markdown("# 温度调节\n温度控制聊天机器人的响应创造性。")
        temperature_slider = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="温度")

    with gr.Row():
        send_button = gr.Button(value="发送")
        reset_button = gr.Button(value="重置")

    with gr.Column():
        gr.Markdown("# 保存结果\n点击导出按钮保存对话记录。")
        export_button = gr.Button(value="导出")

    send_button.click(interact_roleplay, inputs=[chatbot, input_textbox, temperature_slider], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_roleplay, inputs=[chatbot, description_textbox])

demo.launch(debug=True)
