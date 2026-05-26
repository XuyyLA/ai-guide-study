from config import Config

# 初始化OpenAI客户端
client = Config.get_client()

# 初始化对话历史
messages = [{"role": "system", "content": "You are a helpful assistant."}]

# 进行多轮对话，当前为3轮
for i in range(3):
    # 获取用户输入
    user_input = input("请输入：")

    # 添加用户消息到对话历史
    messages.append({"role": "user", "content": user_input})

    # 调用API获取模型回复
    response = client.chat.completions.create(
        model=Config.MODEL,
        messages=messages
    )

    # 提取模型回复内容
    assistant_output = response.choices[0].message.content

    # 将模型回复添加到对话历史
    messages.append({"role": "assistant", "content": assistant_output})

    print(f'用户输入：{user_input}')
    print(f'模型输出：{assistant_output}\n')
