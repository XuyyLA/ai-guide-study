from config import Config

# 初始化 OpenAI 客户端
client = Config.get_client()

# 调用 API 获取模型回复
response = client.chat.completions.create(
    model=Config.MODEL,
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}]
    )

# 打印模型回复内容
print(response.choices[0].message.content)
