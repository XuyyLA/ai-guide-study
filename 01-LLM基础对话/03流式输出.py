from config import Config

# 初始化 OpenAI 客户端
client = Config.get_client()

# 开启流式输出
response = client.chat.completions.create(
    model=Config.MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. 你叫 Luxa Bot 。回答前称呼我为Luxa"},
        {"role": "user", "content": "你是谁？"}
    ],
    stream=True,
)

# 实时打印模型回复的增量内容
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
