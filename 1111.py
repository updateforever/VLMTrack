
import os
from openai import OpenAI
client = OpenAI(
    api_key="sk-qRbXAVvchaiWDVqh4xHrHfc3OP65VUx1fI7kfBkv8FVYixut", #您的API-Key   
    base_url="http://10.128.202.100:3010/v1",
)
completion = client.chat.completions.create(
    model="deepseek-v3.1",  #  qwen3-coder-flash
    messages=[{'role': 'user', 'content': '你是谁？'}]
)
print(completion.choices[0].message.content)