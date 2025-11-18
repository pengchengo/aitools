import os
# Install SDK:  pip install 'volcengine-python-sdk[ark]'
from volcenginesdkarkruntime import Ark 

client = Ark(
    # The base URL for model invocation
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # Get API Key：https://console.volcengine.com/ark/region:ark+cn-beijing/apikey
    api_key=os.getenv('ARK_API_KEY'), 
)

num = 100
createWordTip = "我在做一个收纳游戏，帮我生成{1}个主题，例如：西双版纳，猎人的家等等，用逗号隔开，中文"
completion = client.chat.completions.create(
    # Replace with Model ID
    model = "kimi-k2-250905",
    messages = [
        {"role": "user", "content": createWordTip},
    ],
)

tipWordContent = completion.choices[0].message.content
tipWordContent.replace("，", ",")
tipWordList = tipWordContent.split(",")

print(tipWordContent)
