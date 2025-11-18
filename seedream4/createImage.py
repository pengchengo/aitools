import os
import re
import requests
from pathlib import Path
from datetime import datetime
# Install SDK:  pip install 'volcengine-python-sdk[ark]' .
from volcenginesdkarkruntime import Ark 
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions

def sanitize_filename(text, max_length=80):
    """
    清理文本，使其适合作为文件名
    移除或替换Windows文件名不允许的字符
    """
    # Windows文件名不允许的字符: < > : " / \ | ? *
    # 替换为下划线
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    # 将逗号、句号等标点符号替换为下划线，使文件名更清晰
    text = re.sub(r'[，。、；：！？]', '_', text)
    # 将多个连续的下划线替换为单个下划线
    text = re.sub(r'_+', '_', text)
    # 移除前后空格、点和下划线
    text = text.strip('. _')
    # 限制长度
    if len(text) > max_length:
        text = text[:max_length]
    # 如果为空，使用默认名称
    if not text:
        text = "image"
    return text

client = Ark(
    # The base URL for model invocation .
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # Get API Key：https://console.volcengine.com/ark/region:ark+cn-beijing/apikey
    # 方式1：从环境变量读取（推荐，更安全）
    # 先设置环境变量：export ARK_API_KEY='your-api-key' (Linux/Mac) 或 set ARK_API_KEY=your-api-key (Windows)
    api_key=os.getenv('ARK_API_KEY')
)

# 提示词
prompt = "手绘卡通,2d,天空之城,室内布局,50-60个物品,画面丰富紧凑,色彩丰富,家具造型优美有特色,比例 16:9。没有人物。颜色靓丽,没有人物。"

# 生成文件名基础（从提示词提取）
filename_base = sanitize_filename(prompt)
 
imagesResponse = client.images.generate( 
    # Replace with Model ID .
    model="doubao-seedream-4-0-250828", 
    prompt=prompt,
    image=["https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimages_1.png", "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimages_2.png"],
    size="2K",
    sequential_image_generation="auto",
    sequential_image_generation_options=SequentialImageGenerationOptions(max_images=3),
    response_format="url",
    watermark=True
) 
 
# 创建images文件夹（如果不存在）
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# 遍历所有图片数据
for idx, image in enumerate(imagesResponse.data, 1):
    # 输出当前图片的url和size
    print(f"图片 {idx}: URL: {image.url}, Size: {image.size}")
    
    try:
        # 下载图片
        response = requests.get(image.url, timeout=30)
        response.raise_for_status()
        
        # 生成文件名（使用提示词内容和序号）
        filename = f"{filename_base}_{idx}.png"
        filepath = images_dir / filename
        
        # 保存图片
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"  ✓ 已保存到: {filepath}")
        
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")

print(f"\n所有图片已处理完成，保存在 {images_dir.absolute()} 目录下")