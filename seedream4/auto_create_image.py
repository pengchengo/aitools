"""
整合脚本：根据文本提示词生成主题，然后为每个主题生成图片
"""

import os
import json
import re
import requests
import random
import time
from pathlib import Path
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions


def load_config(config_path="config.json"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_filename(text, max_length=80):
    """
    清理文本，使其适合作为文件名
    移除或替换Windows文件名不允许的字符
    """
    # Windows文件名不允许的字符: < > : " / \ | ? *
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    # 将逗号、句号等标点符号替换为下划线
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


def generate_themes(client, config):
    """
    使用文本模型生成主题列表
    """
    image_count = config['image_count']
    text_template = config['text_prompt_template']
    
    # 替换模板中的{1}为图片数量
    text_prompt = text_template.replace('{1}', str(image_count))
    
    print(f"正在生成 {image_count} 个主题...")
    print(f"提示词: {text_prompt}\n")
    
    try:
        completion = client.chat.completions.create(
            model=config['text_model'],
            messages=[
                {"role": "user", "content": text_prompt},
            ],
        )
        
        tip_word_content = completion.choices[0].message.content
        print(f"模型返回: {tip_word_content}\n")
        
        # 清理文本：将中文逗号替换为英文逗号
        tip_word_content = tip_word_content.replace("，", ",")
        # 分割主题
        theme_list = [theme.strip() for theme in tip_word_content.split(",") if theme.strip()]
        
        print(f"解析到 {len(theme_list)} 个主题:")
        for i, theme in enumerate(theme_list, 1):
            print(f"  {i}. {theme}")
        print()
        
        return theme_list
        
    except Exception as e:
        print(f"生成主题时出错: {e}")
        return []


def generate_images_for_theme(client, config, theme, theme_index, total_themes):
    """
    为单个主题生成图片
    """
    image_template = config['image_prompt_template']
    
    # 替换模板中的{1}为主题
    image_prompt = image_template.replace('{1}', theme)
    
    print(f"[{theme_index}/{total_themes}] 正在为主题 '{theme}' 生成图片...")
    print(f"  提示词: {image_prompt}")
    
    try:
        images_response = client.images.generate(
            model=config['image_model'],
            prompt=image_prompt,
            image=config.get('reference_images', []),
            size=config.get('image_size', '2K'),
            sequential_image_generation="auto",
            sequential_image_generation_options=SequentialImageGenerationOptions(
                max_images=config.get('max_images_per_prompt', 3)
            ),
            response_format="url",
            watermark=config.get('watermark', True)
        )
        
        return images_response.data
        
    except Exception as e:
        print(f"  ✗ 生成图片失败: {e}")
        return []


def save_image(image_url, filepath, max_retries=5, retry_delay=3):
    """
    下载并保存图片（带重试机制）
    
    Args:
        image_url: 图片URL
        filepath: 保存路径
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    """
    for attempt in range(1, max_retries + 1):
        try:
            # 使用更长的超时时间和流式下载
            response = requests.get(
                image_url, 
                timeout=(10, 60),  # (连接超时, 读取超时)
                stream=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            response.raise_for_status()
            
            # 流式写入文件，避免内存问题
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"  ⚠ 下载超时，{retry_delay}秒后重试 ({attempt}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"  ✗ 下载失败: 超时（已重试{max_retries}次）")
                return False
                
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                print(f"  ⚠ 连接中断，{retry_delay}秒后重试 ({attempt}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"  ✗ 下载失败: 连接错误（已重试{max_retries}次）- {e}")
                return False
                
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⚠ 下载出错，{retry_delay}秒后重试 ({attempt}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                print(f"  ✗ 下载失败: {e}（已重试{max_retries}次）")
                return False
    
    return False


def main():
    """主函数"""
    # 加载配置
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    config = load_config(config_path)
    
    # 初始化客户端
    api_key = os.getenv('ARK_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量 ARK_API_KEY")
        print("Windows: set ARK_API_KEY=your-api-key")
        print("Linux/Mac: export ARK_API_KEY='your-api-key'")
        return
    
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key
    )
    
    # 创建images文件夹
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    # 步骤1: 生成主题列表
    themes = generate_themes(client, config)
    
    if not themes:
        print("未能生成主题，程序退出")
        return
    
    #打乱主题列表
    random.shuffle(themes)
    # 如果生成的主题数量超过配置的图片数量，只取前N个
    if len(themes) > config['image_count']:
        themes = themes[:config['image_count']]
        print(f"主题数量超过配置，只使用前 {config['image_count']} 个主题\n")
    
    # 步骤2: 为每个主题生成图片
    total_saved = 0
    total_failed = 0
    
    for theme_index, theme in enumerate(themes, 1):
        # 生成图片
        images_data = generate_images_for_theme(
            client, config, theme, theme_index, len(themes)
        )
        
        if not images_data:
            total_failed += 1
            # 即使失败也等待一下，避免请求过快
            time.sleep(1)
            continue
        
        # 保存图片
        for img_idx, image in enumerate(images_data, 1):
            # 生成文件名（使用主题名称）
            filename_base = sanitize_filename(theme)
            filename = f"{filename_base}_{img_idx}.png"
            filepath = images_dir / filename
            
            # 下载并保存
            if save_image(image.url, filepath):
                print(f"  ✓ 已保存: {filepath.name}")
                total_saved += 1
            else:
                total_failed += 1
            
            # 每张图片下载后稍作延迟，避免请求过快
            if img_idx < len(images_data):
                time.sleep(0.5)
        
        # 每个主题处理完后稍作延迟
        if theme_index < len(themes):
            time.sleep(1)
        
        print()
    
    # 输出统计信息
    print("=" * 60)
    print(f"处理完成!")
    print(f"  主题数量: {len(themes)}")
    print(f"  成功保存: {total_saved} 张")
    print(f"  失败: {total_failed} 张")
    print(f"  保存目录: {images_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

