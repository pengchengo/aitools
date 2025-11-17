"""
关卡图片复制工具
从levelResources目录中查找包含"参考"的图片，复制到images文件夹并重命名
"""

import os
import shutil
from pathlib import Path
from typing import List


def copy_level_images(
    level_ids: List[str],
    source_dir: str = r"D:\Workspace\collect\creator\assets\levelResources",
    target_dir: str = "images",
    keyword: str = "参考"
):
    """
    复制关卡图片
    
    Args:
        level_ids: 关卡ID列表
        source_dir: 源目录路径
        target_dir: 目标目录路径（images文件夹）
        keyword: 图片文件名需要包含的关键字（默认："参考"）
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录（如果不存在）
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF'}
    
    copied_count = 0
    not_found_count = 0
    
    print(f"开始处理 {len(level_ids)} 个关卡...")
    print(f"源目录: {source_path}")
    print(f"目标目录: {target_path}")
    print(f"关键字: {keyword}\n")
    
    for level_id in level_ids:
        # 查找以关卡ID命名的文件夹（确保level_id是字符串）
        level_id_str = str(level_id)
        level_folder = source_path / level_id_str
        
        if not level_folder.exists() or not level_folder.is_dir():
            print(f"⚠️  关卡 {level_id_str}: 文件夹不存在 - {level_folder}")
            not_found_count += 1
            continue
        
        # 在文件夹中查找包含关键字的图片
        found_images = []
        for ext in image_extensions:
            # 查找所有包含关键字的图片文件
            for img_file in level_folder.glob(f"*{keyword}*{ext}"):
                if img_file.is_file():
                    found_images.append(img_file)
            # 也查找大写扩展名
            for img_file in level_folder.glob(f"*{keyword}*{ext.upper()}"):
                if img_file.is_file() and img_file not in found_images:
                    found_images.append(img_file)
        
        if not found_images:
            print(f"⚠️  关卡 {level_id_str}: 未找到包含'{keyword}'的图片")
            not_found_count += 1
            continue
        
        # 如果找到多张图片，使用第一张（或者可以全部复制）
        if len(found_images) > 1:
            print(f"⚠️  关卡 {level_id_str}: 找到 {len(found_images)} 张图片，将使用第一张")
        
        source_image = found_images[0]
        
        # 确定目标文件名和扩展名
        target_extension = source_image.suffix
        target_filename = f"level{level_id_str}{target_extension}"
        target_image = target_path / target_filename
        
        # 如果目标文件已存在，询问是否覆盖（或者直接覆盖）
        if target_image.exists():
            print(f"⚠️  关卡 {level_id_str}: 目标文件已存在，将覆盖 - {target_filename}")
        
        try:
            # 复制文件
            shutil.copy2(source_image, target_image)
            print(f"✅ 关卡 {level_id_str}: {source_image.name} -> {target_filename}")
            copied_count += 1
        except Exception as e:
            print(f"❌ 关卡 {level_id_str}: 复制失败 - {e}")
            not_found_count += 1
    
    print(f"\n=== 处理完成 ===")
    print(f"成功复制: {copied_count} 张")
    print(f"未找到/失败: {not_found_count} 张")
    print(f"总计: {len(level_ids)} 个关卡")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='关卡图片复制工具')
    parser.add_argument('--level_ids', type=str, nargs='+', required=True,
                       help='关卡ID列表，例如: --level_ids 3001 3008')
    parser.add_argument('--source_dir', type=str, 
                       default=r"D:\Workspace\collect\creator\assets\levelResources",
                       help='源目录路径')
    parser.add_argument('--target_dir', type=str, default='images',
                       help='目标目录路径（默认: images）')
    parser.add_argument('--keyword', type=str, default='参考',
                       help='图片文件名需要包含的关键字（默认: 参考）')
    
    args = parser.parse_args()
    
    copy_level_images(
        level_ids=args.level_ids,
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        keyword=args.keyword
    )


# 如果直接运行脚本，可以使用硬编码的关卡ID列表
if __name__ == "__main__":
    import sys
    
    # 如果提供了命令行参数，使用命令行模式
    if len(sys.argv) > 1:
        main()
    else:
        # 否则使用硬编码的关卡ID列表
        # 在这里修改你的关卡ID列表
        level_ids = [4,13,23,6,0,2,17,15,22,12,1,9,16,21,3,14,7,5,11,10,8,27,28,29,30,31,32,33,34,35,36,37,38,40,39,41,42,43,3002,3004,3003,44,3001
                       ,3005,45,46,3006,47,3101,3008,3009,3102,3103,310475,76,72,73,74,106,79,77,78,100,59,60,61,62,101,80,55,69,58,71,49,67,103,54,102,64,63,105,53
        ,104,82,65,48,107,52,81,57,68,56,70,66,109,85,83,87,90,92,88,86,84]
        
        copy_level_images(level_ids=level_ids)

