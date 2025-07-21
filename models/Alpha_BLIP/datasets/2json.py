#!/usr/bin/env python3
"""
Visual Genome数据集转换脚本 - 生成单个标注文件
"""
import json
import os
from pathlib import Path
import argparse
import random
from tqdm import tqdm

def convert_vg_to_single_annotation(image_data_file, region_descriptions_file, 
                                   image_folder, output_file, max_samples_per_image=5):
    """将VG数据转换为单个标注文件"""
    print("🔧 开始转换VG数据集...")
    
    # 1. 加载图像数据
    print(f"📖 加载图像数据: {image_data_file}")
    with open(image_data_file, 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    
    # 建立image_id到文件名的映射
    id_to_filename = {}
    missing_count = 0
    
    for img in image_data:
        image_id = img['image_id']
        filename = f"{image_id}.jpg"
        
        # 检查文件是否存在
        full_path = os.path.join(image_folder, filename)
        if os.path.exists(full_path):
            id_to_filename[image_id] = filename
        else:
            missing_count += 1
    
    print(f"✅ 找到 {len(id_to_filename)} 个图像文件")
    if missing_count > 0:
        print(f"⚠️  缺失 {missing_count} 个图像文件")
    
    # 2. 加载区域描述
    print(f"📖 加载区域描述: {region_descriptions_file}")
    with open(region_descriptions_file, 'r', encoding='utf-8') as f:
        region_data = json.load(f)
    
    # 3. 转换数据
    annotations = []
    skipped_images = 0
    
    for img_data in tqdm(region_data, desc="转换区域描述"):
        image_id = img_data['id']
        
        # 检查是否有对应的图像文件
        if image_id not in id_to_filename:
            skipped_images += 1
            continue
        
        filename = id_to_filename[image_id]
        regions = img_data.get('regions', [])
        
        # 限制每张图像的区域数量
        if len(regions) > max_samples_per_image:
            regions = random.sample(regions, max_samples_per_image)
        
        for region in regions:
            phrase = region.get('phrase', '').strip()
            if not phrase or len(phrase) < 5:  # 过滤太短的描述
                continue
            
            # 提取边界框信息
            bbox = None
            if all(key in region for key in ['x', 'y', 'width', 'height']):
                bbox = [
                    region['x'], 
                    region['y'], 
                    region['width'], 
                    region['height']
                ]
                # 过滤掉过小的区域
                if bbox[2] < 10 or bbox[3] < 10:
                    continue
            
            # 基本文本清理
            phrase = ' '.join(phrase.split())  # 规范化空格
            
            annotations.append({
                "image": filename,
                "caption": phrase,
                "bbox": bbox
            })
    
    print(f"✅ 转换了 {len(annotations)} 个区域描述")
    if skipped_images > 0:
        print(f"⚠️  跳过了 {skipped_images} 个没有图像文件的条目")
    
    # 4. 统计信息
    unique_images = set(ann['image'] for ann in annotations)
    caption_lengths = [len(ann['caption'].split()) for ann in annotations]
    with_bbox = sum(1 for ann in annotations if ann.get('bbox'))
    
    print(f"\n📊 数据集统计:")
    print(f"总标注数: {len(annotations)}")
    print(f"图像数量: {len(unique_images)}")
    print(f"平均描述长度: {sum(caption_lengths) / len(caption_lengths):.1f} 词")
    print(f"包含边界框: {with_bbox} ({with_bbox/len(annotations)*100:.1f}%)")
    
    # 5. 保存文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"💾 保存到: {output_file}")
    
    # 6. 显示示例
    print(f"\n📋 示例标注:")
    for i, sample in enumerate(annotations[:5]):
        print(f"  {i+1}. 图像: {sample['image']}")
        print(f"     描述: {sample['caption']}")
        if sample.get('bbox'):
            print(f"     边界框: {sample['bbox']}")
    
    return len(annotations)

def main():
    parser = argparse.ArgumentParser(description="转换Visual Genome数据集为单个标注文件")
    parser.add_argument("--image-data", default="/home/chenzc/cvd/visual_genome/image_data.json",
                       help="图像数据文件路径")
    parser.add_argument("--region-descriptions", default="/home/chenzc/cvd/visual_genome/region_descriptions.json", 
                       help="区域描述文件路径")
    parser.add_argument("--image-folder", default="/home/chenzc/cvd/visual_genome/VG_100K_all",
                       help="图像文件夹路径")
    parser.add_argument("--output", default="data/vg_annotations.json",
                       help="输出标注文件路径")
    parser.add_argument("--max-samples-per-image", type=int, default=5,
                       help="每张图像最大标注数量")
    
    args = parser.parse_args()
    
    print("🔧 Visual Genome数据集转换工具")
    print("=" * 50)
    
    # 检查输入文件
    if not os.path.exists(args.image_data):
        print(f"❌ 找不到图像数据文件: {args.image_data}")
        return
    
    if not os.path.exists(args.region_descriptions):
        print(f"❌ 找不到区域描述文件: {args.region_descriptions}")
        return
    
    if not os.path.exists(args.image_folder):
        print(f"❌ 找不到图像文件夹: {args.image_folder}")
        return
    
    try:
        num_annotations = convert_vg_to_single_annotation(
            args.image_data,
            args.region_descriptions,
            args.image_folder,
            args.output,
            args.max_samples_per_image
        )
        
        if num_annotations > 0:
            print("\n✅ 转换完成！")
            print(f"📄 输出文件: {args.output}")
            print(f"\n🚀 现在可以在训练脚本中使用:")
            print(f"  --annotations-train {args.output}")
            print(f"  --annotations-val {args.output}")
            print(f"  --images-dir {args.image_folder}")
        else:
            print("❌ 没有生成任何标注数据")
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()