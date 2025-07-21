import json
import random
from pathlib import Path

def split_visual_genome_dataset(image_data_file, 
                               region_descriptions_file,
                               output_dir="./",
                               subset_size=3000,
                               seed=42):
    """
    将Visual Genome数据集分割成两部分
    
    Args:
        image_data_file: image_data.json文件路径
        region_descriptions_file: region_descriptions.json文件路径
        output_dir: 输出目录
        subset_size: 子集大小（默认3000）
        seed: 随机种子
    """
    
    print(f"🚀 开始分割Visual Genome数据集...")
    print(f"📊 子集大小: {subset_size} 张图像")
    
    # 设置随机种子
    random.seed(seed)
    
    # 1. 加载数据
    print("📖 加载image_data.json...")
    with open(image_data_file, 'r') as f:
        image_data = json.load(f)
    
    print("📖 加载region_descriptions.json...")
    with open(region_descriptions_file, 'r') as f:
        region_data = json.load(f)
    
    print(f"✅ 总共 {len(image_data)} 张图像")
    
    # 2. 随机选择子集图像
    if len(image_data) <= subset_size:
        print(f"⚠️  总图像数({len(image_data)})小于等于目标子集大小({subset_size})")
        subset_size = len(image_data)
        remaining_size = 0
    else:
        remaining_size = len(image_data) - subset_size
    
    print(f"🎯 随机选择 {subset_size} 张图像作为子集...")
    selected_images = random.sample(image_data, subset_size)
    selected_image_ids = {img['image_id'] for img in selected_images}
    
    # 3. 分割image_data
    remaining_images = [img for img in image_data if img['image_id'] not in selected_image_ids]
    
    print(f"📊 子集: {len(selected_images)} 张图像")
    print(f"📊 剩余: {len(remaining_images)} 张图像")
    
    # 4. 分割region_descriptions
    print("🔍 分割region_descriptions...")
    subset_regions = []
    remaining_regions = []
    
    for image_entry in region_data:
        # 检查这个entry包含哪些图像的regions
        regions_list = image_entry.get('regions', [])
        
        subset_entry_regions = []
        remaining_entry_regions = []
        
        for region in regions_list:
            image_id = region.get('image_id')
            if image_id in selected_image_ids:
                subset_entry_regions.append(region)
            else:
                remaining_entry_regions.append(region)
        
        # 如果有regions属于子集，创建子集entry
        if subset_entry_regions:
            subset_entry = {
                'id': image_entry.get('id'),
                'regions': subset_entry_regions
            }
            subset_regions.append(subset_entry)
        
        # 如果有regions属于剩余部分，创建剩余entry
        if remaining_entry_regions:
            remaining_entry = {
                'id': image_entry.get('id'),
                'regions': remaining_entry_regions
            }
            remaining_regions.append(remaining_entry)
    
    # 5. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 6. 保存子集文件
    subset_dir = output_path / f"VG_subset_{subset_size}"
    subset_dir.mkdir(exist_ok=True)
    
    print(f"💾 保存子集到 {subset_dir}...")
    
    with open(subset_dir / 'image_data.json', 'w') as f:
        json.dump(selected_images, f, indent=2)
    
    with open(subset_dir / 'region_descriptions.json', 'w') as f:
        json.dump(subset_regions, f, indent=2)
    

    
    # 7. 保存剩余部分（如果有的话）
    if remaining_size > 0:
        remaining_dir = output_path / f"VG_remaining_{remaining_size}"
        remaining_dir.mkdir(exist_ok=True)
        
        print(f"💾 保存剩余部分到 {remaining_dir}...")
        
        with open(remaining_dir / 'image_data.json', 'w') as f:
            json.dump(remaining_images, f, indent=2)
        
        with open(remaining_dir / 'region_descriptions.json', 'w') as f:
            json.dump(remaining_regions, f, indent=2)
        

    
    # 8. 生成统计信息
    def count_regions(region_data):
        total = 0
        for entry in region_data:
            total += len(entry.get('regions', []))
        return total
    
    subset_region_count = count_regions(subset_regions)
    remaining_region_count = count_regions(remaining_regions)
    
    stats = {
        'original_total_images': len(image_data),
        'original_total_regions': count_regions(region_data),
        'subset': {
            'images': len(selected_images),
            'regions': subset_region_count,
            'avg_regions_per_image': subset_region_count / len(selected_images) if selected_images else 0
        },
        'remaining': {
            'images': len(remaining_images),
            'regions': remaining_region_count,
            'avg_regions_per_image': remaining_region_count / len(remaining_images) if remaining_images else 0
        }
    }
    
    with open(output_path / 'split_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("✨ 分割完成！")
    print(f"📈 统计信息:")
    print(f"   原始数据: {stats['original_total_images']} 张图像, {stats['original_total_regions']} 个regions")
    print(f"   子集: {stats['subset']['images']} 张图像, {stats['subset']['regions']} 个regions")
    if remaining_size > 0:
        print(f"   剩余: {stats['remaining']['images']} 张图像, {stats['remaining']['regions']} 个regions")
    
    return subset_dir, remaining_dir if remaining_size > 0 else None


def quick_split():
    """快速分割 - 修改这里的路径"""
    
    # 🔧 修改你的文件路径
    image_data_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/image_data.json"
    region_descriptions_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/region_descriptions.json"
    
    # 分割数据集
    subset_dir, remaining_dir = split_visual_genome_dataset(
        image_data_file=image_data_file,
        region_descriptions_file=region_descriptions_file,
        output_dir="./",  # 输出到当前目录
        subset_size=1000,
        seed=42
    )
    
    print(f"\n🎉 分割完成！")
    print(f"📁 子集保存在: {subset_dir}")
    if remaining_dir:
        print(f"📁 剩余部分保存在: {remaining_dir}")
    
    print(f"\n💡 使用方法:")
    print(f"   # 使用3000张图像的子集")
    print(f"   dataset = VisualGenomeDataset(")
    print(f"       region_descriptions_file='{subset_dir}/region_descriptions.json',")
    print(f"       image_data_file='{subset_dir}/image_data.json',")
    print(f"       images_dir='./VG_100k_all',  # 原图像路径不变")
    print(f"       ...)")


if __name__ == "__main__":
    quick_split()