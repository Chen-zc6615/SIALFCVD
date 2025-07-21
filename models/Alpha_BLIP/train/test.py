#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLIP模型结构打印工具
支持多种方式查看BLIP模型的网络结构
"""

import torch
import torch.nn as nn
from transformers import BlipModel, BlipConfig
import torchsummary
from torchinfo import summary

def print_blip_structure_basic():
    """基础方法：直接打印模型结构"""
    print("=" * 60)
    print("方法1: 基础模型结构打印")
    print("=" * 60)
    
    # 加载预训练的BLIP模型
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # 打印完整模型结构
    print(model)
    
    return model

def print_blip_structure_detailed():
    """详细方法：使用torchinfo获取详细信息"""
    print("\n" + "=" * 60)
    print("方法2: 详细模型结构 (使用torchinfo)")
    print("=" * 60)
    
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # 定义输入尺寸
    # BLIP需要图像和文本两个输入
    batch_size = 1
    image_channels = 3
    image_size = 384  # BLIP默认图像尺寸
    text_length = 30  # 文本序列长度
    
    try:
        # 使用torchinfo获取详细摘要
        model_summary = summary(
            model,
            input_size=[
                (batch_size, image_channels, image_size, image_size),  # 图像输入
                (batch_size, text_length)  # 文本输入
            ],
            dtypes=[torch.float32, torch.long],
            verbose=1
        )
        print(model_summary)
    except Exception as e:
        print(f"torchinfo summary failed: {e}")
        print("使用基础方法显示结构...")
        print(model)

def print_blip_components():
    """分组件打印：分别展示各个组件"""
    print("\n" + "=" * 60)
    print("方法3: 分组件结构打印")
    print("=" * 60)
    
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    
    print("视觉编码器 (Vision Encoder):")
    print("-" * 40)
    print(model.vision_model)
    
    print("\n文本编码器 (Text Encoder):")
    print("-" * 40)
    print(model.text_encoder)
    
    print("\n文本解码器 (Text Decoder):")
    print("-" * 40)
    print(model.text_decoder)
    
    print("\n视觉投影层 (Vision Projection):")
    print("-" * 40)
    print(model.vision_projection)
    
    print("\n文本投影层 (Text Projection):")
    print("-" * 40)
    print(model.text_projection)

def print_blip_parameters():
    """打印参数统计信息"""
    print("\n" + "=" * 60)
    print("方法4: 模型参数统计")
    print("=" * 60)
    
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"不可训练参数数量: {total_params - trainable_params:,}")
    
    # 按组件统计参数
    print("\n各组件参数数量:")
    print("-" * 30)
    
    components = {
        'Vision Model': model.vision_model,
        'Text Encoder': model.text_encoder,
        'Text Decoder': model.text_decoder,
        'Vision Projection': model.vision_projection,
        'Text Projection': model.text_projection
    }
    
    for name, component in components.items():
        if component is not None:
            params = sum(p.numel() for p in component.parameters())
            print(f"{name}: {params:,}")

def print_blip_config():
    """打印模型配置信息"""
    print("\n" + "=" * 60)
    print("方法5: 模型配置信息")
    print("=" * 60)
    
    config = BlipConfig.from_pretrained("Salesforce/blip-image-captioning-base")
    
    print("BLIP模型配置:")
    print("-" * 30)
    for key, value in config.to_dict().items():
        if not key.startswith('_'):
            print(f"{key}: {value}")

def create_model_graph():
    """创建模型计算图"""
    print("\n" + "=" * 60)
    print("方法6: 保存模型结构图")
    print("=" * 60)
    
    try:
        from torchviz import make_dot
        
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        model.eval()
        
        # 创建dummy输入
        batch_size = 1
        image = torch.randn(batch_size, 3, 384, 384)
        input_ids = torch.randint(0, 1000, (batch_size, 30))
        
        # 前向传播
        with torch.no_grad():
            outputs = model(pixel_values=image, input_ids=input_ids)
        
        # 创建计算图
        dot = make_dot(outputs.last_hidden_state, params=dict(model.named_parameters()))
        dot.render("blip_model_graph", format="png")
        print("模型计算图已保存为 blip_model_graph.png")
        
    except ImportError:
        print("需要安装 torchviz: pip install torchviz")
    except Exception as e:
        print(f"创建计算图失败: {e}")

def main():
    """主函数：执行所有打印方法"""
    print("BLIP模型结构分析工具")
    print("=" * 60)
    
    try:
        # 基础结构打印
        model = print_blip_structure_basic()
        
        # 详细结构打印
        print_blip_structure_detailed()
        
        # 分组件打印
        print_blip_components()
        
        # 参数统计
        print_blip_parameters()
        
        # 配置信息
        print_blip_config()
        
        # 模型图（可选）
        create_model_graph()
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("请确保已安装必要的依赖包:")
        print("pip install torch transformers torchinfo")

if __name__ == "__main__":
    main()

# 使用示例:
"""
# 1. 安装依赖
pip install torch transformers torchinfo

# 2. 运行脚本
python blip_structure.py

# 3. 或者在Jupyter中逐个执行函数
model = print_blip_structure_basic()
print_blip_components()
print_blip_parameters()
"""