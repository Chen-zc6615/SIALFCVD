#!/usr/bin/env python3
"""
Alpha BLIP 数据处理模块
负责图像、Alpha通道、文本等数据的预处理
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# 确保NLTK数据可用
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class ImageProcessor:
    """图像数据处理器"""
    
    def __init__(self, processor=None):
        """
        初始化图像处理器
        
        Args:
            processor: HuggingFace的processor对象，用于图像预处理
        """
        self.processor = processor
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def preprocess_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        预处理RGB图像
        
        Args:
            image_input: 图像路径字符串或PIL Image对象
            
        Returns:
            处理后的PIL Image对象 (RGB格式)
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图像文件不存在: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError(f"不支持的图像输入类型: {type(image_input)}")
        
        return image
    
    def preprocess_alpha(self, alpha_input: Union[str, Image.Image, None]) -> Optional[Image.Image]:
        """
        预处理Alpha通道图像
        
        Args:
            alpha_input: Alpha图像路径、PIL Image对象或None
            
        Returns:
            处理后的灰度PIL Image对象或None
        """
        if alpha_input is None:
            return None
        
        if isinstance(alpha_input, str):
            if not os.path.exists(alpha_input):
                raise FileNotFoundError(f"Alpha图像文件不存在: {alpha_input}")
            alpha_image = Image.open(alpha_input)
        elif isinstance(alpha_input, Image.Image):
            alpha_image = alpha_input
        else:
            raise TypeError(f"不支持的Alpha输入类型: {type(alpha_input)}")
        
        # 确保是灰度图像
        if alpha_image.mode != 'L':
            alpha_image = alpha_image.convert('L')
        
        return alpha_image
    
    def process_batch_images(self, 
                           image_paths: List[str],
                           alpha_paths: Optional[List[str]] = None) -> Tuple[List[Image.Image], List[Optional[Image.Image]]]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            alpha_paths: Alpha图像路径列表（可选）
            
        Returns:
            (处理后的图像列表, 处理后的Alpha图像列表)
        """
        processed_images = []
        processed_alphas = []
        
        if alpha_paths is None:
            alpha_paths = [None] * len(image_paths)
        
        if len(image_paths) != len(alpha_paths):
            raise ValueError(f"图像数量({len(image_paths)})与Alpha数量({len(alpha_paths)})不匹配")
        
        for img_path, alpha_path in zip(image_paths, alpha_paths):
            try:
                # 处理主图像
                processed_img = self.preprocess_image(img_path)
                processed_images.append(processed_img)
                
                # 处理Alpha图像
                processed_alpha = self.preprocess_alpha(alpha_path)
                processed_alphas.append(processed_alpha)
                
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                # 添加占位符，保持列表长度一致
                processed_images.append(None)
                processed_alphas.append(None)
        
        return processed_images, processed_alphas
    
    def get_images_from_directory(self, 
                                image_dir: str, 
                                alpha_dir: Optional[str] = None) -> Tuple[List[str], List[Optional[str]]]:
        """
        从目录获取图像文件路径
        
        Args:
            image_dir: 图像目录路径
            alpha_dir: Alpha图像目录路径（可选）
            
        Returns:
            (图像路径列表, Alpha路径列表)
        """
        image_dir = Path(image_dir)
        alpha_dir = Path(alpha_dir) if alpha_dir else None
        
        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        
        # 获取所有支持的图像文件
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in self.supported_formats]
        image_files.sort()  # 排序确保一致性
        
        image_paths = [str(f) for f in image_files]
        alpha_paths = []
        
        # 查找对应的Alpha图像
        for image_file in image_files:
            alpha_path = None
            if alpha_dir and alpha_dir.exists():
                # 尝试多种命名模式
                alpha_candidates = [
                    alpha_dir / image_file.name,
                    alpha_dir / f"{image_file.stem}_mask{image_file.suffix}",
                    alpha_dir / f"{image_file.stem}_alpha{image_file.suffix}",
                    alpha_dir / f"{image_file.stem}_seg{image_file.suffix}",
                ]
                
                for candidate in alpha_candidates:
                    if candidate.exists():
                        alpha_path = str(candidate)
                        break
            
            alpha_paths.append(alpha_path)
        
        return image_paths, alpha_paths
    
    def prepare_model_inputs(self, 
                           images: List[Image.Image], 
                           alpha_images: List[Optional[Image.Image]],
                           device: torch.device) -> Dict[str, torch.Tensor]:
        """
        准备模型输入张量
        
        Args:
            images: 处理后的图像列表
            alpha_images: 处理后的Alpha图像列表
            device: 目标设备
            
        Returns:
            模型输入字典
        """
        if self.processor is None:
            raise ValueError("未设置processor，无法准备模型输入")
        
        # 过滤掉None值的图像
        valid_images = [img for img in images if img is not None]
        valid_alphas = [alpha for alpha in alpha_images if alpha is not None]
        
        if not valid_images:
            raise ValueError("没有有效的图像可处理")
        
        # 处理主图像
        inputs = self.processor(
            images=valid_images,
            return_tensors="pt",
            padding=True
        )
        
        # 处理Alpha通道
        if valid_alphas:
            alpha_tensors = []
            for alpha_img in alpha_images:
                if alpha_img is not None:
                    alpha_processed = self.processor.image_processor(
                        alpha_img, 
                        return_tensors="pt"
                    )['pixel_values']
                    
                    # 如果是RGB图像，取第一个通道作为alpha
                    if alpha_processed.shape[1] == 3:
                        alpha_processed = alpha_processed[:, 0:1, :, :]
                    
                    alpha_tensors.append(alpha_processed)
                else:
                    # 为None的alpha创建零张量占位符
                    dummy_alpha = torch.zeros(1, 1, 224, 224)  # 假设输入尺寸为224x224
                    alpha_tensors.append(dummy_alpha)
            
            if alpha_tensors:
                inputs['alpha_values'] = torch.cat(alpha_tensors, dim=0)
        
        # 移动到指定设备
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        return inputs


class TextProcessor:
    """文本数据处理器"""
    
    def __init__(self):
        """初始化文本处理器"""
        self.smoothing = SmoothingFunction().method1
    
    def preprocess_text(self, text: str) -> str:
        """
        预处理文本：标准化格式
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 转小写
        text = text.lower().strip()
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            词汇列表
        """
        text = self.preprocess_text(text)
        tokens = text.split()
        return tokens
    
    def load_ground_truth(self, gt_file: str) -> List[List[str]]:
        """
        加载ground truth数据
        
        Args:
            gt_file: ground truth文件路径
            
        Returns:
            参考文本列表（每个样本可能有多个参考）
        """
        gt_file = Path(gt_file)
        
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth文件不存在: {gt_file}")
        
        if gt_file.suffix.lower() == '.json':
            with open(gt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            references = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # 尝试多个可能的字段名
                        text = None
                        for field in ['text', 'caption', 'description', 'sent', 'sentence', 'label']:
                            if field in item:
                                text = item[field]
                                break
                        
                        if text:
                            # 如果text是列表，保持原样；否则包装成列表
                            if isinstance(text, list):
                                references.append(text)
                            else:
                                references.append([text])
                        else:
                            print(f"警告: 无法从数据项中提取文本: {item}")
                            references.append([""])
                    elif isinstance(item, str):
                        references.append([item])
                    else:
                        print(f"警告: 不支持的数据项类型: {type(item)}")
                        references.append([""])
            
            elif isinstance(data, dict):
                # 假设是 {"image_id": "description"} 或 {"image_id": ["desc1", "desc2"]} 格式
                references = []
                for key, value in data.items():
                    if isinstance(value, list):
                        references.append(value)
                    else:
                        references.append([str(value)])
        
        elif gt_file.suffix.lower() == '.txt':
            with open(gt_file, 'r', encoding='utf-8') as f:
                references = [[line.strip()] for line in f.readlines() if line.strip()]
        
        else:
            raise ValueError(f"不支持的文件格式: {gt_file.suffix}")
        
        return references
    
    def save_predictions(self, 
                        predictions: List[str], 
                        image_paths: List[str],
                        output_file: str,
                        alpha_paths: Optional[List[str]] = None) -> None:
        """
        保存预测结果
        
        Args:
            predictions: 预测文本列表
            image_paths: 图像路径列表
            output_file: 输出文件路径
            alpha_paths: Alpha图像路径列表（可选）
        """
        results = []
        
        alpha_paths = alpha_paths or [None] * len(predictions)
        
        for i, (pred, img_path, alpha_path) in enumerate(zip(predictions, image_paths, alpha_paths)):
            result = {
                'id': i,
                'image_path': img_path,
                'alpha_path': alpha_path,
                'prediction': pred
            }
            results.append(result)
        
        # 保存为JSON格式
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_samples': len(results),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"预测结果已保存到: {output_path}")


class DatasetProcessor:
    """数据集处理器 - 整合图像和文本处理"""
    
    def __init__(self, processor=None):
        """
        初始化数据集处理器
        
        Args:
            processor: HuggingFace的processor对象
        """
        self.image_processor = ImageProcessor(processor)
        self.text_processor = TextProcessor()
    
    def process_dataset(self, 
                       image_dir: str,
                       alpha_dir: Optional[str] = None,
                       ground_truth_file: Optional[str] = None) -> Dict:
        """
        处理完整数据集
        
        Args:
            image_dir: 图像目录
            alpha_dir: Alpha图像目录（可选）
            ground_truth_file: ground truth文件（可选）
            
        Returns:
            处理后的数据集信息
        """
        # 获取图像路径
        image_paths, alpha_paths = self.image_processor.get_images_from_directory(
            image_dir, alpha_dir
        )
        
        # 加载ground truth（如果提供）
        references = None
        if ground_truth_file:
            try:
                references = self.text_processor.load_ground_truth(ground_truth_file)
                print(f"加载了 {len(references)} 个参考文本")
            except Exception as e:
                print(f"加载ground truth失败: {e}")
        
        dataset_info = {
            'image_paths': image_paths,
            'alpha_paths': alpha_paths,
            'references': references,
            'num_samples': len(image_paths)
        }
        
        # 数据一致性检查
        if references and len(references) != len(image_paths):
            print(f"警告: 图像数量({len(image_paths)}) 与参考文本数量({len(references)}) 不匹配")
        
        print(f"数据集处理完成: {len(image_paths)} 张图像")
        if alpha_dir:
            alpha_count = sum(1 for p in alpha_paths if p is not None)
            print(f"找到 {alpha_count} 个Alpha图像")
        
        return dataset_info


# 便捷函数
def create_data_processor(processor=None) -> DatasetProcessor:
    """创建数据处理器的便捷函数"""
    return DatasetProcessor(processor)

def quick_process_directory(image_dir: str, 
                          alpha_dir: Optional[str] = None,
                          ground_truth_file: Optional[str] = None) -> Dict:
    """快速处理目录的便捷函数"""
    processor = create_data_processor()
    return processor.process_dataset(image_dir, alpha_dir, ground_truth_file)


if __name__ == "__main__":
    # 示例用法
    processor = create_data_processor()
    
    # 处理单张图像
    image_proc = processor.image_processor
    img = image_proc.preprocess_image("sample.jpg")
    alpha = image_proc.preprocess_alpha("sample_mask.jpg")
    
    # 处理目录
    dataset_info = processor.process_dataset(
        image_dir="./images",
        alpha_dir="./masks", 
        ground_truth_file="./annotations.json"
    )
    
    print("数据处理模块测试完成！")