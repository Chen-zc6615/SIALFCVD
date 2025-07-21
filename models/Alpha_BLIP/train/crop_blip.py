import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import json
from tqdm import tqdm
import os
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
import random
import sys
sys.path.append('/home/chenzc/cvd/model_blip')  # 添加你的项目路径
from data.datasets_loader import VisualGenomeDataset, DualImageTransform  # 修改为实际路径

def crop_image_with_bbox(pil_image, bbox):
    """
    根据bbox精确切割PIL图像，不添加padding
    Args:
        pil_image: PIL Image对象
        bbox: [x, y, width, height] - COCO格式
    Returns:
        PIL Image: 切割后的图像
    """
    img_width, img_height = pil_image.size
    x, y, w, h = bbox
    
    # 确保边界不超出图像范围
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_width, x + w)
    y2 = min(img_height, y + h)
    
    # 切割图像
    cropped_image = pil_image.crop((x1, y1, x2, y2))
    return cropped_image

class VGBlipDataset(Dataset):
    """适配BLIP的Visual Genome数据集 - bbox切割图像训练"""
    def __init__(self, 
                 region_descriptions_file,
                 image_data_file,
                 images_dir,
                 processor,
                 max_length=64
                 ):
        
        self.processor = processor
        self.max_length = max_length
        
        self.vg_dataset = VisualGenomeDataset(
            region_descriptions_file,
            image_data_file, 
            images_dir
        )
        
        print("Dataset initialized with exact bbox cropping")
    
    def __len__(self):
        return len(self.vg_dataset)
    
    def __getitem__(self, idx):
        # 从VG数据集获取样本
        vg_sample = self.vg_dataset[idx]
        
        orig_image_pil = vg_sample['orig_image']  # PIL Image对象
        target_text = vg_sample['text']
        bbox = vg_sample['bbox']
        
        # 精确按bbox切割图像
        cropped_image = crop_image_with_bbox(orig_image_pil, bbox)
        
        # 空的输入prompt - 让模型直接描述切割后的图像
        input_prompt = ""
        
        # 处理输入（切割后的图像 + 空prompt）
        inputs = self.processor(
            cropped_image,
            input_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            do_rescale=False
        )
        
        # 处理目标文本
        target_encoded = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length", 
            truncation=True,
            max_length=self.max_length
        )
        
        # 移除batch维度
        pixel_values = inputs['pixel_values'].squeeze()
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = target_encoded['input_ids'].squeeze()
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class VGBlipTrainer:
    """VG数据集BLIP训练器 - bbox切割图像方法"""
    def __init__(self, 
                 model_name="Salesforce/blip-image-captioning-base",
                 vg_region_descriptions_file=None,
                 vg_image_data_file=None,
                 vg_images_dir=None,
                 output_dir="./blip_vg_cropped",
                 max_length=128
                 ):
        
        self.model_name = model_name
        self.output_dir = output_dir
        
        # 初始化processor和model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        print(f"Loaded BLIP model: {model_name}")
        print(f"Vocabulary size: {len(self.processor.tokenizer)}")
        
        # 创建数据集
        if vg_region_descriptions_file and vg_image_data_file and vg_images_dir:
            print("Creating VG dataset with cropped images...")
            self.dataset = VGBlipDataset(
                vg_region_descriptions_file,
                vg_image_data_file,
                vg_images_dir,
                self.processor,
                max_length=max_length
            )
            # 创建测试用例索引列表
            self.test_indices = list(range(len(self.dataset)))
        else:
            self.dataset = None
            self.test_indices = []
    
    def create_dataloader(self, batch_size=8, num_workers=4):
        """创建训练数据加载器"""
        if self.dataset is None:
            raise ValueError("Dataset not initialized")
        
        print(f"Using full dataset: {len(self.dataset)} samples")
        
        # 创建单个DataLoader，使用全部数据
        train_loader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader
    
    def run_test_on_samples(self, device, num_samples=8):
        """从训练集中随机抽取样本进行测试"""
        self.model.eval()
        
        # 随机选择测试样本
        test_sample_indices = random.sample(self.test_indices, min(num_samples, len(self.test_indices)))
        
        test_results = []
        
        with torch.no_grad():
            for idx in test_sample_indices:
                # 获取原始VG样本
                vg_sample = self.dataset.vg_dataset[idx]
                orig_image_pil = vg_sample['orig_image']
                ground_truth = vg_sample['text']
                bbox = vg_sample['bbox']
                
                # 精确切割图像（与训练时相同的方式）
                cropped_image = crop_image_with_bbox(orig_image_pil, bbox)
                
                # 处理输入 - 空prompt
                inputs = self.processor(
                    cropped_image, 
                    "", 
                    return_tensors="pt",
                    do_rescale=False
                ).to(device)
                
                # 生成描述
                generated_ids = self.model.generate(
                    pixel_values=inputs['pixel_values'],
                    max_new_tokens=25,
                    num_beams=3,
                    do_sample=False,
                    early_stopping=True,
                )
                
                # 解码生成的文本
                generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                test_results.append({
                    'sample_idx': idx,
                    'bbox': bbox,
                    'ground_truth': ground_truth,
                    'generated_text': generated_text
                })
        
        self.model.train()  # 切换回训练模式
        return test_results
    
    def print_test_results(self, test_results, step):
        """打印测试结果"""
        print(f"\n{'='*60}")
        print(f"TEST RESULTS AT STEP {step} (Exact Bbox Cropping)")
        print(f"{'='*60}")
        
        for i, result in enumerate(test_results, 1):
            print(f"\n--- Sample {i} (Index: {result['sample_idx']}) ---")
            print(f"BBox: {result['bbox']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Generated:    {result['generated_text']}")
            print("-" * 50)
        
        print(f"{'='*60}\n")
    
    def save_test_results(self, test_results, step):
        """保存测试结果到文件"""
        results_dir = Path(self.output_dir) / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"test_results_step_{step}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'step': step,
                'results': test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Test results saved to: {results_file}")
    
    def train(self,
              num_epochs=1,
              batch_size=8,
              learning_rate=2e-5,
              warmup_steps=500,
              save_steps=1000,
              test_steps=1000,
              max_grad_norm=1.0):
        """训练模型"""
        
        # 创建数据加载器
        train_loader = self.create_dataloader(batch_size=batch_size)
        
        # 设置优化器和调度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        print(f"Training on device: {device}")
        print(f"Total training steps: {total_steps}")
        print(f"Training samples: {len(self.dataset)}")
        print("Using exact bbox cropping")
        print(f"Test every {test_steps} steps")
        
        # 训练循环
        global_step = 0
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            
            for step, batch in enumerate(train_bar):
                # 移动数据到设备
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                global_step += 1
                
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': global_step
                })
                
                # 定期测试
                if global_step % test_steps == 0:
                    print(f"\nRunning test at step {global_step}...")
                    test_results = self.run_test_on_samples(device, num_samples=8)
                    self.print_test_results(test_results, global_step)
                    self.save_test_results(test_results, global_step)
                
                # 定期保存检查点
                if global_step % save_steps == 0:
                    self.save_model(f"checkpoint_step_{global_step}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Average Train Loss: {avg_train_loss:.4f}")
            print(f"  Total Steps: {global_step}")
            
            # 每个epoch结束保存模型
            self.save_model(f"final_model_epoch_{epoch+1}")
        
        print(f"\nTraining completed! Final average loss: {avg_train_loss:.4f}")
        return avg_train_loss
    
    def save_model(self, checkpoint_name):
        """保存模型"""
        save_path = Path(self.output_dir) / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")
    
    def test_model(self, image_path, bbox=None):
        """测试模型生成能力"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 如果提供了bbox，则精确切割图像
        if bbox is not None:
            cropped_image = crop_image_with_bbox(image, bbox)
            
            print(f"Original image size: {image.size}")
            print(f"Cropped image size: {cropped_image.size}")
            print(f"BBox: {bbox}")
        else:
            cropped_image = image
        
        # 处理输入 - 空prompt
        inputs = self.processor(cropped_image, "", return_tensors="pt", do_rescale=False).to(device)
        
        # 生成文本
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs['pixel_values'],
                max_length=50,
                num_beams=4,
                do_sample=True,
                temperature=0.7
            )
        
        generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

# 使用示例
if __name__ == "__main__":
    # 配置路径 - 使用你的实际路径
    config = {
        'vg_region_descriptions_file': "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_train_105077/region_descriptions.json",
        'vg_image_data_file': "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_train_105077/image_data.json", 
        'vg_images_dir': "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_100K_all",
        'output_dir': "./blip_vg_cropped",
        'model_name': "Salesforce/blip-image-captioning-base"
    }
    
    # 创建训练器 - 只使用精确bbox切割
    trainer = VGBlipTrainer(
        model_name=config['model_name'],
        vg_region_descriptions_file=config['vg_region_descriptions_file'],
        vg_image_data_file=config['vg_image_data_file'],
        vg_images_dir=config['vg_images_dir'],
        output_dir=config['output_dir']
    )
    
    # 开始训练
    trainer.train(
        num_epochs=1,
        batch_size=4,  # 根据GPU内存调整
        learning_rate=2e-5,
        warmup_steps=500,
        save_steps=1000,
        test_steps=200  # 每200步测试一次
    )
    
    # 测试模型示例
    # result = trainer.test_model("test_image.jpg", bbox=[100, 50, 200, 150])
    # print(f"Generated description: {result}")