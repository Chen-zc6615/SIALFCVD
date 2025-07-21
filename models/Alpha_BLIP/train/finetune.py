#!/usr/bin/env python3
"""Alpha BLIP 训练脚本"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict
import random
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm

from model.alpha_blip import AlphaBlipForConditionalGeneration
from data.datasets_loader import CVDDataset, VisualGenomeDataset

# 默认配置
DEFAULT_CONFIG = {
    "model": {"pretrained_model_name": "Salesforce/blip-image-captioning-base", "max_text_length": 77},
    "training": {
        "output_dir": "./outputs/alpha_blip_training", "device": "cuda", "num_epochs": 10,
        "batch_size": 8, "learning_rate": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.1,
        "max_grad_norm": 1.0, "val_split": 0.1, "seed": 42, "num_workers": 4,
        "save_steps": 1, "alpha_only": False, "resume_from_checkpoint": None
    },
    "cvd_dataset": {
        "enabled": False, "image_dir": "/home/chenzc/cvd/Places365_with_cvd_7_20K_4o",
        "annotations_file": "/home/chenzc/cvd/Place365CVDdata.json"
    },
    "visual_genome_dataset": {
        "enabled": True, "region_descriptions_file": "/home/chenzc/cvd/visual_genome/region_descriptions.json",
        "image_data_file": "/home/chenzc/cvd/visual_genome/image_data.json",
        "images_dir": "/home/chenzc/cvd/visual_genome/VG_100K_all", "mask_type": "ellipse", "min_region_size": 1000
    }
}

class AlphaBlipTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.output_dir = Path(config['training']['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.inference_dir = self.output_dir / 'inference_samples'
        for directory in [self.output_dir, self.checkpoint_dir, self.inference_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.inference_samples = None  # 存储固定的推理样本
        
    def collate_fn(self, batch):
        """自定义collate函数处理PIL图像"""
        orig_images = [item['orig_image'] for item in batch]
        mask_images = [item['mask_image'] for item in batch]
        texts = [item['text'] for item in batch]
        
        return {
            'orig_image': orig_images,
            'mask_image': mask_images,
            'text': texts
        }

    def load_datasets(self):
        all_datasets = []
        
        # CVD数据集
        if self.config.get('cvd_dataset', {}).get('enabled', False):
            cvd_config = self.config['cvd_dataset']
            cvd_dataset = CVDDataset(image_dir=cvd_config['image_dir'], annotations_file=cvd_config['annotations_file'])
            all_datasets.append(cvd_dataset)
            print(f"CVD数据集: {len(cvd_dataset)} 样本")
        
        # Visual Genome数据集
        if self.config.get('visual_genome_dataset', {}).get('enabled', False):
            vg_config = self.config['visual_genome_dataset']
            vg_dataset = VisualGenomeDataset(
                region_descriptions_file=vg_config['region_descriptions_file'],
                image_data_file=vg_config['image_data_file'],
                images_dir=vg_config['images_dir'],
                mask_type=vg_config.get('mask_type', 'ellipse')
            )
            all_datasets.append(vg_dataset)
            print(f"VG数据集: {len(vg_dataset)} 样本")
        
        if not all_datasets:
            raise ValueError("没有加载任何数据集！")
        
        combined_dataset = ConcatDataset(all_datasets)
        total_size = len(combined_dataset)
        val_size = int(total_size * self.config['training']['val_split'])
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            combined_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['training']['seed'])
        )
        
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                     num_workers=num_workers, pin_memory=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True, collate_fn=self.collate_fn)
        
        print(f"训练集: {train_size}, 验证集: {val_size}")
        
        # 从训练数据中随机选择16个样本用于推理测试
        self.select_inference_samples(train_dataset)

    def setup_model(self):
        model_name = self.config['model']['pretrained_model_name']
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AlphaBlipForConditionalGeneration.from_pretrained_alpha(model_name)
        self.model.to(self.device)
        
        if self.config['training']['alpha_only']:
            for name, param in self.model.named_parameters():
                param.requires_grad = 'alpha' in name.lower()
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable_params:,}")

    def select_inference_samples(self, dataset):
        """从训练数据中随机选择16个样本用于推理测试"""
        total_samples = len(dataset)
        random.seed(42)  # 固定随机种子确保每次选择相同的样本
        sample_indices = random.sample(range(total_samples), min(16, total_samples))
        
        self.inference_samples = []
        for idx in sample_indices:
            sample = dataset[idx]
            self.inference_samples.append({
                'orig_image': sample['orig_image'],
                'mask_image': sample['mask_image'],
                'text': sample['text'],
                'index': idx
            })
        
        print(f"选择了 {len(self.inference_samples)} 个样本用于推理测试")

    def run_inference_test(self):
        """运行推理测试并保存结果"""
        if not self.inference_samples:
            return
            
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for i, sample in enumerate(self.inference_samples):
                try:
                    # 准备输入
                    images = [sample['orig_image']]
                    alpha_channels = [sample['mask_image']] if sample['mask_image'] else None
                    
                    # 处理图像
                    inputs = self.processor(
                        images=images,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # 处理alpha通道
                    if alpha_channels is not None:
                        alpha_img = alpha_channels[0]
                        alpha_processed = self.processor.image_processor(
                            alpha_img, 
                            return_tensors="pt"
                        )['pixel_values']
                        
                        if alpha_processed.shape[1] == 3:
                            alpha_processed = alpha_processed[:, 0:1, :, :]
                        
                        inputs['alpha_values'] = alpha_processed
                    
                    # 移动到设备
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    
                    # 生成文本
                    generated_ids = self.model.generate(
                        pixel_values=inputs['pixel_values'],
                        alpha_values=inputs.get('alpha_values'),
                        max_length=50,
                        num_beams=3,
                        early_stopping=True
                    )
                    
                    generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    
                    result = {
                        'sample_index': sample['index'],
                        'ground_truth': sample['text'],
                        'generated_text': generated_text,
                        'step': self.global_step,
                        'epoch': self.epoch
                    }
                    results.append(result)
                    
                    # 保存图像
                    step_dir = self.inference_dir / f"step_{self.global_step}"
                    step_dir.mkdir(exist_ok=True)
                    
                    # 保存原图和mask
                    sample['orig_image'].save(step_dir / f"sample_{i}_orig.jpg")
                    if sample['mask_image']:
                        sample['mask_image'].save(step_dir / f"sample_{i}_mask.jpg")
                        
                except Exception as e:
                    print(f"推理样本 {i} 时出错: {e}")
                    continue
        
        # 保存推理结果到JSON文件
        if results:
            step_dir = self.inference_dir / f"step_{self.global_step}"
            step_dir.mkdir(exist_ok=True)
            
            results_file = step_dir / "inference_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"推理测试完成，结果保存到: {results_file}")
            
            # 打印部分结果
            print("\n=== 推理结果预览 ===")
            for i, result in enumerate(results[:3]):  # 只显示前3个
                print(f"样本 {i+1}:")
                print(f"  真实文本: {result['ground_truth']}")
                print(f"  生成文本: {result['generated_text']}")
                print()
        
        self.model.train()  # 恢复训练模式

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            images = batch['orig_image']
            texts = batch['text']
            alpha_channels = batch['mask_image']
            
            # 使用processor处理图像和文本
            inputs = self.processor(
                images=images, 
                text=texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True, 
                max_length=self.config['model']['max_text_length']
            )
            
            # 处理alpha通道
            if alpha_channels is not None:
                alpha_tensors = []
                for alpha_img in alpha_channels:
                    alpha_processed = self.processor.image_processor(
                        alpha_img, 
                        return_tensors="pt"
                    )['pixel_values']
                    
                    # 如果是RGB图像，取第一个通道作为alpha
                    if alpha_processed.shape[1] == 3:
                        alpha_processed = alpha_processed[:, 0:1, :, :]
                    
                    alpha_tensors.append(alpha_processed)
                
                inputs['alpha_values'] = torch.cat(alpha_tensors, dim=0)
            
            # 移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            outputs = self.model(
                pixel_values=inputs['pixel_values'], 
                alpha_values=inputs.get('alpha_values'),
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                labels=inputs['input_ids']
            )
            
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config['training'].get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 每1000步运行一次推理测试
            if self.global_step % 1000 == 0:
                print(f"\n[Step {self.global_step}] 运行推理测试...")
                self.run_inference_test()
            
            # 更新进度条
            avg_loss = total_loss / num_batches
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })
        
        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch in progress_bar:
                images = batch['orig_image']
                texts = batch['text']
                alpha_channels = batch['mask_image']
                
                # 使用processor处理图像和文本
                inputs = self.processor(
                    images=images, 
                    text=texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True, 
                    max_length=self.config['model']['max_text_length']
                )
                
                # 处理alpha通道
                if alpha_channels is not None:
                    alpha_tensors = []
                    for alpha_img in alpha_channels:
                        alpha_processed = self.processor.image_processor(
                            alpha_img, 
                            return_tensors="pt"
                        )['pixel_values']
                        
                        # 如果是RGB图像，取第一个通道作为alpha
                        if alpha_processed.shape[1] == 3:
                            alpha_processed = alpha_processed[:, 0:1, :, :]
                        
                        alpha_tensors.append(alpha_processed)
                    
                    inputs['alpha_values'] = torch.cat(alpha_tensors, dim=0)
                
                # 移动到设备
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                outputs = self.model(
                    pixel_values=inputs['pixel_values'], 
                    alpha_values=inputs.get('alpha_values'),
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'], 
                    labels=inputs['input_ids']
                )
                
                loss = outputs.loss.item()
                total_loss += loss
                num_batches += 1
                
                # 更新验证进度条
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'val_loss': f'{loss:.4f}',
                    'avg_val_loss': f'{avg_loss:.4f}'
                })
        
        return total_loss / num_batches

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch, 'global_step': self.global_step, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss, 'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
            print(f"保存最佳模型")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        print(f"加载检查点: Epoch {self.epoch}, Loss {self.best_loss:.4f}")

    def train(self):
        print("开始训练...")
        
        # 加载数据集
        self.load_datasets()
        
        # 设置模型
        self.setup_model()
        
        # 设置优化器和调度器
        training_config = self.config['training']
        if training_config['alpha_only']:
            params = [p for name, p in self.model.named_parameters() if p.requires_grad and 'alpha' in name.lower()]
        else:
            params = self.model.parameters()
            
        self.optimizer = optim.AdamW(params, lr=training_config['learning_rate'], 
                                   weight_decay=training_config['weight_decay'])
        
        total_steps = len(self.train_loader) * training_config['num_epochs']
        warmup_steps = int(total_steps * training_config['warmup_ratio'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        
        print(f"优化器设置完成: LR={training_config['learning_rate']}, 总步数={total_steps}")
        
        # 加载检查点（如果指定）
        if self.config['training'].get('resume_from_checkpoint'):
            self.load_checkpoint(self.config['training']['resume_from_checkpoint'])
        
        # 训练循环
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % self.config['training']['save_steps'] == 0:
                self.save_checkpoint(is_best)
        
        print("训练完成！")

def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        if config_path:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return DEFAULT_CONFIG.copy()

def main():
    parser = argparse.ArgumentParser(description='Alpha BLIP Training')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.resume:
        config['training']['resume_from_checkpoint'] = args.resume
    
    trainer = AlphaBlipTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()