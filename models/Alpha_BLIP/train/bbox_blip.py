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
import re
import sys
sys.path.append('/home/chenzc/cvd/model_blip')  # 添加你的项目路径
from data.datasets_loader import VisualGenomeDataset, DualImageTransform  # 修改为实际路径

def format_bbox_text(bbox, original_size, processed_size):
    """
    将bbox格式化为归一化的xyxy文本格式
    Args:
        bbox: [x, y, width, height] - COCO格式
        original_size: (width, height) - 原始图片尺寸
        processed_size: (width, height) - processor处理后的图片尺寸
    Returns:
        str: 格式化的bbox文本，如 "<xyxy>[0.123,0.456,0.789,0.012]</xyxy>"
    """
    x, y, w, h = bbox  # COCO格式: x,y,width,height
    orig_w, orig_h = original_size
    proc_w, proc_h = processed_size
    x1, y1, x2, y2 = x, y, x + w, y + h
    
    x_scale = proc_w / orig_w
    y_scale = proc_h / orig_h
    
    x1_scaled = x1 * x_scale
    y1_scaled = y1 * y_scale
    x2_scaled = x2 * x_scale
    y2_scaled = y2 * y_scale

    x1_norm = x1_scaled / proc_w
    y1_norm = y1_scaled / proc_h
    x2_norm = x2_scaled / proc_w
    y2_norm = y2_scaled / proc_h
    
    # 保留3位小数
    return f"<xyxy>[{x1_norm:.3f},{y1_norm:.3f},{x2_norm:.3f},{y2_norm:.3f}]</xyxy>"

class VGBlipDataset(Dataset):
    """改进的BLIP数据集，正确处理输入和标签"""
    def __init__(self, 
                 region_descriptions_file,
                 image_data_file,
                 images_dir,
                 processor,
                 max_length=128
                 ):
        
        self.processor = processor
        self.max_length = max_length
        
        self.vg_dataset = VisualGenomeDataset(
            region_descriptions_file,
            image_data_file, 
            images_dir
        )
        
        # 使用全部数据作为训练集
        total_size = len(self.vg_dataset)
        self.train_indices = list(range(total_size))
        self.val_indices = []  # 不使用验证集
        
        print(f"Using full dataset for training: {len(self.train_indices)} samples")
    
    def __len__(self):
        return len(self.vg_dataset)
    
    def get_train_indices(self):
        return self.train_indices
    
    def get_val_indices(self):
        return self.val_indices
    
    def __getitem__(self, idx):
        # 从VG数据集获取样本
        vg_sample = self.vg_dataset[idx]
        
        orig_image_pil = vg_sample['orig_image']
        target_text = vg_sample['text']
        bbox = vg_sample['bbox']
        
        # 获取原始图片尺寸
        original_size = orig_image_pil.size
        
        # 先用processor处理图像获取尺寸
        temp_processor_output = self.processor(
            orig_image_pil,
            return_tensors="pt",
            do_rescale=False
        )
        
        processed_pixel_values = temp_processor_output['pixel_values']
        processed_height = processed_pixel_values.shape[2]
        processed_width = processed_pixel_values.shape[3]
        processed_size = (processed_width, processed_height)
        
        # 创建bbox文本
        bbox_text = format_bbox_text(bbox, original_size, processed_size)
        
        # 构建完整的输入序列：bbox + 描述
        full_sequence = f"{bbox_text} {target_text}"
        
        # 处理图像和完整序列
        inputs = self.processor(
            orig_image_pil,
            full_sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            do_rescale=False
        )
        
        # 创建labels，需要mask掉bbox部分
        # 首先获取bbox部分的长度
        bbox_inputs = self.processor.tokenizer(
            bbox_text,
            return_tensors="pt",
            add_special_tokens=False  # 不添加特殊token
        )
        bbox_length = bbox_inputs['input_ids'].shape[1]
        
        # 创建labels，bbox部分设为-100（忽略损失）
        labels = inputs['input_ids'].clone()
        
        # 找到第一个非padding token的位置
        first_token_pos = 1 if inputs['input_ids'][0, 0] == self.processor.tokenizer.bos_token_id else 0
        
        # mask掉bbox部分（从第一个token开始的bbox_length个token）
        labels[0, first_token_pos:first_token_pos + bbox_length] = -100
        
        # 如果有padding，也要mask掉
        attention_mask = inputs['attention_mask']
        labels[attention_mask == 0] = -100
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'bbox_text': bbox_text,  # 用于测试时参考
            'target_text': target_text  # 用于测试时参考
        }

class VGBlipTrainer:
    """改进的VG数据集BLIP训练器"""
    def __init__(self, 
                 model_name="Salesforce/blip-image-captioning-base",
                 vg_region_descriptions_file=None,
                 vg_image_data_file=None,
                 vg_images_dir=None,
                 output_dir="./blip_vg_finetuned",
                 max_length=128
                 ):
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # 获取图像处理器的默认尺寸
        if hasattr(self.processor, 'image_processor'):
            img_proc = self.processor.image_processor
            if hasattr(img_proc, 'size'):
                self.default_image_size = img_proc.size
                print(f"BLIP processor default image size: {self.default_image_size}")
            else:
                self.default_image_size = {"height": 384, "width": 384}
                print(f"Using default BLIP image size: {self.default_image_size}")
        else:
            self.default_image_size = {"height": 384, "width": 384}
            print(f"Using fallback image size: {self.default_image_size}")

        # 添加特殊token
        special_tokens = ["<xyxy>", "</xyxy>"]
        self.processor.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        print(f"Added xyxy bbox tokens, vocabulary size: {len(self.processor.tokenizer)}")
        
        # 创建数据集
        if vg_region_descriptions_file and vg_image_data_file and vg_images_dir:
            print("Creating VG dataset...")
            self.dataset = VGBlipDataset(
                vg_region_descriptions_file,
                vg_image_data_file,
                vg_images_dir,
                self.processor,
                max_length=max_length
            )
            self.train_indices = self.dataset.get_train_indices()
            self.val_indices = self.dataset.get_val_indices()
        else:
            self.dataset = None
            self.train_indices = []
            self.val_indices = []
    
    def create_dataloader(self, batch_size=8, num_workers=4):
        """创建训练数据加载器（使用全部数据）"""
        if self.dataset is None:
            raise ValueError("Dataset not initialized")
        
        print(f"Using full dataset for training: {len(self.dataset)} samples")
        
        train_loader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader
    
    def run_test_on_samples(self, device, num_samples=8):
        """从训练集抽取样本进行测试"""
        self.model.eval()
        
        # 从训练集中随机选择样本
        test_sample_indices = random.sample(self.train_indices, min(num_samples, len(self.train_indices)))
        test_results = []
        
        print(f"\n{'='*80}")
        print(f"TESTING ON {num_samples} RANDOM TRAINING SAMPLES")
        print(f"{'='*80}")
        
        with torch.no_grad():
            for i, idx in enumerate(test_sample_indices, 1):
                vg_sample = self.dataset.vg_dataset[idx]
                orig_image_pil = vg_sample['orig_image']
                ground_truth = vg_sample['text']
                bbox = vg_sample['bbox']
                
                original_size = orig_image_pil.size
                processed_size = (self.default_image_size["width"], self.default_image_size["height"])
                
                # 构建输入prompt（只有bbox）
                bbox_text = format_bbox_text(bbox, original_size, processed_size)
                
                # 使用bbox作为prompt生成
                inputs = self.processor(
                    orig_image_pil, 
                    bbox_text,
                    return_tensors="pt",
                    do_rescale=False
                ).to(device)
                
                # 生成描述
                generated_ids = self.model.generate(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=30,
                    num_beams=3,
                    do_sample=False,
                    early_stopping=True
                )
                
                # 解码时跳过输入部分
                input_length = inputs['input_ids'].shape[1]
                generated_text = self.processor.decode(
                    generated_ids[0][input_length:], 
                    skip_special_tokens=True
                ).strip()
                
                # 如果生成为空，尝试完整解码并清理
                if not generated_text:
                    full_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    generated_text = re.sub(r'<xyxy>\[.*?\]</xyxy>', '', full_text).strip()
                
                # 立即打印结果
                print(f"\n--- Sample {i} (Dataset Index: {idx}) ---")
                print(f"BBox (COCO format): {bbox}")
                print(f"BBox Text: {bbox_text}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Generated:    {generated_text}")
                print("-" * 60)
                
                test_results.append({
                    'sample_idx': idx,
                    'bbox': bbox,
                    'bbox_text': bbox_text,
                    'ground_truth': ground_truth,
                    'generated_text': generated_text
                })
        
        print(f"{'='*80}\n")
        self.model.train()
        return test_results
    
    def print_test_results(self, test_results, step):
        """打印测试结果摘要"""
        print(f"\n{'='*80}")
        print(f"TEST RESULTS SUMMARY AT STEP {step}")
        print(f"{'='*80}")
        print(f"Total samples tested: {len(test_results)}")
        print(f"Average generated length: {sum(len(r['generated_text'].split()) for r in test_results) / len(test_results):.1f} words")
        print(f"{'='*80}\n")
    
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
              num_epochs=3,
              batch_size=8,
              learning_rate=1e-5,
              warmup_steps=200,
              test_steps=100,
              max_grad_norm=1.0):
        """训练模型（无早停版本）"""
        
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
        print(f"Training samples: {len(self.train_indices)}")
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
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Train Loss: {avg_train_loss:.4f}")
            print(f"  Total Steps: {global_step}")
            
            # 保存每个epoch的模型
            self.save_model(f"model_epoch_{epoch+1}")
        
        # 保存最终模型
        self.save_model("final_model")
        
        avg_final_loss = total_train_loss / len(train_loader)
        print(f"\nTraining completed! Final average loss: {avg_final_loss:.4f}")
        return avg_final_loss
    
    def save_model(self, checkpoint_name):
        """保存模型"""
        save_path = Path(self.output_dir) / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, checkpoint_path):
        """加载模型"""
        self.model = BlipForConditionalGeneration.from_pretrained(checkpoint_path)
        self.processor = BlipProcessor.from_pretrained(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    
    def test_model(self, image_path, bbox=None, prompt="Describe this region:"):
        """测试模型生成能力"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 添加bbox到prompt
        if bbox:
            processed_size = (self.default_image_size["width"], self.default_image_size["height"])
            bbox_text = format_bbox_text(bbox, original_size, processed_size)
            full_prompt = f"{bbox_text} {prompt}"
            
            print(f"Original image size: {original_size}")
            print(f"Processed image size: {processed_size}")
            print(f"Original bbox (COCO): {bbox}")
            print(f"Formatted bbox text: {bbox_text}")
        else:
            full_prompt = prompt
        
        # 处理输入
        inputs = self.processor(image, full_prompt, return_tensors="pt", do_rescale=False).to(device)
        
        # 生成文本
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=30,
                num_beams=4,
                do_sample=True,
            )
        
        # 解码生成的文本
        if bbox:
            # 跳过输入部分
            input_length = inputs['input_ids'].shape[1]
            generated_text = self.processor.decode(
                generated_ids[0][input_length:], 
                skip_special_tokens=True
            ).strip()
        else:
            generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text

    def test_random_samples(self, device, num_samples=8):
        """独立的测试函数，只测试不保存"""
        print(f"\n🔍 Testing {num_samples} random samples from training set...")
        test_results = self.run_test_on_samples(device, num_samples)
        return test_results

# 使用示例
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 配置路径
    config = {
        'vg_region_descriptions_file': "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_train_105077/region_descriptions.json",
        'vg_image_data_file': "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_train_105077/image_data.json", 
        'vg_images_dir': "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_100K_all",
        'output_dir': "./blip_vg_finetuned_v2",
        'model_name': "Salesforce/blip-image-captioning-base"
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 创建训练器
    trainer = VGBlipTrainer(
        model_name=config['model_name'],
        vg_region_descriptions_file=config['vg_region_descriptions_file'],
        vg_image_data_file=config['vg_image_data_file'],
        vg_images_dir=config['vg_images_dir'],
        output_dir=config['output_dir'],
        max_length=128
    )
    
    # 开始训练
    print("Starting training...")
    final_loss = trainer.train(
        num_epochs=1,  # 只训练一个epoch
        batch_size=16,  # 根据GPU内存调整
        learning_rate=2e-5,
        test_steps=200  # 每200步测试一次
    )
    
    print(f"Training completed with final loss: {final_loss:.4f}")
    
    # 最终测试
    print("Running final test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_test_results = trainer.test_random_samples(device, num_samples=10)