import argparse
import os
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import BlipProcessor
from tqdm import tqdm

from model.pretrain import create_alpha_blip_pretrain_model
from model_blip.data.datasets_loader import CVDDataset, VisualGenomeDataset
from torch.utils.data import ConcatDataset

def get_lr_schedule(optimizer, global_step, config, total_steps_per_epoch):
    """统一的学习率调度函数"""
    warmup_steps = config['warmup_steps']
    
    # Warmup阶段
    if global_step < warmup_steps:
        lr = config['warmup_lr'] + (config['init_lr'] - config['warmup_lr']) * global_step / warmup_steps
    else:
        # Warmup之后的epoch-based decay
        effective_epoch = global_step // total_steps_per_epoch
        lr = max(config['init_lr'] * (config['lr_decay_rate'] ** effective_epoch), config['min_lr'])
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(model, data_loader, optimizer, epoch, device, config):
    """简化版训练函数"""
    model.train()
    
    # 简单的统计变量
    total_loss_ita = 0
    total_loss_itm = 0  
    total_loss_lm = 0
    total_loss = 0
    num_batches = len(data_loader)
    
    # 创建进度条
    progress_bar = tqdm(
        enumerate(data_loader), 
        total=num_batches,
        desc=f'Epoch {epoch+1}/{config["max_epoch"]}',
        ncols=120
    )

    for i, batch in progress_bar:
        # 计算全局步数
        global_step = epoch * num_batches + i
        
        # 统一的学习率调度
        current_lr = get_lr_schedule(optimizer, global_step, config, num_batches)
        
        optimizer.zero_grad()
        
        try:
            # 准备数据
            image = batch['orig_image'].to(device)
            caption = batch['text']  # 文本列表
            alpha_image = batch['mask_image'].to(device)
            
            # Alpha权重的渐进增加
            alpha_weight = config.get('alpha_weight', 0.4)
            alpha_weight = alpha_weight * min(1, global_step / (2 * num_batches))

            # 前向传播
            loss_ita, loss_itm, loss_lm = model(
                image=image, 
                caption=caption, 
                alpha_values=alpha_image,
                alpha_weight=alpha_weight
            )  
            loss = loss_ita + loss_itm + loss_lm  

            # 反向传播
            loss.backward()
            optimizer.step()    

            # 统计
            total_loss_ita += loss_ita.item()
            total_loss_itm += loss_itm.item()
            total_loss_lm += loss_lm.item()
            total_loss += loss.item()
            
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
        
        # 更新进度条（每10步更新一次）
        if i % 10 == 0:
            avg_ita = total_loss_ita / (i + 1)
            avg_itm = total_loss_itm / (i + 1)
            avg_lm = total_loss_lm / (i + 1)
            
            progress_bar.set_postfix({
                'ITA': f"{avg_ita:.4f}",
                'ITM': f"{avg_itm:.4f}",
                'LM': f"{avg_lm:.4f}",
                'LR': f"{current_lr:.2e}",
                'Alpha': f"{alpha_weight:.3f}",
                'Step': f"{global_step}"
            })
        
        # 定期清理GPU缓存
        if i % 100 == 0:
            torch.cuda.empty_cache()

    progress_bar.close()
    
    # 计算平均值
    avg_loss_ita = total_loss_ita / num_batches
    avg_loss_itm = total_loss_itm / num_batches
    avg_loss_lm = total_loss_lm / num_batches
    avg_total_loss = total_loss / num_batches
    
    print(f"Epoch {epoch+1} 完成:")
    print(f"  平均 ITA Loss: {avg_loss_ita:.4f}")
    print(f"  平均 ITM Loss: {avg_loss_itm:.4f}")
    print(f"  平均 LM Loss: {avg_loss_lm:.4f}")
    print(f"  平均总损失: {avg_total_loss:.4f}")
    
    return {
        'loss_ita': f"{avg_loss_ita:.4f}",
        'loss_itm': f"{avg_loss_itm:.4f}",
        'loss_lm': f"{avg_loss_lm:.4f}",
        'total_loss': f"{avg_total_loss:.4f}",
        'lr': f"{current_lr:.6f}"
    }

def validate_config(config):
    """验证配置参数"""
    required_keys = ['batch_size', 'init_lr', 'max_epoch', 'warmup_steps']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if config['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    if config['warmup_steps'] < 0:
        raise ValueError("warmup_steps must be non-negative")

def check_dataset_paths(config):
    """检查数据集路径是否存在"""
    for dataset_name in ['cvd_dataset', 'visual_genome_dataset']:
        if config.get(dataset_name, {}).get('enabled', False):
            dataset_config = config[dataset_name]
            for key, path in dataset_config.items():
                if ('file' in key or 'dir' in key) and isinstance(path, str):
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"{dataset_name} {key}: {path} not found")

def main(args, config):
    # 验证配置
    validate_config(config)
    check_dataset_paths(config)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset...")
    
    datasets = []
    
    # CVD数据集
    if config.get('cvd_dataset', {}).get('enabled', False):
        print("Loading CVD dataset...")
        cvd_config = config['cvd_dataset']
        cvd_dataset = CVDDataset(
            image_dir=cvd_config['image_dir'],
            annotations_file=cvd_config['annotations_file']
        )
        datasets.append(cvd_dataset)
        print(f"CVD dataset size: {len(cvd_dataset)}")
    
    # Visual Genome数据集
    if config.get('visual_genome_dataset', {}).get('enabled', False):
        print("Loading Visual Genome dataset...")
        vg_config = config['visual_genome_dataset']
        vg_dataset = VisualGenomeDataset(
            region_descriptions_file=vg_config['region_descriptions_file'],
            image_data_file=vg_config['image_data_file'],
            images_dir=vg_config['images_dir'],
            mask_type=vg_config.get('mask_type', 'ellipse'),
            min_region_size=vg_config.get('min_region_size', 1000)
        )
        datasets.append(vg_dataset)
        print(f"Visual Genome dataset size: {len(vg_dataset)}")
    
    # 组合数据集
    if len(datasets) == 1:
        final_dataset = datasets[0]
    elif len(datasets) > 1:
        final_dataset = ConcatDataset(datasets)
    else:
        raise ValueError("没有启用任何数据集！请在配置文件中启用CVD或Visual Genome数据集。")
    
    print(f'Total training samples: {len(final_dataset)}')

    # 创建数据加载器
    data_loader = DataLoader(
        final_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 避免最后一个batch大小不一致
    )
    
    print(f'Batches per epoch: {len(data_loader)}')

    #### Model #### 
    print("Creating model...")
    model = create_alpha_blip_pretrain_model(
        pretrained_model_name=config.get('pretrained_model_name', "Salesforce/blip-image-captioning-base"),
        embed_dim=config.get('embed_dim', 256),
        queue_size=config.get('queue_size', 57600),
        momentum=config.get('momentum', 0.995)
    )
    
    tokenizer = model.tokenizer
    model = model.to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=config['warmup_lr'],  # 从warmup学习率开始
        weight_decay=config['weight_decay']
    )
    
    start_epoch = 0
    if args.checkpoint:    
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1                
        print(f'Resumed from checkpoint, starting at epoch {start_epoch}')    
        
    print("Start training...")
    start_time = time.time()
    
    # 整体训练进度条
    epoch_progress = tqdm(
        range(start_epoch, config['max_epoch']),
        desc="Training Progress",
        position=0,
        leave=True,
        ncols=100
    )
    
    for epoch in epoch_progress:
        train_stats = train(model, data_loader, optimizer, epoch, device, config) 
        
        # 更新整体进度条信息
        epoch_progress.set_postfix({
            'Epoch': f'{epoch+1}/{config["max_epoch"]}',
            'Loss': train_stats['total_loss'],
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # 记录日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }
        
        # 保存检查点
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        
        # 每5个epoch或最后一个epoch保存检查点
        if epoch % 5 == 0 or epoch == config['max_epoch'] - 1:
            torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth'))
        
        # 总是保存最新的检查点
        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_latest.pth'))
        
        # 写入日志文件
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    epoch_progress.close()
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training completed! Total time: {total_time_str}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # 配置参数
    config = {
        # 训练参数
        'batch_size': 8,
        'init_lr': 1e-4,
        'max_epoch': 10,
        'warmup_steps': 2000,  # 建议设置为总步数的5-10%
        'warmup_lr': 1e-6,     # 更小的warmup起始学习率
        'weight_decay': 0.05,
        'min_lr': 1e-6,
        'lr_decay_rate': 0.95,  # 稍微缓和一些的衰减
        'alpha_weight': 0.4,
        
        # CVD数据集配置
        'cvd_dataset': {
            'enabled': False,
            'image_dir': '/home/chenzc/cvd/Places365_with_cvd_7_20K_4o',
            'annotations_file': '/home/chenzc/cvd/Place365CVDdata.json'
        },
        
        # Visual Genome数据集配置
        'visual_genome_dataset': {
            'enabled': True,
            'region_descriptions_file': '/home/chenzc/cvd/visual_genome/region_descriptions.json',
            'image_data_file': '/home/chenzc/cvd/visual_genome/image_data.json',
            'images_dir': '/home/chenzc/cvd/visual_genome/VG_100K_all',
            'mask_type': 'ellipse',
            'min_region_size': 1000
        },
        
        # Alpha BLIP模型配置
        'pretrained_model_name': 'Salesforce/blip-image-captioning-base',
        'embed_dim': 256,
        'queue_size': 57600,
        'momentum': 0.995
    }

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        main(args, config)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()