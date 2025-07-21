import sys
import os
# 添加项目根目录到Python路径
sys.path.append('/home/chenzc/cvd/model_blip')
sys.path.append('/home/chenzc/cvd/model_blip/evaluate')

import re
import json
from typing import List, Dict, Tuple, Optional, Callable, Union
import numpy as np
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import sacrebleu
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader

# 使用 HuggingFace transformers - BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor
# kosmos
from transformers import AutoProcessor, AutoModelForVision2Seq

# 导入您的数据集
from data.datasets_loader_evaluator import VisualGenomeDataset, RefCOCOPlusDataset

from model_blip.inference import AlphaBlipInference

class BLIPCaptioner:
    """使用 HuggingFace BLIP 模型的图像描述生成器"""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        """
        初始化 BLIP 模型
        
        Args:
            model_name: HuggingFace BLIP 模型名称
            device: 设备 (cuda/cpu)，如果为None则自动选择
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        self.model.to(self.device)
        self.model.eval()
    

    def inference(self, image, max_length=20, num_beams=3):
        """
        对图像生成描述
        
        Args:
            image: PIL Image
            max_length: 生成文本的最大长度
            num_beams: beam search的beam数量
            
        Returns:
            str: 生成的文本
        """
        try:
            # 处理输入 - 只传入图像，不使用文本prompt
            inputs = self.processor(image, return_tensors="pt")
            
            # 将输入移到正确的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成文本 - 只使用beam search
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            
            # 解码生成的文本
            generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"推理出错: {e}")
            return ""
    
class Kosmos2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.model.to(self.device)
        self.model.eval()  
    
    def inference(self, image, bbox, prompt="<grounding>What is<phrase> this object</phrase>?"):
        inputs = self.processor(
            text=prompt, 
            images=image, 
            bboxes=bbox,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():  
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text = self.processor.post_process_generation(generated_text)
        
        return processed_text


nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class TextMetrics:
    """文本评估指标计算器"""
    
    def __init__(self):
        """初始化文本指标计算器"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
        self.smoothing = SmoothingFunction().method1 
    
    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        text = self.preprocess_text(text)
        tokens = text.split()
        return tokens
    
    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict:
        refs_transposed = []
        max_refs = max(len(refs) for refs in references) if references else 1
        
        for i in range(max_refs):
            ref_list = [refs[i] if i < len(refs) else refs[0] for refs in references]
            refs_transposed.append(ref_list)
        
        bleu = sacrebleu.corpus_bleu(predictions, refs_transposed)
        
        return {
            'BLEU-1': bleu.precisions[0],
            'BLEU-2': bleu.precisions[1], 
            'BLEU-3': bleu.precisions[2],
            'BLEU-4': bleu.score,
        }
    
    def compute_meteor(self, predictions: List[str], references: List[List[str]]) -> float:
        meteor_scores = []
        
        for pred, refs in zip(predictions, references):
            pred_tokens = self.tokenize(pred)
            
            ref_scores = []
            for ref in refs:
                ref_tokens = self.tokenize(ref)
                score = meteor_score([ref_tokens], pred_tokens)
                ref_scores.append(score)
             
            meteor_scores.append(max(ref_scores) if ref_scores else 0.0)
        
        return np.mean(meteor_scores)
    
    def compute_rouge_l(self, predictions: List[str], references: List[List[str]]) -> float:
        """计算ROUGE-L分数"""
        rouge_scores = []
        
        for pred, refs in zip(predictions, references):
            pred_clean = self.preprocess_text(pred)
            
            ref_scores = []
            for ref in refs:
                ref_clean = self.preprocess_text(ref)
                scores = self.rouge_scorer.score(ref_clean, pred_clean)
                ref_scores.append(scores['rougeL'].fmeasure)
            rouge_scores.append(max(ref_scores) if ref_scores else 0.0)
        
        return np.mean(rouge_scores)
    
    def compute_cider(self, predictions: List[str], references: List[List[str]]) -> float:
        gts = {}
        res = {}
        
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            pred_clean = self.preprocess_text(pred)
            refs_clean = [self.preprocess_text(ref) for ref in refs]
            
            gts[i] = refs_clean
            res[i] = [pred_clean]
        
        score, scores = self.cider_scorer.compute_score(gts, res)
        return score

class ModelComparisionEvaluator:
    def __init__(self, config):
        self.config = config
        
        # 初始化标准 BLIP 模型
        self.blip_caption = BLIPCaptioner(
            model_name=getattr(config, 'base_model', "Salesforce/blip-image-captioning-base"),
            device=getattr(config, 'device', None)
        )
        
        self.alpha_blip_caption = AlphaBlipInference(
            checkpoint_path=getattr(config, 'checkpoint_path', None))   
 
        # 初始化Kosmos2模型（如果需要的话）
        self.kosmos2 = None
        if getattr(config, 'use_kosmos2', False):
            try:
                self.kosmos2 = Kosmos2()
                print("✅ Kosmos2 模型加载成功")
            except Exception as e:
                print(f"❌ 加载 Kosmos2 模型失败: {e}")
        
        self.metrics = TextMetrics()
        
        # 定义每个模型使用的mask类型
        self.model_mask_config = getattr(config, 'model_mask_config', {
            'blip': {
                'VisualGenome': None,  # BLIP不使用mask，只用原图
                'RefCOCOPlus': None
            },
            'alphablip': {
                'VisualGenome': 'ellipse_mask',  # AlphaBlip在VG上使用ellipse_mask
                'RefCOCOPlus': 'gray_mask'       # AlphaBlip在RefCOCO+上使用gray_mask
            }
        })
        
        # 如果配置中包含Kosmos2，添加到模型配置中
        if self.kosmos2:
            if 'kosmos2' not in self.model_mask_config:
                self.model_mask_config['kosmos2'] = {
                    'VisualGenome': None,  # Kosmos2使用bbox信息，不需要mask
                    'RefCOCOPlus': None
                }

        self.samples = {}
        self.texts = {}
        self.predictions = {}
        self.results = {}

    def load_datasets(self):
        """加载数据集"""
        datasets = {}
        
        try:
            vg_dataset = VisualGenomeDataset(
                region_descriptions_file=self.config.vg_region_descriptions_file,
                image_data_file=self.config.vg_image_data_file,
                images_dir=self.config.vg_images_dir,
                image_size=(384, 384)
            )
            datasets['VisualGenome'] = vg_dataset
            print(f"✅ VisualGenome 数据集加载成功")
        except Exception as e:
            print(f"❌ 加载 VisualGenome 数据集失败: {e}")
        
        try:
            refcoco_dataset = RefCOCOPlusDataset(
                refs_file=self.config.refs_file,
                instances_file=self.config.instances_file,
                images_dir=self.config.images_dir,
                image_size=(384, 384)
            )
            datasets['RefCOCOPlus'] = refcoco_dataset
            print(f"✅ RefCOCOPlus 数据集加载成功")
        except Exception as e:
            print(f"❌ 加载 RefCOCOPlus 数据集失败: {e}")
        
        return datasets
    
    def extract_samples_from_dataset(self, dataset, dataset_name):
        """从数据集中提取样本 - 保存所有可用的mask类型"""
        samples = []
        texts = []
        
        total_samples = len(dataset)
        
        # 获取最大样本数限制
        max_samples = getattr(self.config, 'max_samples_per_dataset', None)
        
        # 如果设置了采样限制，则随机采样
        if max_samples is not None and total_samples > max_samples:
            print(f"📊 {dataset_name} 数据集总样本数: {total_samples}, 随机采样: {max_samples} 个")
            import random
            random.seed(42)  # 设置随机种子保证可重复性
            sample_indices = random.sample(range(total_samples), max_samples)
            sample_indices.sort()  # 排序以保持某种顺序
        else:
            print(f"📊 {dataset_name} 数据集使用全部样本: {total_samples} 个")
            sample_indices = range(total_samples)
        
        for idx, i in enumerate(sample_indices):
            try:
                sample = dataset[i]
                text = sample["text"]
                
                if isinstance(text, list):
                    text = text[0] if text else ""
                
                texts.append(text)
                
                # 保存原始样本和所有可用的mask
                sample_data = {
                    'index': i,
                    'sample_idx': idx,  # 采样后的索引
                    'orig_image': sample['orig_image'],
                    'text': text,
                    'dataset_name': dataset_name,
                    'original_sample': sample
                }
                
                # 根据数据集类型保存所有可用的mask和其他信息
                if dataset_name == "VisualGenome":
                    sample_data['available_masks'] = {
                        'bbox_mask': sample.get('bbox_mask', None),
                        'ellipse_mask': sample.get('ellipse_mask', None)
                    }
                    # 保存bbox信息（用于Kosmos2等需要bbox的模型）
                    sample_data['bbox'] = sample.get('normalized_bbox', None)
                    
                elif dataset_name == "RefCOCOPlus":
                    sample_data['available_masks'] = {
                        'binary_mask': sample.get('binary_mask', None),
                        'gray_mask': sample.get('gray_mask', None)
                    }
                    # 保存bbox信息（用于Kosmos2等需要bbox的模型）
                    sample_data['bbox'] = sample.get('normalized_bbox', None)
                
                samples.append(sample_data)
                    
            except Exception as e:
                print(f"  ⚠️  处理样本 {i} 时出错: {e}")
                continue
        
        print(f"✅ {dataset_name} 数据集成功提取: {len(samples)} 个样本")
        return samples, texts

    def get_mask_for_model(self, sample, model_name, dataset_name):
        """根据模型和数据集获取对应的mask"""
        if model_name not in self.model_mask_config:
            return None
        
        mask_type = self.model_mask_config[model_name].get(dataset_name, None)
        
        if mask_type is None:
            return None
        
        available_masks = sample.get('available_masks', {})
        return available_masks.get(mask_type, None)

    def load_and_extract_datasets(self):
        """加载数据集并提取样本"""
        datasets = self.load_datasets()
        
        if not datasets:
            return
        
        for dataset_name, dataset in datasets.items():
            try:
                samples, texts = self.extract_samples_from_dataset(dataset, dataset_name)
                
                if samples:
                    self.samples[dataset_name] = samples
                    self.texts[dataset_name] = texts
                
            except Exception as e:
                continue

    def run_inference_for_dataset(self, dataset_name):
        """为单个数据集运行推理"""
        if dataset_name not in self.samples:
            return
        
        samples = self.samples[dataset_name]
        
        # 初始化预测结果字典
        predictions_dict = {}
        
        print(f"\n🚀 开始对 {dataset_name} 数据集进行推理...")
        print(f"总样本数: {len(samples)}")
        
        # 打印模型配置信息
        print(f"📋 模型配置:")
        for model_name, dataset_config in self.model_mask_config.items():
            mask_type = dataset_config.get(dataset_name, None)
            mask_info = f"使用 {mask_type}" if mask_type else "不使用mask(仅原图)"
            print(f"  - {model_name}: {mask_info}")

        for idx, sample in enumerate(samples):
            try:
                orig_image = sample['orig_image']
                
                # 标准BLIP推理
                if 'blip' in self.model_mask_config:
                    if 'blip' not in predictions_dict:
                        predictions_dict['blip'] = []
                    
                    blip_pred = self.blip_caption.inference(
                        orig_image,
                        max_length=getattr(self.config, 'max_length', 50),
                        num_beams=getattr(self.config, 'num_beams', 5)
                    )
                    predictions_dict['blip'].append(blip_pred)
                
                # AlphaBlip推理
                if 'alphablip' in self.model_mask_config and self.alpha_blip_caption:
                    if 'alphablip' not in predictions_dict:
                        predictions_dict['alphablip'] = []
                    
                    alpha_blip_mask = self.get_mask_for_model(sample, 'alphablip', dataset_name)
                    alpha_blip_pred = self.alpha_blip_caption.generate_caption(
                        orig_image,
                        mask_image=alpha_blip_mask,
                        max_length=getattr(self.config, 'max_length', 20),
                        num_beams=getattr(self.config, 'num_beams', 3)
                    )
                    predictions_dict['alphablip'].append(alpha_blip_pred)
                
                # Kosmos2推理
                if 'kosmos2' in self.model_mask_config and self.kosmos2:
                    if 'kosmos2' not in predictions_dict:
                        predictions_dict['kosmos2'] = []
                    
                    bbox = sample.get('bbox', None)
                    if bbox:
                        kosmos2_pred = self.kosmos2.inference(orig_image, bbox)
                        predictions_dict['kosmos2'].append(kosmos2_pred)
                    else:
                        predictions_dict['kosmos2'].append("")
                
                # 每100个样本打印一次进度
                if (idx + 1) % 100 == 0:
                    print(f"  已处理: {idx + 1}/{len(samples)} 样本")
                    
            except Exception as e:
                print(f"  ⚠️  推理样本 {idx} 时出错: {e}")
                # 为所有模型添加空预测
                for model_name in self.model_mask_config.keys():
                    if model_name == 'blip' or \
                       (model_name == 'alphablip' and self.alpha_blip_caption) or \
                       (model_name == 'kosmos2' and self.kosmos2):
                        if model_name not in predictions_dict:
                            predictions_dict[model_name] = []
                        predictions_dict[model_name].append("")
                continue
        
        # 存储预测结果
        self.predictions[dataset_name] = predictions_dict
        
        print(f"✅ {dataset_name} 推理完成！")

    def run_evaluation(self):
        """运行完整的评估流程"""
        print("🔄 开始评估流程...")
        
        # 打印模型mask配置
        print("\n🎯 模型Mask配置:")
        for model_name, dataset_configs in self.model_mask_config.items():
            print(f"  {model_name}:")
            for dataset_name, mask_type in dataset_configs.items():
                mask_info = mask_type if mask_type else "无mask(仅原图)"
                print(f"    - {dataset_name}: {mask_info}")
        
        # 加载数据集并提取样本
        print("\n📂 加载数据集...")
        self.load_and_extract_datasets()
        
        if not self.samples:
            print("❌ 没有成功加载任何数据集！")
            return
        
        # 显示加载的数据集信息
        print(f"\n📊 成功加载 {len(self.samples)} 个数据集:")
        for dataset_name, samples in self.samples.items():
            print(f"  - {dataset_name}: {len(samples)} 个样本")
            
            # 显示该数据集可用的mask类型
            if samples:
                available_masks = samples[0].get('available_masks', {})
                mask_types = list(available_masks.keys())
                print(f"    可用mask类型: {mask_types}")
        
        # 为每个数据集运行推理
        print(f"\n🚀 开始推理阶段...")
        for dataset_name in self.samples.keys():
            try:
                self.run_inference_for_dataset(dataset_name)
            except Exception as e:
                print(f"❌ 处理数据集 {dataset_name} 时出错: {e}")
                continue
        
        print("\n✅ 所有推理完成！")
    
    def calculate_metrics_for_dataset(self, dataset_name):
        """为单个数据集计算指标"""
        if dataset_name not in self.predictions or dataset_name not in self.texts:
            return {}
        
        references = self.texts[dataset_name]
        predictions = self.predictions[dataset_name]
        
        ref_format = [[ref] for ref in references]
        
        dataset_results = {}
        
        for model_name, preds in predictions.items():
            try:
                bleu_scores = self.metrics.compute_bleu(preds, ref_format)
                meteor_score = self.metrics.compute_meteor(preds, ref_format)
                rouge_score = self.metrics.compute_rouge_l(preds, ref_format)
                cider_score = self.metrics.compute_cider(preds, ref_format)
                
                dataset_results[model_name] = {
                    'BLEU-1': bleu_scores['BLEU-1'],
                    'BLEU-2': bleu_scores['BLEU-2'],
                    'BLEU-3': bleu_scores['BLEU-3'],
                    'BLEU-4': bleu_scores['BLEU-4'],
                    'METEOR': meteor_score,
                    'ROUGE-L': rouge_score,
                    'CIDEr': cider_score
                }
                
            except Exception as e:
                print(f"❌ 计算 {model_name} 在 {dataset_name} 上的指标时出错: {e}")
                continue
        
        return dataset_results
    
    def calculate_all_metrics(self):
        """计算所有数据集的指标"""
        all_results = {}
        
        for dataset_name in self.predictions.keys():
            dataset_metrics = self.calculate_metrics_for_dataset(dataset_name)
            all_results[dataset_name] = dataset_metrics
        
        return all_results
    
    def print_results(self, all_results):
        """打印对比表格格式的结果"""
        print("\n" + "="*120)
        print("📊 多模型对比评估结果")
        print("="*120)
        
        # 打印mask使用配置
        print("🎯 模型Mask配置:")
        for model_name, dataset_configs in self.model_mask_config.items():
            print(f"  {model_name}:")
            for dataset_name, mask_type in dataset_configs.items():
                mask_info = mask_type if mask_type else "无mask(仅原图)"
                print(f"    - {dataset_name}: {mask_info}")
        print("-" * 120)
        
        # 获取所有数据集名称
        dataset_names = list(all_results.keys())
        
        if not dataset_names:
            print("❌ 没有评估结果可显示")
            return
        
        # 打印表头
        print(f"{'':15}", end="")
        for dataset_name in dataset_names:
            dataset_display = dataset_name.lower().replace('refcocoplus', 'refcoco').replace('visualgenome', 'visual_genome')
            print(f"{dataset_display:^50}", end="")
        print()
        
        print(f"{'模型':<15}", end="")
        for _ in dataset_names:
            print(f"{'bleu-1':<10}{'bleu-2':<10}{'meteor':<10}{'rouge-l':<10}{'cider':<10}", end="")
        print()
        print("-" * (15 + 50 * len(dataset_names)))
        
        # 获取所有模型名称并排序
        all_models = set()
        for dataset_results in all_results.values():
            all_models.update(dataset_results.keys())
        
        # 按顺序显示模型
        model_order = ['blip', 'alphablip', 'kosmos2']
        model_order = [m for m in model_order if m in all_models]
        
        # 添加其他模型
        for model in sorted(all_models):
            if model not in model_order:
                model_order.append(model)
        
        # 打印每个模型的结果
        for model_name in model_order:
            print(f"{model_name:<15}", end="")
            
            for dataset_name in dataset_names:
                if (dataset_name in all_results and 
                    model_name in all_results[dataset_name]):
                    metrics = all_results[dataset_name][model_name]
                    
                    bleu_1 = metrics.get('BLEU-1', 0.0)
                    bleu_2 = metrics.get('BLEU-2', 0.0)
                    meteor = metrics.get('METEOR', 0.0)
                    rouge_l = metrics.get('ROUGE-L', 0.0)
                    cider = metrics.get('CIDEr', 0.0)
                    
                    print(f"{bleu_1:<10.3f}{bleu_2:<10.3f}{meteor:<10.3f}{rouge_l:<10.3f}{cider:<10.3f}", end="")
                else:
                    # 如果该模型在该数据集上没有结果，显示空值
                    print(f"{'':10}{'':10}{'':10}{'':10}{'':10}", end="")
            print()
        
        print("="*120)
    
    def save_results(self, all_results, output_dir='results'):
        """保存所有结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型配置信息
        config_info = {
            'model_mask_config': self.model_mask_config,
            'evaluation_time': str(pd.Timestamp.now()),
            'config_summary': {
                'max_length': getattr(self.config, 'max_length', 50),
                'num_beams': getattr(self.config, 'num_beams', 5),
                'max_samples_per_dataset': getattr(self.config, 'max_samples_per_dataset', None)
            }
        }
        
        with open(f'{output_dir}/evaluation_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
        
        for dataset_name in self.predictions.keys():
            dataset_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            detailed_results = []
            samples = self.samples[dataset_name]
            references = self.texts[dataset_name]
            predictions = self.predictions[dataset_name]
            
            for i, (sample, ref) in enumerate(zip(samples, references)):
                item = {
                    'id': i,
                    'original_index': sample['index'],
                    'sample_index': sample.get('sample_idx', i),
                    'dataset_name': sample['dataset_name'],
                    'reference': ref,
                }
                
                # 添加模型预测结果和使用的mask信息
                for model_name, preds in predictions.items():
                    if i < len(preds):
                        item[f'{model_name}_prediction'] = preds[i]
                        # 记录该模型使用的mask类型
                        mask_type = self.model_mask_config.get(model_name, {}).get(dataset_name, None)
                        item[f'{model_name}_mask_type'] = mask_type if mask_type else "no_mask"
                    else:
                        item[f'{model_name}_prediction'] = ""
                        item[f'{model_name}_mask_type'] = "unknown"
                
                # 添加原始样本的其他信息
                original_sample = sample.get('original_sample', {})
                if 'ref_id' in original_sample:
                    item['ref_id'] = original_sample['ref_id']
                if 'category_id' in original_sample:
                    item['category_id'] = original_sample['category_id']
                if 'normalized_bbox' in original_sample:
                    item['normalized_bbox'] = original_sample['normalized_bbox']
                
                detailed_results.append(item)
            
            with open(f'{dataset_dir}/predictions.json', 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
            
            if dataset_name in all_results:
                with open(f'{dataset_dir}/metrics.json', 'w') as f:
                    json.dump(all_results[dataset_name], f, indent=2)
        
        with open(f'{output_dir}/all_metrics.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 结果已保存到: {output_dir}/")
        for dataset_name in self.predictions.keys():
            sample_count = len(self.samples[dataset_name])
            print(f"  - {dataset_name}: {sample_count} 个样本的结果")
        print(f"  - 评估配置已保存到: evaluation_config.json")


def main():
    """主函数"""
    
    print("=" * 80)
    print("🚀 多模型对比评估程序")
    print("=" * 80)
    
    # 配置类 - 支持多个模型
    class Config:
        # 基础模型配置
        base_model = "Salesforce/blip-image-captioning-base"  # 基础BLIP模型
        checkpoint_path = "/home/chenzc/cvd/model_blip/outputs/alpha_blip_training/checkpoints/best_model.pt"  # AlphaBlip检查点路径
        device = None  # 自动选择设备，或指定 "cuda" / "cpu"
        max_length = 50   # 生成文本的最大长度
        num_beams = 5     # beam search的beam数量
        
        # 是否使用Kosmos2模型
        use_kosmos2 = True  # 设为True以启用Kosmos2模型
        
        # 数据集采样配置
        max_samples_per_dataset = 100  # 每个数据集最大样本数，设为None则使用全部样本
        
        # VisualGenome 数据集配置
        vg_region_descriptions_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/region_descriptions.json"
        vg_image_data_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/image_data.json"
        vg_images_dir = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_100K_all"
        
        # RefCOCO+ 数据集配置
        refs_file = "/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/refs(unc).p"
        instances_file = "/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/instances.json"
        images_dir = "/home/chenzc/cvd/model_blip/data/data/COCO/train2014"
        
        # 模型mask配置 - 在这里定义每个模型使用什么mask
        model_mask_config = {
            'blip': {
                'VisualGenome': None,  # BLIP不使用mask，只用原图
                'RefCOCOPlus': None
            },
            'alphablip': {
                'VisualGenome': 'ellipse_mask',  # AlphaBlip在VG上使用ellipse_mask
                'RefCOCOPlus': 'gray_mask'       # AlphaBlip在RefCOCO+上使用gray_mask
            },
            # 如果启用Kosmos2，可以在这里配置
            'kosmos2': {
                'VisualGenome': None,   # Kosmos2使用bbox信息，不需要mask
                'RefCOCOPlus': None
            }
        }

    config = Config()
    
    print(f"📋 配置信息:")
    print(f"  - 基础模型: {config.base_model}")
    print(f"  - AlphaBlip检查点: {config.checkpoint_path}")
    print(f"  - 使用Kosmos2: {config.use_kosmos2}")
    print(f"  - 每个数据集最大样本数: {config.max_samples_per_dataset}")
    print(f"  - 最大生成长度: {config.max_length}")
    print(f"  - Beam数量: {config.num_beams}")
    
    print(f"\n🎯 模型Mask配置:")
    for model_name, dataset_configs in config.model_mask_config.items():
        print(f"  {model_name}:")
        for dataset_name, mask_type in dataset_configs.items():
            mask_info = mask_type if mask_type else "无mask(仅原图)"
            print(f"    - {dataset_name}: {mask_info}")
    
    # 创建评估器
    evaluator = ModelComparisionEvaluator(config)
    
    try:
        # 运行评估
        evaluator.run_evaluation()
        
        # 计算指标
        print("\n📊 计算评估指标...")
        all_results = evaluator.calculate_all_metrics()
        
        # 显示对比结果
        evaluator.print_results(all_results)
        
        # 保存结果
        evaluator.save_results(all_results)
        
        print("\n🎉 评估完成!")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()