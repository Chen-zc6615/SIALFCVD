import sys
import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import re

# 分别导入transformers组件，避免导入问题
from transformers import BlipProcessor, BlipForConditionalGeneration
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    KOSMOS2_AVAILABLE = True
except ImportError:
    print("⚠️  Kosmos2相关包导入失败")
    KOSMOS2_AVAILABLE = False

from data.datasets_loader_evaluator import VisualGenomeDataset, RefCOCOPlusDataset
from model_blip.inference import AlphaBlipInference
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import sacrebleu

class SimpleEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 创建保存目录
        self.save_dir = "test_images"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"图像保存目录: {self.save_dir}")
        
        # 初始化模型
        self.init_models()
        
        # 初始化评估指标
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
    
    def init_models(self):
        """初始化所有模型"""
        print("正在加载模型...")
        
        # 1. BLIP模型
        print("  - BLIP...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float32
        )
        self.blip_model.to(self.device).eval()
        
        # 2. AlphaBlip模型
        print("  - AlphaBlip...")
        self.alphablip = AlphaBlipInference(
            checkpoint_path="/home/chenzc/cvd/model_blip/outputs/alpha_blip_training/checkpoints/best_model.pt"
        )
        
        # 3. Kosmos2模型 
        self.kosmos2_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos2_model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos2_model.to(self.device).eval()

        # 4. GLaMM
        self.GlaMM_model = AutoModelForVision2Seq.from_pretrained("MBZUAI/GLaMM-FullScope").to(self.device).eval()
        self.GlaMM_processor = AutoProcessor.from_pretrained("MBZUAI/GLaMM-FullScope")
    
    def save_sample_image(self, image, mask, bbox, sample_idx, dataset_name, text):
        """保存样本图像，包括原图、mask、bbox可视化"""
        try:
            # 保存原图
            orig_path = os.path.join(self.save_dir, f"{dataset_name}_{sample_idx:03d}_original.jpg")
            image.save(orig_path)
            
            # 如果有bbox，绘制bbox
            if bbox is not None and len(bbox) == 4:
                img_with_bbox = image.copy()
                draw = ImageDraw.Draw(img_with_bbox)
                
                # 获取图像尺寸
                img_width, img_height = image.size
                
                # 检查bbox是否已归一化
                x1, y1, x2, y2 = bbox
                if max(bbox) <= 1.0:  # 已归一化
                    x1 = int(x1 * img_width)
                    y1 = int(y1 * img_height)
                    x2 = int(x2 * img_width)
                    y2 = int(y2 * img_height)
                
                # 绘制bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # 添加文本标签
                try:
                    # 尝试使用默认字体
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # 绘制文本背景
                text_bbox = draw.textbbox((x1, y1-25), f"Sample {sample_idx}", font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1-25), f"Sample {sample_idx}", fill="white", font=font)
                
                bbox_path = os.path.join(self.save_dir, f"{dataset_name}_{sample_idx:03d}_with_bbox.jpg")
                img_with_bbox.save(bbox_path)
            
            # 如果有mask，保存mask
            if mask is not None:
                mask_path = os.path.join(self.save_dir, f"{dataset_name}_{sample_idx:03d}_mask.jpg")
                if hasattr(mask, 'save'):  # PIL Image
                    mask.save(mask_path)
                elif isinstance(mask, np.ndarray):  # numpy array
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_img.save(mask_path)
            
            # 保存文本信息
            txt_path = os.path.join(self.save_dir, f"{dataset_name}_{sample_idx:03d}_info.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Sample Index: {sample_idx}\n")
                f.write(f"Image Size: {image.size}\n")
                f.write(f"BBox: {bbox}\n")
                f.write(f"Reference Text: {text}\n")
            
            print(f"    💾 已保存样本{sample_idx}的图像到 {self.save_dir}/")
            
        except Exception as e:
            print(f"    ⚠️  保存图像失败: {str(e)}")
    
    def blip_inference(self, image):
        """BLIP推理"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=20, num_beams=3)
        result = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return result
    
    def alphablip_inference(self, image, mask):
        """AlphaBlip推理"""
        result = self.alphablip.generate_caption(image, mask, max_length=20, num_beams=3)
        return result
    
    def kosmos2_inference(self, image, bbox):
        prompt = "<grounding><phrase> It</phrase>is"
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_tuple = tuple(float(x) for x in bbox)

            bboxes = [bbox_tuple]  # 单个bbox的列表
        else:
            print(f"    ⚠️  bbox格式错误: {bbox}")
            return ""
        
        inputs = self.kosmos2_processor(
            text=prompt,
            images=image,
            bboxes=bboxes,  # 使用正确格式的bbox
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.kosmos2_model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
        
        generated_text = self.kosmos2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text, entities = self.kosmos2_processor.post_process_generation(generated_text)
        
        return processed_text
    
    def glamm_inference(self, image, bbox):
 
        
        # 验证bbox输入
        if bbox is None or len(bbox) != 4:
            raise ValueError("bbox is required and must be a list of 4 coordinates [x1, y1, x2, y2]")
        
        # GLaMM的标准区域描述提示
        prompt = "<image>Can you provide me with a detailed description of the region in the picture marked by <bbox>?"
        
        # 获取图像尺寸
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
        
        # 按照GLaMM的标准方式处理边界框
        # 步骤1: 缩放到336×336 (CLIP编码器输入尺寸)
        x_scale, y_scale = 336 / width, 336 / height
        bbox_scaled = [
            bbox[0] * x_scale,  # x1
            bbox[1] * y_scale,  # y1
            bbox[2] * x_scale,  # x2
            bbox[3] * y_scale   # y2
        ]
        
        # 步骤2: 归一化到[0,1]范围
        height_sc, width_sc = (336, 336)
        norm_bbox = np.array(bbox_scaled) / np.array([width_sc, height_sc, width_sc, height_sc])
        

        inputs = self.GlaMM_processor(
            text=prompt,
            images=image,
            bboxes=[norm_bbox.tolist()],  # 传递归一化后的边界框
            return_tensors="pt"
        )

        
        # 移动到设备
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # 生成文本
        with torch.no_grad():
            generated_ids = self.GlaMM_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=3,
                temperature=0.7,
                do_sample=False,
            )
        
        # 解码生成的文本
        generated_text = self.GlaMM_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # 清理输出文本
        # 移除输入提示
        if prompt in generated_text:
            result = generated_text.replace(prompt, "").strip()
        else:
            # 如果直接替换失败，尝试分割
            parts = generated_text.split("ASSISTANT:")
            if len(parts) > 1:
                result = parts[-1].strip()
            else:
                result = generated_text.strip()
        
        # 进一步清理
        # 移除HTML标签和特殊token
        result = re.sub(r'<.*?>', '', result)
        result = result.replace('[SEG]', '')
        result = result.replace('\n', ' ').replace('  ', ' ')
        result = ' '.join(result.split()).strip()
        
        return result


    def load_dataset(self, dataset_name, max_samples=100):
        """加载数据集"""
        print(f"加载 {dataset_name} 数据集...")
        
        if dataset_name == "VisualGenome":
            dataset = VisualGenomeDataset(
                region_descriptions_file="/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/region_descriptions.json",
                image_data_file="/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/image_data.json",
                images_dir="/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_100K_all",
                image_size=(384, 384)
            )
            mask_key = 'ellipse_mask'  # AlphaBlip在VG上使用ellipse_mask
        else:  # RefCOCOPlus
            dataset = RefCOCOPlusDataset(
                refs_file="/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/refs(unc).p",
                instances_file="/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/instances.json",
                images_dir="/home/chenzc/cvd/model_blip/data/data/COCO/train2014",
                image_size=(384, 384)
            )
            mask_key = 'gray_mask'  # AlphaBlip在RefCOCO+上使用gray_mask
        
        # 采样数据
        total_samples = len(dataset)
        if max_samples and max_samples < total_samples:
            import random
            random.seed(42)
            indices = random.sample(range(total_samples), max_samples)
        else:
            indices = range(total_samples)
        
        # 提取样本
        samples = []
        for i in indices:
            try:
                sample = dataset[i]
                samples.append({
                    'image': sample['orig_image'],
                    'mask': sample.get(mask_key, None),
                    'bbox': sample.get('normalized_bbox', None),
                    'text': sample['text'][0] if isinstance(sample['text'], list) else sample['text']
                })
            except:
                continue
        
        print(f"  成功加载 {len(samples)} 个样本")
        return samples
    
    def run_inference(self, samples, dataset_name):
        """运行推理"""
        print(f"开始推理 {dataset_name}...")
        
        results = {
            'blip': [],
            'alphablip': [],
            'kosmos2': [],
            'references': []
        }
        
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:  # 更频繁的进度显示
                print(f"  进度: {i+1}/{len(samples)}")
            
            # 保存前几个样本的图像（可以调整保存数量）
            if i < 10:  # 保存前10个样本
                self.save_sample_image(
                    sample['image'], 
                    sample['mask'], 
                    sample['bbox'], 
                    i+1, 
                    dataset_name, 
                    sample['text']
                )
            
            # BLIP推理 (只用原图)
            blip_pred = self.blip_inference(sample['image'])
            results['blip'].append(blip_pred)
            
            # AlphaBlip推理 (用原图+mask)
            alphablip_pred = self.alphablip_inference(sample['image'], sample['mask'])
            results['alphablip'].append(alphablip_pred)
            
            # Kosmos2推理 (用原图+bbox)
            kosmos2_pred = self.kosmos2_inference(sample['image'], sample['bbox'])
            results['kosmos2'].append(kosmos2_pred)
            
            # 参考文本
            results['references'].append(sample['text'])
            
            # 显示前几个样本的结果
            if i < 3:
                print(f"    样本{i+1}:")
                print(f"      参考: {sample['text']}")
                print(f"      BLIP: {blip_pred}")
                print(f"      AlphaBlip: {alphablip_pred}")
                print(f"      Kosmos2: {kosmos2_pred}")
                print(f"      Kosmos2(清理后): {self.clean_prediction(kosmos2_pred)}")
                print()
        
        return results
    
    def clean_prediction(self, text):
        """清理预测文本，去掉固定前缀"""
        if not text:
            return ""
        
        # 转换为小写进行匹配
        text_lower = text.lower().strip()
        
        # 定义要去掉的前缀模式
        prefixes_to_remove = [
            "it is",
            "it's",
            "this is", 
            "there is",
            "there are",
            "the image shows",
            "the image depicts"
        ]
        
        # 去掉匹配的前缀
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                # 去掉前缀，保留原始大小写
                remaining = text[len(prefix):].strip()
                # 如果去掉前缀后还有内容，返回剩余部分
                if remaining:
                    return remaining
                break
        
        return text.strip()
    
    def calculate_metrics(self, predictions, references):
        """计算评估指标"""
        if not predictions:
            return {}
        
        # 清理预测文本，去掉固定前缀
        cleaned_predictions = [self.clean_prediction(pred) for pred in predictions]
        
        # 过滤掉空的预测
        valid_pairs = [(pred, ref) for pred, ref in zip(cleaned_predictions, references) if pred.strip()]
        
        if not valid_pairs:
            print("    ⚠️  所有预测都为空，无法计算指标")
            return {}
        
        cleaned_preds, valid_refs = zip(*valid_pairs)
        
        # BLEU
        bleu = sacrebleu.corpus_bleu(cleaned_preds, [valid_refs])
        
        # METEOR
        meteor_scores = []
        for pred, ref in zip(cleaned_preds, valid_refs):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            if pred_tokens:  # 确保预测不为空
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
        
        # ROUGE-L
        rouge_scores = []
        for pred, ref in zip(cleaned_preds, valid_refs):
            if pred.strip():  # 确保预测不为空
                scores = self.rouge_scorer.score(ref.lower(), pred.lower())
                rouge_scores.append(scores['rougeL'].fmeasure)
        
        # CIDEr
        gts = {i: [ref.lower()] for i, ref in enumerate(valid_refs)}
        res = {i: [pred.lower()] for i, pred in enumerate(cleaned_preds)}
        cider_score, _ = self.cider_scorer.compute_score(gts, res)
        
        return {
            'BLEU-1': bleu.precisions[0],
            'BLEU-2': bleu.precisions[1],
            'METEOR': np.mean(meteor_scores) if meteor_scores else 0.0,
            'ROUGE-L': np.mean(rouge_scores) if rouge_scores else 0.0,
            'CIDEr': cider_score
        }
    
    def evaluate_dataset(self, dataset_name, max_samples=1000):
        """评估单个数据集"""
        # 加载数据
        samples = self.load_dataset(dataset_name, max_samples)
        
        # 推理
        results = self.run_inference(samples, dataset_name)
        
        # 计算指标
        metrics = {}
        for model_name in ['blip', 'alphablip', 'kosmos2']:
            metrics[model_name] = self.calculate_metrics(
                results[model_name], results['references']
            )
        return metrics
    
    def run_evaluation(self, max_samples=100):
        """运行完整评估"""
        all_results = {}
        
        # 评估VisualGenome
        print("\n" + "="*50)
        all_results['VisualGenome'] = self.evaluate_dataset('VisualGenome', max_samples)
        
        # 评估RefCOCOPlus
        print("\n" + "="*50)
        all_results['RefCOCOPlus'] = self.evaluate_dataset('RefCOCOPlus', max_samples)
        
        # 打印结果
        self.print_results(all_results)
        
        return all_results
    
    def print_results(self, all_results):
        """打印结果表格"""
        print("\n" + "="*100)
        print("📊 评估结果")
        print("="*100)
        
        # 表头
        print(f"{'模型':<12}", end="")
        for dataset in ['visual_genome', 'refcoco']:
            print(f"{dataset:^50}", end="")
        print()
        
        print(f"{'':12}", end="")
        for _ in range(2):
            print(f"{'bleu-1':<10}{'bleu-2':<10}{'meteor':<10}{'rouge-l':<10}{'cider':<10}", end="")
        print()
        print("-" * 112)
        
        # 数据行
        models = ['blip', 'alphablip','kosmos2']
        
        for model in models:
            print(f"{model:<12}", end="")
            
            for dataset in ['VisualGenome', 'RefCOCOPlus']:
                if dataset in all_results and model in all_results[dataset]:
                    m = all_results[dataset][model]
                    print(f"{m['BLEU-1']:<10.3f}{m['BLEU-2']:<10.3f}{m['METEOR']:<10.3f}"
                          f"{m['ROUGE-L']:<10.3f}{m['CIDEr']:<10.3f}", end="")
                else:
                    print(f"{'':50}", end="")
            print()
        
        print("="*100)

def main():
    """主函数"""
    print("🚀 简化版模型评估")
    print("="*50)
    
    # 创建评估器
    evaluator = SimpleEvaluator()
    
    # 运行评估 (每个数据集1000个样本)
    results = evaluator.run_evaluation(max_samples=1000)
    
    print("\n✅ 评估完成!")
    print(f"📁 测试图像已保存到: {evaluator.save_dir}/")

if __name__ == "__main__":
    main()