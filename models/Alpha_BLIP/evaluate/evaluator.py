import sys
import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq, AutoModel
from data.datasets_loader_evaluator import VisualGenomeDataset, RefCOCOPlusDataset
from model_blip.inference import AlphaBlipInference
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import sacrebleu

def normalize_bbox(bbox, image_size, target_size=None, debug=False):
    if bbox is None:
        if debug:
            print(f"    ⚠️  bbox为None")
        return None
    
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        if debug:
            print(f"    ⚠️  bbox格式错误: {bbox}")
        return None
    
    x, y, w, h = bbox
    img_width, img_height = image_size
    
    if debug:
        print(f"    🔍 调试信息:")
        print(f"        原始bbox: [{x}, {y}, {w}, {h}]")
        print(f"        图像尺寸: {img_width} x {img_height}")
        print(f"        bbox范围: x=[{x}, {x+w}], y=[{y}, {y+h}]")
    
    # 检查原始bbox是否超出边界
    if x + w > img_width or y + h > img_height:
        if debug:
            print(f"    ⚠️  原始bbox超出图像边界!")
            print(f"        x+w={x+w} > img_width={img_width}: {x+w > img_width}")
            print(f"        y+h={y+h} > img_height={img_height}: {y+h > img_height}")
    
    if w <= 0 or h <= 0:
        if debug:
            print(f"    ⚠️  bbox宽高无效: w={w}, h={h}")
        return None
    
    # 转换为x1, y1, x2, y2格式
    x1, y1, x2, y2 = x, y, x + w, y + h
    
    # 如果指定了目标尺寸，先进行缩放
    if target_size is not None:
        target_w, target_h = target_size
        x_scale = target_w / img_width
        y_scale = target_h / img_height
        
        if debug:
            print(f"        缩放比例: x_scale={x_scale:.6f}, y_scale={y_scale:.6f}")
        
        x1_old, y1_old, x2_old, y2_old = x1, y1, x2, y2
        x1 = x1 * x_scale
        y1 = y1 * y_scale
        x2 = x2 * x_scale
        y2 = y2 * y_scale
        
        if debug:
            print(f"        缩放前: ({x1_old}, {y1_old}, {x2_old}, {y2_old})")
            print(f"        缩放后: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
        
        # 更新参考尺寸为目标尺寸
        img_width, img_height = target_w, target_h
    
    # 归一化到[0,1]范围
    x1_norm = x1 / img_width
    y1_norm = y1 / img_height
    x2_norm = x2 / img_width
    y2_norm = y2 / img_height
    
    if debug:
        print(f"        归一化后: ({x1_norm:.6f}, {y1_norm:.6f}, {x2_norm:.6f}, {y2_norm:.6f})")
    
    # 检查是否等于1.0（边界情况）
    if x1_norm >= 1.0 or y1_norm >= 1.0 or x2_norm > 1.0 or y2_norm > 1.0:
        if debug:
            print(f"    ⚠️  归一化后超出[0,1]范围!")
    
    # 处理边界情况：如果恰好等于1.0，稍微调整
    epsilon = 1e-6
    if x2_norm >= 1.0:
        x2_norm = 1.0 - epsilon
        if debug:
            print(f"        调整x2_norm: {x2_norm:.6f}")
    if y2_norm >= 1.0:
        y2_norm = 1.0 - epsilon
        if debug:
            print(f"        调整y2_norm: {y2_norm:.6f}")
    
    # 确保 x2 > x1 和 y2 > y1
    if x2_norm <= x1_norm or y2_norm <= y1_norm:
        if debug:
            print(f"    ⚠️  最终bbox坐标无效: x1={x1_norm:.6f}, y1={y1_norm:.6f}, x2={x2_norm:.6f}, y2={y2_norm:.6f}")
        return None
    
    return (x1_norm, y1_norm, x2_norm, y2_norm)

def format_bbox_text(bbox, original_size, processed_size):
    """使用统一的归一化函数格式化bbox文本"""
    normalized_coords = normalize_bbox(bbox, original_size, processed_size)
    
    if normalized_coords is None:
        return None
    
    x1_norm, y1_norm, x2_norm, y2_norm = normalized_coords
    return f"<xyxy>[{x1_norm:.3f},{y1_norm:.3f},{x2_norm:.3f},{y2_norm:.3f}]</xyxy>"

class SimpleEvaluator:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.init_model(model_name)

        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
    
    def init_model(self, model_name):
        print(f"正在加载模型: {model_name}")
        
        if model_name == "BLIP":
            print("  - BLIP...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base", torch_dtype=torch.float32
            )
            self.blip_model.to(self.device).eval()

        elif model_name == "BboxBLIP":
            print("  - BboxBLIP...")
            self.bbox_blip_processor = BlipProcessor.from_pretrained("/home/chenzc/cvd/model_blip/blip_vg_finetuned_v2/final_model")
            self.bbox_blip_model = BlipForConditionalGeneration.from_pretrained(
                "/home/chenzc/cvd/model_blip/blip_vg_finetuned_v2/final_model", torch_dtype=torch.float32
            )
            self.bbox_blip_model.to(self.device).eval()
            self.bbox_blip_image_size = {"height": 384, "width": 384}
        
        elif model_name == "AlphaBLIP":
            print("  - AlphaBLIP...")
            self.alphablip = AlphaBlipInference(
                checkpoint_path="/home/chenzc/cvd/model_blip/outputs/alpha_blip_training/checkpoints/best_model.pt"
            )
        
        elif model_name == "Kosmos-2":
            print("  - Kosmos-2...")
            self.kosmos2_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
            self.kosmos2_model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
            self.kosmos2_model.to(self.device).eval()
            self.kosmos2_image_size = {"height": 224, "width": 224}

        elif model_name == "GLaMM":
            print("  - GLaMM...")
            self.glamm_model = AutoModel.from_pretrained("MBZUAI/GLaMM-RegCap-VG").to(self.device).eval()
            self.glamm_processor = AutoProcessor.from_pretrained("MBZUAI/GLaMM-RegCap-VG")

        else:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: BLIP, BboxBLIP, AlphaBLIP, Kosmos-2, GLaMM")

    def run_inference(self, image, mask=None, bbox=None):
        if self.model_name == "BLIP":
            return self.blip_inference(image)
        elif self.model_name == "BboxBLIP":
            return self.bbox_blip_inference(image, bbox)
        elif self.model_name == "AlphaBLIP":
            return self.alphablip_inference(image, mask)
        elif self.model_name == "Kosmos-2":
            return self.kosmos2_inference(image, bbox)
        elif self.model_name == "GLaMM":
            return self.glamm_inference(image, mask)
    
    def blip_inference(self, image):
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=20, num_beams=3)
        result = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return result
    
    def bbox_blip_inference(self, image, bbox):
        # 添加bbox检查
        if bbox is None:
            return ""
            
        original_size = image.size
        processed_size = (
            self.bbox_blip_image_size["width"], 
            self.bbox_blip_image_size["height"]
        )
        
        bbox_text = format_bbox_text(bbox, original_size, processed_size)
        
        if bbox_text is None:
            return ""
        
        inputs = self.bbox_blip_processor(
            image,
            bbox_text,
            return_tensors="pt",
            do_rescale=False
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.bbox_blip_model.generate(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=30,
                num_beams=3,
                do_sample=False,
                early_stopping=True
            )

        input_length = inputs['input_ids'].shape[1]
        generated_text = self.bbox_blip_processor.decode(
            generated_ids[0][input_length:], 
            skip_special_tokens=True
        ).strip()
        
        if not generated_text:
            import re
            full_text = self.bbox_blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            generated_text = re.sub(r'<xyxy>\[.*?\]</xyxy>', '', full_text).strip()
        
        return generated_text
    
    def alphablip_inference(self, image, mask):
        result = self.alphablip.generate_caption(image, mask, max_length=20, num_beams=3)
        return result
    
    def kosmos2_inference(self, image, bbox):
        prompt = "<grounding><phrase> It </phrase> is"
        
        if not hasattr(image, 'size'):
            return ""
        
        # 使用统一的归一化函数
        normalized_coords = normalize_bbox(bbox, image.size)
        
        if normalized_coords is None:
            return ""
        
        x1_norm, y1_norm, x2_norm, y2_norm = normalized_coords
        
        # 确保bbox是浮点数元组格式
        bbox_tuple = (float(x1_norm), float(y1_norm), float(x2_norm), float(y2_norm))
        bboxes = [bbox_tuple]  # 单个示例的bbox列表

        
        try:
            inputs = self.kosmos2_processor(
                text=prompt,           # 单个文本字符串
                images=image,          # 单个图像
                bboxes=bboxes,         # 包含单个bbox元组的列表
                return_tensors="pt"
            )
            
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
            
        except Exception as e:
            return ""
    
    def glamm_inference(self, image, mask):
        instruction = "Can you provide me with a detailed description of the region in the picture marked by <bbox>?"
        instruction = instruction.replace('&lt;', '<').replace('&gt;', '>')

        if mask is not None:
            if isinstance(mask, np.ndarray):
                if mask.dtype != np.uint8:
                    mask = (mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask).convert('L')
            else:
                mask_img = mask.convert('L') if mask.mode != 'L' else mask
            
            prompt = "<image><mask>Describe what you see in the highlighted region."
            
            inputs = self.glamm_processor(
                text=prompt,
                images=image,
                masks=mask_img,
                return_tensors="pt"
            )
        else:
            prompt = "<image>Describe what you see in this image."
            inputs = self.glamm_processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.glamm_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                temperature=0.7,
                do_sample=False
            )
        
        generated_text = self.glamm_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        if prompt in generated_text:
            result = generated_text.replace(prompt, "").strip()
        else:
            result = generated_text.strip()
        
        return result

    def load_dataset(self, dataset_name, max_samples=None):
        print(f"加载 {dataset_name} 数据集...")
        
        if dataset_name == "VisualGenome":
            dataset = VisualGenomeDataset(
                region_descriptions_file="/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/region_descriptions.json",
                image_data_file="/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/image_data.json",
                images_dir="/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_100K_all"
            )
            mask_key = 'ellipse_mask'
        else:
            dataset = RefCOCOPlusDataset(
                refs_file="/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/refs(unc).p",
                instances_file="/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/instances.json",
                images_dir="/home/chenzc/cvd/model_blip/data/data/COCO/train2014"
            )
            mask_key = 'gray_mask'
        
        total_samples = len(dataset)
        if max_samples and max_samples < total_samples:
            import random
            random.seed(42)
            indices = random.sample(range(total_samples), max_samples)
        else:
            indices = range(total_samples)
        
        samples = []
        for i in indices:
            sample = dataset[i]
            
            bbox = sample.get('bbox', None)
            if bbox is None:
                continue
                
            samples.append({
                'image': sample['orig_image'],
                'mask': sample.get(mask_key, None),
                'bbox': bbox,
                'text': sample['text'][0] if isinstance(sample['text'], list) else sample['text']
            })
        
        print(f"  成功加载 {len(samples)} 个样本")
        return samples
    
    def run_dataset_inference(self, samples, dataset_name):
        print(f"开始推理 {dataset_name} (模型: {self.model_name})...")
        
        predictions = []
        references = []
        
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"  处理进度: {i + 1}/{len(samples)}")
                
            pred = self.run_inference(
                sample['image'], 
                mask=sample['mask'], 
                bbox=sample['bbox']
            )
            
            predictions.append(pred)
            references.append(sample['text'])
        
        return predictions, references
    
    def clean_prediction(self, text):
        if not text:
            return ""
        
        text_lower = text.lower().strip()
        
        prefixes_to_remove = [
            "it is",
            "it's",
            "this is",
            "there is",
            "there are",
            "the image shows",
            "the image depicts"
        ]
        
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                remaining = text[len(prefix):].strip()
                if remaining:
                    return remaining
                break
        
        return text.strip()
    
    def calculate_metrics(self, predictions, references):
        if not predictions:
            return {}
        
        cleaned_predictions = [self.clean_prediction(pred) for pred in predictions]
        
        valid_pairs = [(pred, ref) for pred, ref in zip(cleaned_predictions, references) if pred.strip()]
        
        if not valid_pairs:
            print("    ⚠️  所有预测都为空，无法计算指标")
            return {}
        
        cleaned_preds, valid_refs = zip(*valid_pairs)
        
        bleu = sacrebleu.corpus_bleu(cleaned_preds, [valid_refs])
        
        meteor_scores = []
        for pred, ref in zip(cleaned_preds, valid_refs):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            if pred_tokens:
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
        
        rouge_scores = []
        for pred, ref in zip(cleaned_preds, valid_refs):
            if pred.strip():
                scores = self.rouge_scorer.score(ref.lower(), pred.lower())
                rouge_scores.append(scores['rougeL'].fmeasure)
        
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
        samples = self.load_dataset(dataset_name, max_samples)
        predictions, references = self.run_dataset_inference(samples, dataset_name)
        metrics = self.calculate_metrics(predictions, references)
        return metrics
    
    def run_evaluation(self, max_samples=100):
        all_results = {}
        
        print("\n" + "="*50)
        print(f"评估模型: {self.model_name}")
        print("="*50)
        
        all_results['VisualGenome'] = self.evaluate_dataset('VisualGenome', max_samples)
        all_results['RefCOCOPlus'] = self.evaluate_dataset('RefCOCOPlus', max_samples)
        
        self.print_results(all_results)
        
        return all_results
    
    def print_results(self, all_results):
        print("\n" + "="*80)
        print(f"📊 {self.model_name} 评估结果")
        print("="*80)
        
        print(f"{'数据集':<15}{'BLEU-1':<10}{'BLEU-2':<10}{'METEOR':<10}{'ROUGE-L':<10}{'CIDEr':<10}")
        print("-" * 75)
        
        for dataset_name, metrics in all_results.items():
            if metrics:
                print(f"{dataset_name:<15}{metrics['BLEU-1']:<10.3f}{metrics['BLEU-2']:<10.3f}"
                      f"{metrics['METEOR']:<10.3f}{metrics['ROUGE-L']:<10.3f}{metrics['CIDEr']:<10.3f}")
            else:
                print(f"{dataset_name:<15}{'无有效结果'}")
        
        print("="*75)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='单模型评估')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['BLIP', 'BboxBLIP', 'AlphaBLIP', 'Kosmos-2', 'GLaMM'],
                       help='要评估的模型名称')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='每个数据集的最大样本数')
    
    args = parser.parse_args()
    
    print(f"🚀 开始评估模型: {args.model}")
    print("="*50)
    
    evaluator = SimpleEvaluator(model_name=args.model)
    results = evaluator.run_evaluation(max_samples=args.max_samples)
    
    print(f"\n✅ {args.model} 评估完成!")

if __name__ == "__main__":
    main()