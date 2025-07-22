import cv2
import torch
import numpy as np
from PIL import Image
from datasets.Machado import process_single_image
from datasets.CCPR import CCPRCalculator
from datasets.difference_map import process_single_image_pair
from models.Alpha_BLIP.inference import AlphaBlipSingleInference
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class OWLViTDetector:
    def __init__(self, model_name="google/owlvit-base-patch32"):
        """
        Initialize OWL-ViT model for zero-shot object detection
        Available OWL-ViT models:
        - "google/owlvit-base-patch32" (base model)
        - "google/owlvit-base-patch16" (higher resolution)
        - "google/owlvit-large-patch14" (larger model, better accuracy)
        - "google/owlv2-base-patch16-ensemble" (OWLv2 - improved version)
        """
        print(f"Loading OWL-ViT model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        print("✅ OWL-ViT model loaded successfully")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
    def detect_objects_with_text(self, image, text_prompt, score_threshold=0.4):
        """
        Detect objects in image using OWL-ViT with text prompt
        
        Args:
            image: numpy array (RGB format) or PIL Image
            text_prompt: text description for what to detect (can be multiple objects separated by commas)
            score_threshold: confidence threshold
            
        Returns:
            detections: list of detection results
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        # Split text prompt into individual queries for better detection
        # OWL-ViT works better with specific object names rather than full sentences
        queries = [query.strip() for query in text_prompt.split(',')]
        if len(queries) == 1:
            # 如果是完整句子，直接使用原文本
            queries = [text_prompt]
        
        print(f"Using detection queries: {queries}")
        
        # Process image and text
        inputs = self.processor(
            text=queries, 
            images=pil_image, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Process outputs - handle the deprecation warning
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        
        # Use the new method if available, otherwise fall back to the old one
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )[0]
        except AttributeError:
            # Fallback to deprecated method
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )[0]
        
        detections = []
        # Extract detections
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            
            # Convert label to text
            if isinstance(label, torch.Tensor):
                label_text = queries[label.item()] if label.item() < len(queries) else f"object_{label.item()}"
            else:
                label_text = str(label)
            
            detections.append({
                "score": round(score.item(), 3),
                "label": label_text,
                "box": box,
                "text_query": text_prompt
            })
            
        return detections

class ImageProcessor:
    def __init__(self, original_image, cvd_type, cvd_level, ccpr_activate=False):

        self.original_image = original_image
        self.cvd_type = int(cvd_type)
        self.cvd_level = int(cvd_level)
        self.ccpr_activate = ccpr_activate  

        self.ccpr_calculator = CCPRCalculator()


    def return_processed_images(self):
        print(f"Processing CVD simulation with type: {self.cvd_type}, level: {self.cvd_level}")
        print(f"Original image shape: {self.original_image.shape}")
        
        simulated_image = process_single_image(self.original_image, self.cvd_type, self.cvd_level)
        print(f"Simulated image result: {type(simulated_image)}")
        if simulated_image is not None:
            print(f"Simulated image shape: {simulated_image.shape}")
        else:
            print("Warning: simulated_image is None")
            return None
        
        if self.ccpr_activate:
            ccpr_value, is_cvd_indistinguishable = self.ccpr_calculator.calculate_ccpr(self.original_image, simulated_image)
            print(f"CCPR value: {ccpr_value}, is_cvd_indistinguishable: {is_cvd_indistinguishable}")
            if is_cvd_indistinguishable:
                diff_map = process_single_image_pair(self.original_image, simulated_image)
                print(f"Difference map generated: {type(diff_map)}")
                if diff_map is not None:
                    print(f"Difference map shape: {diff_map.shape}")
                return diff_map
            else:
                print("CVD simulation not indistinguishable enough")
                return None
        else:
            diff_map = process_single_image_pair(self.original_image, simulated_image)
            print(f"Difference map generated: {type(diff_map)}")
            if diff_map is not None:
                print(f"Difference map shape: {diff_map.shape}")
            return diff_map

def process_pipeline_with_owlvit(original_image_path, cvd_types, cvd_levels, 
                                output_path="detection_result.jpg", score_thr=0.4,
                                owlvit_model_name="google/owlvit-base-patch32"):
    """
    Process pipeline using OWL-ViT model for zero-shot object detection
    """
    import os
    
    # Check if file exists and provide helpful error message
    if not os.path.exists(original_image_path):
        print(f"Error: Image file not found: {original_image_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Available files in current directory:")
        for item in os.listdir('.'):
            if item.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"  - {item}")
        
        # Try some common alternative paths
        alternative_paths = [
            os.path.join(os.getcwd(), original_image_path),
            os.path.join(os.getcwd(), os.path.basename(original_image_path)),
            original_image_path.replace('\\', '/'),  # Windows path fix
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found image at alternative path: {alt_path}")
                original_image_path = alt_path
                break
        else:
            raise ValueError(f"Can't find image file: {original_image_path}")
    
    # Read and process original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Can't read {original_image_path} - file may be corrupted or not a valid image")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    print(f"Successfully loaded image: {original_image_path}")
    print(f"Image shape: {original_image.shape}")
    
    # Process CVD simulation
    difference_map = ImageProcessor(original_image, cvd_types, cvd_levels).return_processed_images()
    if difference_map is None:
        print("The image is not suitable for CVD simulation or indistinguishable.")
        return
    
    # Generate caption using Alpha-BLIP
    print("Generating caption with Alpha-BLIP...")
    lm_generator = AlphaBlipSingleInference()
    caption = lm_generator.generate_caption(image=original_image, mask_image=difference_map)
    print(f"Generated caption: '{caption}'")
    
    # Initialize OWL-ViT model
    print("Initializing OWL-ViT detector...")
    owl_detector = OWLViTDetector(model_name=owlvit_model_name)
    
    # Detect objects using the generated caption as prompt
    print("Running object detection...")
    detections = owl_detector.detect_objects_with_text(
        original_image, 
        text_prompt=caption, 
        score_threshold=score_thr
    )
    
    print(f"\nDetection Results using prompt: '{caption}'")
    print(f"Found {len(detections)} objects:")
    for i, detection in enumerate(detections, 1):
        print(f"  {i}. Label: {detection['label']}, Score: {detection['score']:.3f}, Box: {detection['box']}")
    
    return detections

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CVD Pipeline with OWL-ViT")
    parser.add_argument('--image_path', type=str, required=True, help='path to the input image')
    parser.add_argument('--cvd_types', type=int, default=0,
                        help='type of CVD to simulate')
    parser.add_argument('--ccpr_activate', action='store_false', 
                        help='activate CCPR calculation')
    parser.add_argument('--cvd_levels', type=int, default=60, help='CVD simulation level (0-100)')
    parser.add_argument('--output_path', type=str, default='detection_result.jpg',
                        help='output image path')
    parser.add_argument('--score_thr', type=float, default=0.4, help='detection threshold')
    parser.add_argument('--owlvit_model', type=str, default='google/owlvit-base-patch32',
                        choices=['google/owlvit-base-patch32', 'google/owlvit-base-patch16', 
                               'google/owlvit-large-patch14', 'google/owlv2-base-patch16-ensemble'],
                        help='OWL-ViT model variant to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CVD Pipeline with OWL-ViT Object Detection")
    print("="*60)
    
    process_pipeline_with_owlvit(
        original_image_path=args.image_path,
        cvd_types=args.cvd_types,
        cvd_levels=args.cvd_levels,
        output_path=args.output_path,
        score_thr=args.score_thr,
        owlvit_model_name=args.owlvit_model
    )

if __name__ == '__main__':
    main()