import json
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utiliz.utiliz import set_seed
from PIL import Image
from scipy.spatial.distance import cdist
import random
import pickle
import os

class CVDDataset(Dataset):
    """CVD dataset loader - returns PIL Images"""
    def __init__(self, image_dir, annotations_file):
        self.image_dir = Path(image_dir)
                 
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        print(f"CVD dataset loaded: {len(self.annotations)} samples")
         
    def __len__(self):
        return len(self.annotations)
         
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        orig_image_path = self.image_dir / annotation["orig_image_file"]
        mask_image_path = self.image_dir / annotation["mask_image_file"]
        text = annotation["text"]
                 
        orig_image = cv2.imread(str(orig_image_path))
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        orig_image_pil = Image.fromarray(orig_image)
         
        mask_image = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)
        mask_image_pil = Image.fromarray(mask_image)
                               
        return {
            "orig_image": orig_image_pil,  # PIL Image
            "mask_image": mask_image_pil,  # PIL Image
            "text": text
        }

class VisualGenomeDataset(Dataset):
    """Visual Genome dataset loader - returns PIL Images with multiple mask types"""
    
    def __init__(self, 
                 region_descriptions_file,
                 image_data_file,
                 images_dir
                 ):
        
        self.images_dir = Path(images_dir)

        self._load_visual_genome_data(region_descriptions_file, image_data_file)
        
        print(f"Visual Genome dataset loaded: {len(self.samples)} samples")
    
    def _load_visual_genome_data(self, region_descriptions_file, image_data_file):
        """Load and filter Visual Genome data - handles nested structure"""
        # Load region descriptions
        with open(region_descriptions_file, 'r') as f:
            region_data = json.load(f)
        
        # Load image metadata
        with open(image_data_file, 'r') as f:
            image_data = json.load(f)
        
        # Create mapping
        image_id_to_info = {img['image_id']: img for img in image_data}
        
        # Filter and prepare samples - handle nested structure
        self.samples = []
        
        for image_entry in region_data:
            # Each entry has 'id' and 'regions' fields
            regions_list = image_entry.get('regions', [])
            
            for region in regions_list:
                # Check required fields
                required_fields = ['image_id', 'phrase', 'x', 'y', 'width', 'height']
                if not all(key in region for key in required_fields):
                    continue
                
                image_id = region['image_id']
                
                # Check if image exists in metadata
                if image_id not in image_id_to_info:
                    continue
                
                # Ensure region coordinates are valid
                if region['width'] <= 0 or region['height'] <= 0:
                    continue
                
                # Check if image file exists
                image_filename = f"{image_id}.jpg"
                image_path = self.images_dir / image_filename
                
                if not image_path.exists():
                    # Try other extensions
                    found = False
                    for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                        alt_path = self.images_dir / f"{image_id}{ext}"
                        if alt_path.exists():
                            region['image_filename'] = f"{image_id}{ext}"
                            found = True
                            break
                    if not found:
                        continue
                else:
                    region['image_filename'] = image_filename
                
                self.samples.append(region)

    def _generate_voronoi_noise(self, h, w, num_points=100, mean_value=0, std_dev=20, seed=None):
        """Generate Voronoi-based noise pattern"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random points
        points = np.random.rand(num_points, 2)
        points[:, 0] *= h  
        points[:, 1] *= w  
        
        # Generate random gray values for each point
        gray_values = np.random.normal(mean_value, std_dev, num_points)
        gray_values = np.clip(gray_values, 0, 255)

        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid_points = np.stack([y_coords.ravel(), x_coords.ravel()], axis=1)
        
        # Calculate distances and assign closest point
        distances = cdist(grid_points, points)
        closest_point_idx = np.argmin(distances, axis=1)
        
        # Create Voronoi map
        voronoi_map = closest_point_idx.reshape(h, w)
        voronoi_noise = gray_values[voronoi_map]
        
        return voronoi_noise
    
    def _create_bbox_mask(self, image_shape, bbox):
        """Create simple bbox mask: white inside bbox, black outside"""
        h, w = image_shape[:2]
        x, y, box_w, box_h = bbox
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        box_w = min(box_w, w - x)
        box_h = min(box_h, h - y)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        if box_w > 0 and box_h > 0:
            mask[y:y+box_h, x:x+box_w] = 255
        return mask
    
    def _create_ellipse_mask(self, image_shape, bbox):
        """Create elliptical mask with Voronoi noise outside"""
        h, w = image_shape[:2]
        x, y, box_w, box_h = bbox
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        box_w = min(box_w, w - x)
        box_h = min(box_h, h - y)
        
        if box_w <= 0 or box_h <= 0:
            # Return pure noise if bbox is invalid
            voronoi_noise = self._generate_voronoi_noise(h, w, num_points=100, mean_value=50, std_dev=30, seed=42)
            return voronoi_noise.astype(np.uint8)
        
        center_x = x + box_w // 2
        center_y = y + box_h // 2
        axis_x = box_w // 2
        axis_y = box_h // 2

        # Generate Voronoi noise for the entire image
        voronoi_noise = self._generate_voronoi_noise(h, w, num_points=100, mean_value=50, std_dev=30, seed=42)
        
        # Create ellipse mask
        mask = voronoi_noise.copy()
        
        # Use vectorized operations to create ellipse
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Calculate normalized distance to ellipse center
        norm_x = (x_coords - center_x) / axis_x if axis_x > 0 else 0
        norm_y = (y_coords - center_y) / axis_y if axis_y > 0 else 0
        ellipse_dist = np.sqrt(norm_x**2 + norm_y**2)
        
        # Fill ellipse interior with single value from normal distribution
        inside_ellipse = ellipse_dist <= 1.0
        ellipse_pixel_count = np.sum(inside_ellipse)
        
        if ellipse_pixel_count > 0:
            # Set seed for reproducibility
            np.random.seed(42)
            # Generate single value from normal distribution with mean=220, std=20
            ellipse_value = np.random.normal(220, 20)
            ellipse_value = np.clip(ellipse_value, 0, 255)  # Limit to 0-255 range
            mask[inside_ellipse] = ellipse_value
        
        return mask.astype(np.uint8)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load original image
        image_path = self.images_dir / sample['image_filename']
        
        try:
            orig_image = cv2.imread(str(image_path))
            if orig_image is None:
                # If loading fails, try next sample
                return self.__getitem__((idx + 1) % len(self.samples))
            
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # Get bbox and text from sample
        bbox = [sample['x'], sample['y'], sample['width'], sample['height']]
        text = sample['phrase']
        
        # Original bbox (without normalization) - [x, y, width, height]
        original_bbox = bbox.copy()
        
        # Normalize bbox to [0, 1] range - [x1, y1, x2, y2]
        h, w = orig_image.shape[:2]
        x, y, box_width, box_height = bbox
        normalized_bbox = [
            x / w,                    # x1
            y / h,                    # y1
            (x + box_width) / w,      # x2 = x + width
            (y + box_height) / h      # y2 = y + height
        ]
        
        # Create both types of masks
        bbox_mask = self._create_bbox_mask(orig_image.shape, bbox)
        ellipse_mask = self._create_ellipse_mask(orig_image.shape, bbox)
        
        # Convert to PIL images
        orig_image_pil = Image.fromarray(orig_image)
        bbox_mask_pil = Image.fromarray(bbox_mask)
        ellipse_mask_pil = Image.fromarray(ellipse_mask)
        
        return {
            "orig_image": orig_image_pil,  # PIL Image
            "bbox_mask": bbox_mask_pil,    # PIL Image - simple bbox mask
            "ellipse_mask": ellipse_mask_pil,  # PIL Image - elliptical mask with noise
            "text": text,
            "original_bbox": original_bbox,     # [x, y, width, height] - 未归一化
            "normalized_bbox": normalized_bbox, # [x1, y1, x2, y2] - 已归一化
        }

class RefCOCOPlusDataset(Dataset):
    """RefCOCO+ dataset loader for test data (val, testA, testB combined) - returns PIL Images with multiple mask types"""
    
    def __init__(self, 
                 refs_file,
                 instances_file,
                 images_dir
                ):

        self.images_dir = Path(images_dir)
        
        self._load_refcoco_data(refs_file, instances_file)
    
    def _load_refcoco_data(self, refs_file, instances_file):
        """Load RefCOCO+ test data (val, testA, testB)"""
        
        # Load referring expressions
        print(f"Loading refs from {refs_file}...")
        with open(refs_file, 'rb') as f:
            refs_data = pickle.load(f)
        
        # Load COCO instances
        print(f"Loading instances from {instances_file}...")
        with open(instances_file, 'r') as f:
            instances_data = json.load(f)
        
        # Create mappings
        self.image_id_to_info = {img['id']: img for img in instances_data.get('images', [])}
        self.ann_id_to_info = {ann['id']: ann for ann in instances_data.get('annotations', [])}
        self.cat_id_to_info = {cat['id']: cat for cat in instances_data.get('categories', [])}
        
        # Load test splits: val, testA, testB
        test_splits = ['val', 'testA', 'testB']
        self.samples = []
        split_counts = {}
        
        for ref in refs_data:
            # Only include samples from test splits
            if ref['split'] not in test_splits:
                continue
                
            # Get image and annotation info
            image_id = ref['image_id']
            ann_id = ref['ann_id']
            
            # Check if image and annotation exist
            if image_id not in self.image_id_to_info or ann_id not in self.ann_id_to_info:
                continue
            
            image_info = self.image_id_to_info[image_id]
            ann_info = self.ann_id_to_info[ann_id]
            
            # Check if image file exists
            image_filename = image_info['file_name']
            image_path = self.images_dir / image_filename
            
            if not image_path.exists():
                continue
            
            # Count samples per split
            split_name = ref['split']
            split_counts[split_name] = split_counts.get(split_name, 0) + len(ref['sentences'])
            
            # For each sentence in the referring expression
            for sent_info in ref['sentences']:
                sample = {
                    'ref_id': ref['ref_id'],
                    'ann_id': ann_id,
                    'image_id': image_id,
                    'category_id': ref['category_id'],
                    'sentence': sent_info['sent'],
                    'tokens': sent_info.get('tokens', []),
                    'image_filename': image_filename,
                    'bbox': ann_info['bbox'],  # [x, y, width, height]
                    'segmentation': ann_info.get('segmentation', [])
                }
                
                self.samples.append(sample)
        
        # Print split breakdown
        print(f"Found {len(self.samples)} total samples:")
        for split in test_splits:
            count = split_counts.get(split, 0)
            print(f"  - {split}: {count:,} samples")
    
    def _polygon_to_mask(self, segmentation, image_height, image_width):
        from pycocotools import mask as coco_mask
        
        # Handle different segmentation formats
        if isinstance(segmentation, list):
            # Polygon format
            mask = coco_mask.frPyObjects(segmentation, image_height, image_width)
            if isinstance(mask, list):
                # Multiple polygons
                mask = coco_mask.merge(mask)
            binary_mask = coco_mask.decode(mask)
        else:
            # Already in RLE format
            binary_mask = coco_mask.decode(segmentation)
        
        return binary_mask.astype(np.uint8) * 255
            
    def _generate_voronoi_noise(self, h, w, num_points=100, mean_value=50, std_dev=30, seed=None):
        """Generate Voronoi-based noise pattern"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random points
        points = np.random.rand(num_points, 2)
        points[:, 0] *= h  
        points[:, 1] *= w  
        
        # Generate random gray values for each point
        gray_values = np.random.normal(mean_value, std_dev, num_points)
        gray_values = np.clip(gray_values, 0, 255)

        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid_points = np.stack([y_coords.ravel(), x_coords.ravel()], axis=1)
        
        # Calculate distances and assign closest point
        distances = cdist(grid_points, points)
        closest_point_idx = np.argmin(distances, axis=1)
        
        # Create Voronoi map
        voronoi_map = closest_point_idx.reshape(h, w)
        voronoi_noise = gray_values[voronoi_map]
        
        return voronoi_noise
    
    def _add_voronoi_noise_to_mask(self, mask, 
                                  num_points=100, 
                                  mean_value=50, 
                                  std_dev=30, 
                                 ):
        h, w = mask.shape
        
        voronoi_noise = self._generate_voronoi_noise(
            h, w, 
            num_points=num_points, 
            mean_value=mean_value, 
            std_dev=std_dev, 
        )
        
        enhanced_mask = voronoi_noise.copy().astype(np.uint8)

        target_pixels = mask >= 230  
        
        enhanced_mask[target_pixels] = 255
        
        return enhanced_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load original image
        image_path = self.images_dir / sample['image_filename']
        
        try:
            orig_image = cv2.imread(str(image_path))
            if orig_image is None:
                # If loading fails, try next sample
                return self.__getitem__((idx + 1) % len(self.samples))
            
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # Get text and bbox
        text = sample['sentence']
        bbox = sample['bbox']  # [x, y, width, height]
        
        # Original bbox (without normalization) - [x, y, width, height]
        original_bbox = bbox.copy()
        
        # Normalize bbox to [0, 1] range - [x1, y1, x2, y2]
        h, w = orig_image.shape[:2]
        x, y, box_width, box_height = bbox
        normalized_bbox = [
            x / w,                    # x1
            y / h,                    # y1
            (x + box_width) / w,      # x2 = x + width
            (y + box_height) / h      # y2 = y + height
        ]

        h, w = orig_image.shape[:2]
        if sample['segmentation']:
            # Create binary mask from segmentation
            binary_mask = self._polygon_to_mask(sample['segmentation'], h, w)
            # Create gray mask by adding Voronoi noise
            gray_mask = self._add_voronoi_noise_to_mask(binary_mask)
        else:
            # Fallback: create empty masks if no segmentation
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            gray_mask = np.zeros((h, w), dtype=np.uint8)
            
        # Convert to PIL images
        orig_image_pil = Image.fromarray(orig_image)
        binary_mask_pil = Image.fromarray(binary_mask)
        gray_mask_pil = Image.fromarray(gray_mask)
        
        return {
            "orig_image": orig_image_pil,      # PIL Image
            "binary_mask": binary_mask_pil,    # PIL Image - pure binary mask
            "gray_mask": gray_mask_pil,        # PIL Image - mask with Voronoi noise
            "text": text,
            "ref_id": sample['ref_id'],
            "category_id": sample['category_id'],
            "original_bbox": original_bbox,     # [x, y, width, height] - 未归一化
            "normalized_bbox": normalized_bbox, # [x1, y1, x2, y2] - 已归一化
        }

# 示例使用（评估模式）
if __name__ == "__main__":
    class Config:
        # vg
        vg_region_descriptions_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/region_descriptions.json"
        vg_image_data_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_val_3000/image_data.json"
        vg_images_dir = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_100K_all"

        #refco
        refs_file = "/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/refs(unc).p"
        instances_file = "/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/instances.json"
        images_dir = "/home/chenzc/cvd/model_blip/data/data/COCO/train2014"

    config = Config

    # 创建评估用的数据集
    vg_dataset = VisualGenomeDataset(
        config.vg_region_descriptions_file,
        config.vg_image_data_file,
        config.vg_images_dir
    )
    
    ref_dataset = RefCOCOPlusDataset(
        config.refs_file,
        config.instances_file,
        config.images_dir
    )

    print(f"VG dataset length: {len(vg_dataset)}")
    print(f"RefCOCO+ dataset length: {len(ref_dataset)}")

    # 测试VG数据集返回的格式
    vg_sample = vg_dataset[0]
    print(f"\nVG Sample keys: {vg_sample.keys()}")
    print(f"Image type: {type(vg_sample['orig_image'])}")  # 应该是 PIL.Image.Image
    print(f"Image size: {vg_sample['orig_image'].size}")
    print(f"Bbox mask size: {vg_sample['bbox_mask'].size}")
    print(f"Ellipse mask size: {vg_sample['ellipse_mask'].size}")
    print(f"Text: {vg_sample['text']}")
    print(f"Original bbox: {vg_sample['original_bbox']}")        # 新增：未归一化
    print(f"Normalized bbox: {vg_sample['normalized_bbox']}")    # 已归一化
    
    # 测试RefCOCO+数据集返回的格式
    ref_sample = ref_dataset[0]
    print(f"\nRefCOCO+ Sample keys: {ref_sample.keys()}")
    print(f"Image type: {type(ref_sample['orig_image'])}")
    print(f"Image size: {ref_sample['orig_image'].size}")
    print(f"Binary mask size: {ref_sample['binary_mask'].size}")
    print(f"Gray mask size: {ref_sample['gray_mask'].size}")
    print(f"Text: {ref_sample['text']}")
    print(f"Original bbox: {ref_sample['original_bbox']}")        # 新增：未归一化
    print(f"Normalized bbox: {ref_sample['normalized_bbox']}")    # 已归一化