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

class DualImageTransform:
    """Dual image data augmentation transform"""

    def __init__(self):
        self.rgb_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=30),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=30),
        ])
     
    def __call__(self, orig_image, mask_image):
        """Synchronously transform original image and mask image"""
        current_seed = random.randint(0, 2**32 - 1)
        set_seed(current_seed)
        orig_image_transformed = self.rgb_transform(orig_image)
        set_seed(current_seed)
        mask_image_transformed = self.mask_transform(mask_image)
                 
        return orig_image_transformed, mask_image_transformed

class CVDDataset(Dataset):
    """CVD dataset loader"""
    def __init__(self, image_dir, annotations_file):
        self.image_dir = Path(image_dir)
        self.transform = DualImageTransform()
                 
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
         
        orig_image_transformed, mask_image_transformed = self.transform(orig_image_pil, mask_image_pil)
                               
        return {
            "orig_image": orig_image_transformed,
            "mask_image": mask_image_transformed,
            "text": text,
            "orig_image_path": str(orig_image_path)
        }

class VisualGenomeDataset(Dataset):
    """Visual Genome dataset loader"""
    
    def __init__(self, 
                 region_descriptions_file,
                 image_data_file,
                 images_dir,
                 mask_type='ellipse',
                 ):
        
        self.images_dir = Path(images_dir)
        self.transform = DualImageTransform()
        self.mask_type = mask_type

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
    
    def _create_mask_from_bbox(self, image_shape, bbox, mask_type='ellipse'):
        """
        Create mask from bbox
        
        Args:
            image_shape: Shape of the image (h, w, c)
            bbox: Bounding box [x, y, width, height]
            mask_type: 'ellipse' for elliptical mask with noise, 'simple' for bbox mask
        
        Returns:
            mask: Generated mask array
        """
        h, w = image_shape[:2]
        x, y, box_w, box_h = bbox
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        box_w = min(box_w, w - x)
        box_h = min(box_h, h - y)
        
        if mask_type == 'bbox':
            # Simple bbox mask: white inside bbox, black outside
            mask = np.zeros((h, w), dtype=np.uint8)
            if box_w > 0 and box_h > 0:
                mask[y:y+box_h, x:x+box_w] = 255
            return mask
        
        elif mask_type == 'ellipse':
            # Elliptical mask with Voronoi noise outside
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
        
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}. Use 'ellipse' or 'simple'.")
    
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
        
        # ===== 修改的代码：bbox处理 =====
        # Get bbox and text from sample
        bbox = [sample['x'], sample['y'], sample['width'], sample['height']]
        text = sample['phrase']
        
        # Normalize bbox to [0, 1] range
        h, w = orig_image.shape[:2]
        normalized_bbox = [
            bbox[0] / w,  # x / width
            bbox[1] / h,  # y / height
            bbox[2] / w,  # width / width  
            bbox[3] / h   # height / height
        ]
        # ===== 修改的代码结束 =====
        
        # Create mask
        mask_array = self._create_mask_from_bbox(orig_image.shape, bbox, self.mask_type)
        
        # Convert to PIL images
        orig_image_pil = Image.fromarray(orig_image)
        mask_image_pil = Image.fromarray(mask_array)
        
        # Apply transforms
        orig_image_transformed, mask_image_transformed = self.transform(orig_image_pil, mask_image_pil)
        
        return {
            "orig_image": orig_image_transformed,
            "mask_image": mask_image_transformed,
            "text": text,
            "bbox": bbox,  # ===== 添加：原始bbox =====
            "normalized_bbox": normalized_bbox,  # 归一化bbox
        }


class RefCOCOPlusDataset(Dataset):
    """RefCOCO+ dataset loader for test data (val, testA, testB combined)"""
    
    def __init__(self, 
                 refs_file,
                 instances_file,
                 images_dir,
                 binary_mask=True,
                ):

        self.images_dir = Path(images_dir)
        self.transform = DualImageTransform()
        self.binary_mask = binary_mask
        
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
    
    def add_voronoi_noise_to_mask(self, mask, 
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
        
        # ===== 修改的代码：bbox处理 =====
        # Get text and bbox
        text = sample['sentence']
        bbox = sample['bbox']  # [x, y, width, height]
        
        # Normalize bbox to [0, 1] range
        h, w = orig_image.shape[:2]
        normalized_bbox = [
            bbox[0] / w,  # x / width
            bbox[1] / h,  # y / height
            bbox[2] / w,  # width / width
            bbox[3] / h   # height / height
        ]
        # ===== 修改的代码结束 =====

        h, w = orig_image.shape[:2]
        if sample['segmentation']:
            mask_array = self._polygon_to_mask(sample['segmentation'], h, w)

        if not self.binary_mask:
            mask_array = self.add_voronoi_noise_to_mask(mask_array)
        # Convert to PIL images
        orig_image_pil = Image.fromarray(orig_image)
        mask_image_pil = Image.fromarray(mask_array)
        
        # Apply transforms
        orig_image_transformed, mask_image_transformed = self.transform(orig_image_pil, mask_image_pil)
        
        return {
            "orig_image": orig_image_transformed,
            "mask_image": mask_image_transformed,
            "text": text,
            "ref_id": sample['ref_id'],
            "category_id": sample['category_id'],
            "bbox": bbox,  # ===== 添加：原始bbox =====
            "normalized_bbox": normalized_bbox,  # 归一化bbox
        }
    
if __name__ == "__main__":
    class Config:
        # vg
        
        vg_region_descriptions_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_train_105077/region_descriptions.json"
        vg_image_data_file = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_train_105077/image_data.json"
        vg_images_dir = "/home/chenzc/cvd/model_blip/data/data/visual_genome/VG_100K_all"
        vg_mask_type='ellipse'

        #refco
        refs_file = "/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/refs(unc).p"
        instances_file = "/home/chenzc/cvd/model_blip/data/data/COCO/refcoco+/instances.json"
        images_dir = "/home/chenzc/cvd/model_blip/data/data/COCO/train2014"
        binary_mask=True

    config = Config

    vg_dataset_ell = VisualGenomeDataset(config.vg_region_descriptions_file,
                                     config.vg_image_data_file,
                                     config.vg_images_dir,
                                     mask_type = 'ellipse'
                                     )
    
    vg_dataset_bbox = VisualGenomeDataset(config.vg_region_descriptions_file,
                                     config.vg_image_data_file,
                                     config.vg_images_dir,
                                     mask_type = 'bbox'
                                     )
    
    ref_dataset_binary = RefCOCOPlusDataset(config.refs_file,
                                     config.instances_file,
                                     config.images_dir,
                                     binary_mask=True
                                     )
    
    ref_dataset_gray = RefCOCOPlusDataset(config.refs_file,
                                     config.instances_file,
                                     config.images_dir,
                                     binary_mask=False
                                     )
    print(len(vg_dataset_ell))

    save_dir = 'vg_dataset_ell'
    os.makedirs(save_dir, exist_ok=True)

    data_dict = {}  
    
    for i in range(8):
        sample = vg_dataset_ell[i]
        orig = sample['orig_image']
        mask = sample['mask_image']
        text = sample['text']
        bbox = sample['bbox']  # ===== 添加：获取原始bbox =====
        normalized_bbox = sample['normalized_bbox']  

        # 保存图像和 mask
        from torchvision.utils import save_image
        save_image(orig, os.path.join(save_dir, f"image_{i}.png"))
        save_image(mask, os.path.join(save_dir, f"mask_{i}.png"))

        # ===== 修改：保存text、原始bbox和normalized_bbox =====
        data_dict[str(i)] = {
            "text": text,
            "bbox": bbox,  # 原始bbox
            "normalized_bbox": normalized_bbox  # 归一化bbox
        }

    with open(os.path.join(save_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    print(len(vg_dataset_bbox))

    save_dir = 'vg_dataset_bbox'
    os.makedirs(save_dir, exist_ok=True)

    data_dict = {}  
    
    for i in range(8):
        sample = vg_dataset_bbox[i]
        orig = sample['orig_image']
        mask = sample['mask_image']
        text = sample['text']
        bbox = sample['bbox']  # ===== 添加：获取原始bbox =====
        normalized_bbox = sample['normalized_bbox']  

        # 保存图像和 mask
        from torchvision.utils import save_image
        save_image(orig, os.path.join(save_dir, f"image_{i}.png"))
        save_image(mask, os.path.join(save_dir, f"mask_{i}.png"))

        # ===== 修改：保存text、原始bbox和normalized_bbox =====
        data_dict[str(i)] = {
            "text": text,
            "bbox": bbox,  # 原始bbox
            "normalized_bbox": normalized_bbox  # 归一化bbox
        }

    with open(os.path.join(save_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    

    print(len(ref_dataset_binary))

    save_dir = 'ref_dataset_binary'
    os.makedirs(save_dir, exist_ok=True)

    data_dict = {}  
    
    for i in range(8):
        sample = ref_dataset_binary[i]
        orig = sample['orig_image']
        mask = sample['mask_image']
        text = sample['text']
        bbox = sample['bbox']  # ===== 添加：获取原始bbox =====
        normalized_bbox = sample['normalized_bbox']  

        # 保存图像和 mask
        from torchvision.utils import save_image
        save_image(orig, os.path.join(save_dir, f"image_{i}.png"))
        save_image(mask, os.path.join(save_dir, f"mask_{i}.png"))

        # ===== 修改：保存text、原始bbox、normalized_bbox和其他字段 =====
        data_dict[str(i)] = {
            "text": text,
            "bbox": bbox,  # 原始bbox
            "normalized_bbox": normalized_bbox,  # 归一化bbox
            "ref_id": sample['ref_id'],
            "category_id": sample['category_id']
        }

    with open(os.path.join(save_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    print(len(ref_dataset_gray))

    save_dir = 'ref_dataset_gray'
    os.makedirs(save_dir, exist_ok=True)

    data_dict = {}  
    
    for i in range(8):
        sample = ref_dataset_gray[i]
        orig = sample['orig_image']
        mask = sample['mask_image']
        text = sample['text']
        bbox = sample['bbox']  # ===== 添加：获取原始bbox =====
        normalized_bbox = sample['normalized_bbox']  

        # 保存图像和 mask
        from torchvision.utils import save_image
        save_image(orig, os.path.join(save_dir, f"image_{i}.png"))
        save_image(mask, os.path.join(save_dir, f"mask_{i}.png"))

        # ===== 修改：保存text、原始bbox、normalized_bbox和其他字段 =====
        data_dict[str(i)] = {
            "text": text,
            "bbox": bbox,  # 原始bbox
            "normalized_bbox": normalized_bbox,  # 归一化bbox
            "ref_id": sample['ref_id'],
            "category_id": sample['category_id']
        }

    with open(os.path.join(save_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)