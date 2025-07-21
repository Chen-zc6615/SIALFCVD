import torch
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from pathlib import Path

from models.Alpha_BLIP.model.alpha_blip import AlphaBlipForConditionalGeneration

class AlphaBlipSingleInference:
    def __init__(self, 
                 checkpoint_path="models/Alpha_BLIP/checkpoints/checkpoint_epoch_4.pt", 
                 base_model="Salesforce/blip-image-captioning-base"
                 ):
        """
        Initialize Alpha BLIP inference - single data processing version
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            base_model: Pretrained model name
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(base_model)
        self.model = AlphaBlipForConditionalGeneration.from_pretrained_alpha(base_model)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Handle position_ids compatibility
        embeddings = self.model.text_decoder.bert.embeddings
        if not hasattr(embeddings, 'position_ids'):
            position_ids = torch.arange(512).expand((1, -1))
            embeddings.register_buffer('position_ids', position_ids, persistent=False)

        print("✅ Model loaded successfully")

    def generate_caption(self, image, mask_image=None, max_length=20, num_beams=3):
        """
        Generate caption from image (supports both PIL and numpy array inputs)
        
        Args:
            image: PIL Image object or numpy array (RGB format)
            mask_image: PIL Image object, numpy array, or None (optional)
            max_length: Maximum length of generated text
            num_beams: Beam search size
        
        Returns:
            Generated caption text
        """
        # Handle different input formats for main image
        if isinstance(image, np.ndarray):
            orig_image = Image.fromarray(image).convert('RGB')
        else:
            orig_image = image.convert('RGB')
        
        images = [orig_image]  # processor needs list format
        
        # Process original image
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Handle mask image
        alpha_values = None
        if mask_image is not None:
            # Handle different input formats for mask image
            if isinstance(mask_image, np.ndarray):
                # Convert numpy array to PIL Image
                if len(mask_image.shape) == 2:  # Grayscale
                    alpha_image = Image.fromarray(mask_image, mode='L').convert('RGB')
                else:  # RGB
                    alpha_image = Image.fromarray(mask_image).convert('RGB')
            else:
                alpha_image = mask_image.convert('RGB')
            
            alpha_processed = self.processor.image_processor(
                alpha_image, 
                return_tensors="pt"
            )['pixel_values']
                
            # Ensure single channel
            if alpha_processed.shape[1] == 3:
                alpha_processed = alpha_processed[:, 0:1, :, :]
            
            alpha_values = alpha_processed
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        # Data type matching
        model_dtype = next(self.model.parameters()).dtype
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model_dtype)
        
        if alpha_values is not None:
            alpha_values = alpha_values.to(device=self.device, dtype=model_dtype)
        
        # Prepare generation parameters
        generation_kwargs = {
            'pixel_values': inputs['pixel_values'],
            'alpha_values': alpha_values,
            'max_length': max_length,
            'num_beams': num_beams,
            'early_stopping': True,
        }
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(**generation_kwargs)
        
        # Decode generated text
        generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text.strip()