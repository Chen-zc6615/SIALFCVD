import torch
import numpy as np
import cv2
from skimage import color
from typing import Tuple

class CCPRCalculator:
    """CCPR calculator for processing numpy arrays directly"""
    
    def __init__(self, delta: float = 6.0, ccpr_threshold: float = 0.7, device: str = "cuda"):
        """
        Initialize CCPR calculator
        
        Args:
            delta: Color difference threshold, default 6.0, the value can't be changed
            ccpr_threshold: CCPR threshold, default 0.7
            device: Computing device 'cuda' or 'cpu'
        """
        self.delta = delta
        self.ccpr_threshold = ccpr_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.neighbor_offsets = torch.tensor([
            (-1, -1), (-1, 0), (-1, 1),  
            (0, -1),           (0, 1),  
            (1, -1),  (1, 0),  (1, 1)    
        ], dtype=torch.long, device=self.device)

    def convert_to_lab(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert RGB image to Lab color space
        
        Args:
            image: RGB image numpy array
            
        Returns:
            torch.Tensor: Image tensor in Lab color space
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        lab_image = color.rgb2lab(image)
        return torch.tensor(lab_image, dtype=torch.float32, device=self.device)

    def get_neighboring_pixel_pairs(self, lab_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all 8-direction neighboring pixel pairs
        
        Args:
            lab_image: Image tensor in Lab color space
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Center pixels and neighbor pixel pairs
        """
        height, width, _ = lab_image.shape
        y, x = torch.meshgrid(torch.arange(height, device=self.device), 
                            torch.arange(width, device=self.device), 
                            indexing="ij")

        # Process all directions
        all_pairs = []
        for dy, dx in self.neighbor_offsets:
            ny, nx = y + dy, x + dx
            valid_mask = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            
            y_idx, x_idx = torch.where(valid_mask)
            ny, nx = ny[y_idx, x_idx], nx[y_idx, x_idx]

            p1 = lab_image[y_idx, x_idx]  # center pixel
            p2 = lab_image[ny, nx]        # neighbor pixel
            all_pairs.append(torch.stack((p1, p2)))

        # Combine all direction pixel pairs
        all_pairs = torch.cat(all_pairs, dim=1)
        return all_pairs[0], all_pairs[1]

    def color_difference(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Euclidean distance between two pixels in Lab color space
        
        Args:
            p1: First pixel
            p2: Second pixel
            
        Returns:
            torch.Tensor: Color difference values
        """
        return torch.sqrt(torch.sum((p1 - p2) ** 2, dim=1))

    def calculate_ccpr(self, original_image: np.ndarray, cvd_image: np.ndarray) -> Tuple[float, bool]:
        """
        Calculate CCPR value between two images
        
        Args:
            original_image: Original image numpy array (RGB format)
            cvd_image: CVD simulated image numpy array (RGB format)

        Returns:
            Tuple[float, bool]: CCPR value and whether CVD indistinguishable
            
        Raises:
            ValueError: If images are None or dimensions don't match
        """
        # Input validation
        if original_image is None or cvd_image is None:
            raise ValueError("Input images cannot be None")

        if original_image.shape != cvd_image.shape:
            raise ValueError("Original and CVD images must have the same dimensions")

        # Convert to Lab color space
        lab_original = self.convert_to_lab(original_image)
        lab_cvd = self.convert_to_lab(cvd_image)

        # Get neighboring pixel pairs
        original_center, original_neighbors = self.get_neighboring_pixel_pairs(lab_original)
        cvd_center, cvd_neighbors = self.get_neighboring_pixel_pairs(lab_cvd)

        # Calculate color differences
        original_diff = self.color_difference(original_center, original_neighbors)
        cvd_diff = self.color_difference(cvd_center, cvd_neighbors)

        # Create masks
        omega_mask = original_diff >= self.delta        # distinguishable pixel pairs in original image
        preserved_mask = cvd_diff >= self.delta         # still distinguishable pixel pairs in CVD image

        # Calculate CCPR
        omega_count = torch.sum(omega_mask)             # total number of distinguishable pairs in original
        preserved_count = torch.sum(preserved_mask)     # number of preserved distinguishable pairs
        
        ccpr = preserved_count / omega_count if omega_count > 0 else 1.0

        # Determine if CVD indistinguishable
        is_cvd_indistinguishable = ccpr < self.ccpr_threshold

        print(f"CCPR value: {ccpr:.3f}")
        print(f"CVD indistinguishable: {is_cvd_indistinguishable}")

        return float(ccpr), bool(is_cvd_indistinguishable)