import numpy as np
import cv2


def process_single_image_pair(original_image: np.ndarray, cvd_image: np.ndarray):
    """
    Generate difference map between original and CVD images
    
    Args:
        original_image: Original image numpy array (RGB format)
        cvd_image: CVD simulated image numpy array (RGB format)
    
    Returns:
        numpy.ndarray: Difference map numpy array, returns None if failed
    """
    if original_image is None or cvd_image is None:
        return None
    
    # Calculate difference map (both images are already in RGB format)
    difference = cv2.absdiff(original_image, cvd_image)
    
    # Calculate average difference across three channels
    average_difference = np.mean(difference, axis=2)
    
    # Normalize to 0-255 range
    if average_difference.max() > average_difference.min():
        average_difference = ((average_difference - average_difference.min()) / 
                           (average_difference.max() - average_difference.min()) * 255).astype(np.uint8)
    else:
        average_difference = np.zeros_like(average_difference, dtype=np.uint8)
    
    return average_difference