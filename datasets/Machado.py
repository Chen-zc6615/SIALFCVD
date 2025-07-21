import numpy as np
import copy
import os
import cv2
import random
from pathlib import Path
import shutil

# Simulate matrix
def cvd_matrix(cvd_type, cvd_level):
    if cvd_type == 0:

        T = {
            10: [0.856167, 0.182038, -0.038205,
                 0.029342, 0.955115, 0.015544,
                 -0.002880, -0.001563, 1.004443],

            20: [0.734766, 0.334872, -0.069637,
                 0.051840, 0.919198, 0.028963,
                 -0.004928, -0.004209, 1.009137],

            30: [0.630323, 0.465641, -0.095964,
                 0.069181, 0.890046, 0.040773,
                 -0.006308, -0.007724, 1.014032],
            40: [0.539009, 0.579343, -0.118352,
                 0.082546, 0.866121, 0.051332,
                 -0.007136, -0.011959, 1.019095],
            50: [0.458064, 0.679578, -0.137642,
                 0.092785, 0.846313, 0.060902,
                 -0.007494, -0.016807, 1.024301],
            60: [0.385450, 0.769005, -0.154455,
                 0.100526, 0.829802, 0.069673,
                 -0.007442, -0.022190, 1.029632],
            70: [0.319627, 0.849633, -0.169261,
                 0.106241, 0.815969, 0.077790,
                 -0.007025, -0.028051, 1.035076],
            80: [0.259411, 0.923008, -0.182420,
                 0.110296, 0.804340, 0.085364,
                 -0.006276, -0.034346, 1.040622],
            90: [0.203876, 0.990338, -0.194214,
                 0.112975, 0.794542, 0.092483,
                 -0.005222, -0.041043, 1.046265],
            100: [0.152286, 1.052583, -0.204868,
                  0.114503, 0.786281, 0.099216,
                  -0.003882, -0.048116, 1.051998]
        }
        return T.get(cvd_level, None)
    
    elif cvd_type == 1:

        T = {
            10: [0.866435, 0.177704, -0.044139,
                 0.049567, 0.939063, 0.011370,
                 -0.003453, 0.007233, 0.996220],
            20: [0.760729, 0.319078, -0.079807,
                 0.090568, 0.889315, 0.020117,
                 -0.006027, 0.013325, 0.992702],
            30: [0.675425, 0.433850, -0.109275,
                 0.125303, 0.847755, 0.026942,
                 -0.007950, 0.018572, 0.989378],
            40: [0.605511, 0.528560, -0.134071,
                 0.155318, 0.812366, 0.032316,
                 -0.009376, 0.023176, 0.986200],
            50: [0.547494, 0.607765, -0.155259,
                 0.181692, 0.781742, 0.036566,
                 -0.010410, 0.027275, 0.983136],
            60: [0.498864, 0.674741, -0.173604,
                 0.205199, 0.754872, 0.039929,
                 -0.011131, 0.030969, 0.980162],
            70: [0.457771, 0.731899, -0.189670,
                 0.226409, 0.731012, 0.042579,
                 -0.011595, 0.034333, 0.977261],
            80: [0.422823, 0.781057, -0.203881,
                 0.245752, 0.709602, 0.044646,
                 -0.011843, 0.037423, 0.974421],
            90: [0.392952, 0.823610, -0.216562,
                 0.263559, 0.690210, 0.046232,
                 -0.011910, 0.040281, 0.971630],
            100: [0.367322, 0.860646, -0.227968,
                  0.280085, 0.672501, 0.047413,
                  -0.011820, 0.042940, 0.968881],
        }
        return T.get(cvd_level, None)
    elif cvd_type == 2:
        T = {
            20: [0.895720, 0.133330, -0.029050,
                 0.029997, 0.945400, 0.024603,
                 0.013027, 0.104707, 0.882266],
            40: [0.948035, 0.089490, -0.037526,
                 0.014364, 0.946792, 0.038844,
                 0.010853, 0.193991, 0.795156],
            60: [1.104996, -0.046633, -0.058363,
                 -0.032137, 0.971635, 0.060503,
                 0.001336, 0.317922, 0.680742],
            80: [1.257728, -0.139648, -0.118081,
                 -0.078003, 0.975409, 0.102594,
                 -0.003316, 0.501214, 0.502102],
            100: [1.255528, -0.076749, -0.178779,
                  -0.078411, 0.930809, 0.147602,
                  0.004733, 0.691367, 0.303900],
        }
        return T.get(cvd_level, None)
    
# img, transforme matrxi is 3*3
def cvd_simulate(img, T, choose=0):
    if choose == 0:
        [img_h, img_w, img_depth] = img.shape
        img = img.astype(float)
        sim_img = np.dot(np.reshape(img, (img_h * img_w, 3)), T.transpose())
        sim_img = np.reshape(sim_img, (img_h, img_w, 3))
    if choose == 1:
        B = img[:, :, 0]
        G = img[:, :, 1]
        R = img[:, :, 2]
        img_rgb = copy.deepcopy(img)
        img_rgb[:, :, 0] = R  # [R,G,B]
        img_rgb[:, :, 1] = G
        img_rgb[:, :, 2] = B
        [img_h, img_w, img_depth] = img.shape
        img = img_rgb.astype(float)  # 变成RGB顺序的图像
        # np.reshape(img, (img_h * img_w, 3)) #把每个通道的二维矩阵变成一维的 再成T的转置矩阵
        sim_img = np.dot(np.reshape(img, (img_h * img_w, 3)), T.transpose())  # T.transpose()转置矩阵
        sim_img = np.reshape(sim_img, (img_h, img_w, 3))

    return sim_img
    

def batch_process_images(input_dir, output_dir):
    """
    Process all images in the input directory     
    Args:
          input_dir: Path to the input directory
          output_dir: Path to the output directory
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    img_extentions = ['.jpg', '.jpeg', '.png']

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.rglob('*'):
          if img_path.is_file() and img_path.suffix.lower() in img_extentions:

               rel_path = img_path.relative_to(input_dir)

               nwe_dir = output_dir / rel_path.parent / rel_path.stem
               nwe_dir.mkdir(parents=True, exist_ok=True)

               process_single_image(img_path, nwe_dir)

               
def process_single_image(img, cvd_type, cvd_level):
    """
    Process a single image with specified CVD type and level
    
    Args:
        img_path: Path to the input image
        cvd_type: CVD type (0: Protan, 1: Deutan, 2: Tritan)
        cvd_level: CVD severity level
    
    Returns:
        numpy array: processed image (None if failed)
    """
    
    T = cvd_matrix(cvd_type, cvd_level)
    if T is None:
        return None
    
    T = np.reshape(T, (3, 3))
    
    # 进行CVD模拟
    simulated_img = cvd_simulate(img, T, choose=1)

    # 归一化处理
    simulated_img = (simulated_img - simulated_img.min()) / (simulated_img.max() - simulated_img.min())
    simulated_img = (simulated_img * 255).astype(np.uint8)

    return simulated_img


# 使用示例
if __name__ == "__main__":
    # 指定CVD参数处理图像
    img_path = "/path/to/your/image.jpg"
    
    # 示例1: Protan 70%
    simulated_image = process_single_image(img_path, cvd_type=0, cvd_level=70)
    
    # 示例2: Deutan 50%
    # simulated_image = process_single_image(img_path, cvd_type=1, cvd_level=50)
    
    # 示例3: Tritan 80%
    # simulated_image = process_single_image(img_path, cvd_type=2, cvd_level=80)
    
    if simulated_image is not None:
        print(f"图像尺寸: {simulated_image.shape}")
        # 现在你可以使用 simulated_image 进行进一步处理
    else:
        print("图像处理失败")
    
