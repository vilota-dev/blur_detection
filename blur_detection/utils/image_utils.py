import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        standardized = clahe.apply(gray)
        return cv2.cvtColor(standardized, cv2.COLOR_GRAY2BGR)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)

class CropGridSelector:
    def __init__(self, image: np.ndarray, grid_size: int = 3):
        self.image = image
        self.grid_size = grid_size
        self.h, self.w = image.shape[:2]

    def get_grid_coords(self):
        coords = {}
        cell_h = self.h // self.grid_size
        cell_w = self.w // self.grid_size
        cell_num = 1
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = (col + 1) * cell_w if col < self.grid_size - 1 else self.w
                y1 = (row + 1) * cell_h if row < self.grid_size - 1 else self.h
                coords[cell_num] = (x0, y0, x1, y1)
                cell_num += 1
        return coords

    def get_crop(self, cell_num: int) -> np.ndarray:
        coords = self.get_grid_coords()
        if cell_num not in coords:
            raise ValueError(f"Cell must be 1-{self.grid_size**2}")
        x0, y0, x1, y1 = coords[cell_num]
        
        if cell_num in {1, 7}:
            x1 = max(x0, x1 - 200)
            
        return self.image[y0:y1, x0:x1].copy()

def directional_crop_and_pad(image, bboxes, target_size=(200, 200), bg_color=(0, 0, 0), dx=-50, dy=0):
    img_h, img_w = image.shape[:2]
    target_w, target_h = target_size
    far_bbox = max(bboxes, key=lambda b: b[2])
    _, f_y0, f_x1, f_y1 = far_bbox
    anchor_right_x = f_x1
    anchor_center_y = (f_y0 + f_y1) // 2

    crop_x1 = anchor_right_x + dx
    crop_x0 = crop_x1 - target_w
    
    crop_y0 = anchor_center_y - (target_h // 2) + dy
    crop_y1 = crop_y0 + target_h

    if crop_x0 < 0:
        crop_x0 = 0
        crop_x1 = target_w
    elif crop_x1 > img_w:
        crop_x1 = img_w
        crop_x0 = img_w - target_w

    if crop_y0 < 0:
        crop_y0 = 0
        crop_y1 = target_h
    elif crop_y1 > img_h:
        crop_y1 = img_h
        crop_y0 = img_h - target_h

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    valid_x0 = max(0, crop_x0)
    valid_x1 = min(img_w, crop_x1)
    valid_y0 = max(0, crop_y0)
    valid_y1 = min(img_h, crop_y1)
    
    if valid_x0 >= valid_x1 or valid_y0 >= valid_y1:
        return canvas

    canvas_x0 = valid_x0 - crop_x0
    canvas_x1 = canvas_x0 + (valid_x1 - valid_x0)
    canvas_y0 = valid_y0 - crop_y0
    canvas_y1 = canvas_y0 + (valid_y1 - valid_y0)
    canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1] = image[valid_y0:valid_y1, valid_x0:valid_x1]
    return canvas

def get_horizontal_patch_mask_from_array(img_bgr, image_size=224, patch_size=16, threshold=50):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_size, image_size))

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    abs_sobel_y = np.absolute(sobel_y)

    if abs_sobel_y.max() == 0:
        sobel_8u = np.zeros_like(abs_sobel_y, dtype=np.uint8)
    else:
        sobel_8u = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))

    _, binary_mask = cv2.threshold(sobel_8u, threshold, 255, cv2.THRESH_BINARY)
    mask_tensor = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    patch_mask_2d = F.max_pool2d(mask_tensor, kernel_size=patch_size, stride=patch_size)
    patch_mask_flat = patch_mask_2d.view(-1) > 0

    return patch_mask_flat

def get_laplacian_tensor_from_array(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    gray = pil.convert("L")
    gray_t = torch.from_numpy(np.array(gray)).float().unsqueeze(0).unsqueeze(0) / 255.0

    lap_kernel = torch.tensor([[[[0, 1, 0], [0, -2, 0], [0, 1, 0]]]], dtype=torch.float32)
    lap = F.conv2d(gray_t, lap_kernel, padding=1)

    return torch.abs(lap)

def stretch_contrast(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Clip and stretch the V (brightness) channel based on percentiles
    low_val, high_val = np.percentile(v, (low_percentile, high_percentile))
    v_stretched = np.clip(v, low_val, high_val)
    if high_val > low_val:
        v_stretched = ((v_stretched - low_val) / (high_val - low_val) * 255.0).astype(np.uint8)
    else:
        v_stretched = v_stretched.astype(np.uint8)
        
    # Merge back and convert to BGR
    hsv_stretched = cv2.merge((h, s, v_stretched))
    return cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)

def equalize_contrast(image: np.ndarray) -> np.ndarray:
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply global histogram equalization to the V channel
    v_equalized = cv2.equalizeHist(v)
    
    # Merge back and convert to BGR
    hsv_equalized = cv2.merge((h, s, v_equalized))
    return cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

def stretch_contrast_rgb(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    # Clip and stretch BGR channels globally based on percentiles
    low_val, high_val = np.percentile(image, (low_percentile, high_percentile))
    stretched = np.clip(image, low_val, high_val)
    if high_val > low_val:
        stretched = ((stretched - low_val) / (high_val - low_val) * 255.0).astype(np.uint8)
    else:
        stretched = stretched.astype(np.uint8)
    return stretched

def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # Adjust image brightness non-linearly using a lookup table
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)