import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import yaml

# -----------------------------
# Load configuration
# -----------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path=CONFIG_PATH):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

# -----------------------------
# Utilities (crop / grid / CLAHE)
# -----------------------------
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        standardized = clahe.apply(gray)
        return cv2.cvtColor(standardized, cv2.COLOR_GRAY2BGR)
    else:
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
        return self.image[y0:y1, x0:x1].copy()

# -----------------------------
# SAM3 wrapper (attempt import)
# -----------------------------
class Sam3BuildingSegmenter:
    def __init__(self, model_path=None):
        self.current_image = None
        self.current_crop = None
        self.grid_selector = None
        self.processor = None
        self._init_model(model_path)

    def _init_model(self, model_path=None):
        try:
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            if model_path is None:
                model_path = f"{sam3_root}/sam3/assets/sam3.pt"

            self.model = build_sam3_image_model(checkpoint_path=model_path, bpe_path=bpe_path)
            self.processor = Sam3Processor(self.model)
        except Exception as e:
            raise ImportError("sam3 not available or failed to load: " + str(e))

    def load_image(self, image_path: str) -> bool:
        try:
            img_pil = Image.open(image_path).convert("RGB")
            self.current_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.grid_selector = CropGridSelector(self.current_image)
            return True
        except Exception:
            return False

    def select_crop_region(self, cell_num: int) -> bool:
        try:
            self.current_crop = self.grid_selector.get_crop(cell_num)
            return True
        except Exception:
            return False

    def add_text_prompt(self, text: str):
        try:
            crop_pil = Image.fromarray(cv2.cvtColor(self.current_crop, cv2.COLOR_BGR2RGB))
            state = self.processor.set_image(crop_pil)
            state = self.processor.set_text_prompt(text, state)
            return {"masks": state.get("masks", [])}
        except Exception:
            return None

# -----------------------------
# Directional crop (uses configurable left shift)
# -----------------------------
def directional_crop_and_pad(image, bboxes, target_size=(200, 200), bg_color=(0, 0, 0), shift_left_pixels=50):
    img_h, img_w = image.shape[:2]
    target_w, target_h = target_size
    far_bbox = max(bboxes, key=lambda b: b[2])
    f_x0, f_y0, f_x1, f_y1 = far_bbox
    anchor_right_x = f_x1
    anchor_center_y = (f_y0 + f_y1) // 2
    crop_x1 = anchor_right_x - shift_left_pixels
    crop_x0 = crop_x1 - target_w
    crop_y0 = anchor_center_y - (target_h // 2)
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

# -----------------------------
# Blur classifier (depends on dinov3)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class BlurClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes=4, laplacian_input_dim=196):
        super().__init__()
        self.backbone = backbone
        linear_input_dim = 2 * embed_dim
        self.laplacian_projector = nn.Sequential(
            nn.Linear(laplacian_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        fused_input_dim = linear_input_dim + 128
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            ResidualBlock(2048, 1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, patch_mask, laplacian_tensor):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        mask_weights = patch_mask.float().unsqueeze(-1)
        weighted_patches = patch_tokens * mask_weights
        summed_patches = weighted_patches.sum(dim=1)
        valid_patch_count = mask_weights.sum(dim=1) + 1e-6
        masked_patch_mean = summed_patches / valid_patch_count
        dino_feature = torch.cat([cls_token, masked_patch_mean], dim=1)
        pooled_laplacian = F.adaptive_avg_pool2d(laplacian_tensor, (14, 14))
        flat_laplacian = pooled_laplacian.view(pooled_laplacian.size(0), -1)
        laplacian_features = self.laplacian_projector(flat_laplacian)
        fused_input = torch.cat([dino_feature, laplacian_features], dim=1)
        logits = self.classifier_head(fused_input)
        return logits

# -----------------------------
# Helpers: patch mask and laplacian from numpy image
# -----------------------------
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
    gray = pil.convert('L')
    gray_t = torch.from_numpy(np.array(gray)).float().unsqueeze(0).unsqueeze(0) / 255.0
    lap_kernel = torch.tensor([[[[0,1,0],[0,-2,0],[0,1,0]]]], dtype=torch.float32)
    lap = F.conv2d(gray_t, lap_kernel, padding=1)
    return torch.abs(lap)

# -----------------------------
# Main processing and Streamlit UI
# -----------------------------
def process_and_predict(input_folder, output_folder, filtered_images_folder, config, device):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    out_filtered = Path(filtered_images_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    out_filtered.mkdir(parents=True, exist_ok=True)

    # Initialize SAM3 segmenter
    try:
        segmenter = Sam3BuildingSegmenter(model_path=config.get('sam3_checkpoint'))
    except Exception as e:
        raise RuntimeError('Failed to initialize SAM3: ' + str(e))

    # Initialize DINOv3 backbone if available
    try:
        from dinov3.hub.backbones import dinov3_vith16plus, Weights
        # Create backbone with pretrained weights if provided
        weights_arg = config.get('dino_backbone_weights') or Weights.LVD1689M
        backbone = dinov3_vith16plus(pretrained=True, weights=weights_arg)
    except Exception as e:
        raise RuntimeError('Failed to import or build dinov3 backbone: ' + str(e))

    # Build classifier wrapper
    embed_dim = backbone.embed_dim
    model = BlurClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=config.get('num_classes',4))
    model = model.to(device)

    # Load trained head
    head_path = config.get('trained_head_path')
    if not Path(head_path).exists():
        raise FileNotFoundError(f"Trained head not found: {head_path}")
    try:
        state = torch.load(head_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        model.classifier_head.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError('Failed to load classifier head: ' + str(e))

    # Transform
    import torchvision.transforms as T
    dino_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    # Collect image files
    valid_exts = {'.bmp', '.png', '.jpg', '.jpeg'}
    image_paths = [p for p in input_path.rglob('*') if p.suffix.lower() in valid_exts]
    raw_predictions = []

    for img_p in image_paths:
        if not segmenter.load_image(str(img_p)):
            continue
        cell =  get_grid_cell_from_name(img_p.name)
        if not segmenter.select_crop_region(cell):
            continue
        result = segmenter.add_text_prompt('building')
        if not result or len(result.get('masks',[]))==0:
            continue
        crop = segmenter.current_crop.copy()
        h_crop, w_crop = crop.shape[:2]
        bboxes = []
        for mask in result['masks'][:2]:
            mask_np = mask[0].cpu().numpy() if hasattr(mask[0],'cpu') else mask[0]
            if mask_np.ndim==3:
                mask_np = mask_np[0]
            mask_uint = (mask_np>0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask_resized>0)
            if xs.size==0: continue
            x0,y0 = int(xs.min()), int(ys.min())
            x1,y1 = int(xs.max())+1, int(ys.max())+1
            bboxes.append([x0,y0,x1,y1])
        if not bboxes: continue
        final_image = directional_crop_and_pad(crop, bboxes, target_size=tuple(config.get('target_size',(200,200))), shift_left_pixels=config.get('shift_left_pixels',50))

        # run through classifier
        input_tensor = dino_transform(Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        patch_mask = get_horizontal_patch_mask_from_array(final_image).unsqueeze(0).to(device)
        lap = get_laplacian_tensor_from_array(final_image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor, patch_mask, lap)
            probs = torch.softmax(logits, dim=1)[0]

        # pick ok index from config
        ok_idx = config.get('ok_class_index', 1)
        ok_prob = probs[ok_idx].item()
        if ok_prob >= config.get('confidence_gate', 0.9):
            final_pred = 'o'
            conf = ok_prob
            action = 'Pass (Confirmed OK)'
        else:
            defect_idxs = [i for i in range(len(probs)) if i!=ok_idx]
            defect_probs = probs[defect_idxs]
            max_idx = defect_idxs[int(torch.argmax(defect_probs).item())]
            final_pred = config.get('id_to_label', {0:'f',1:'o',2:'sn',3:'n'}).get(max_idx, str(max_idx))
            conf = probs[max_idx].item()
            action = 'Review (Flagged for Double Check)'

        raw_predictions.append({
            'SN': img_p.stem.replace('processed_','').split('-')[0],
            'Position': int(img_p.stem.replace('processed_','').split('-')[1]) if '-' in img_p.stem else 0,
            'Prediction': final_pred,
            'Confidence': round(conf*100,2),
            'Action': action,
            'image_path': str(img_p)
        })

        # optionally save final_image to output folder per-SN
        out_path = output_path / f"processed_{img_p.name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)).save(out_path)

    # pivot to consolidated matrix
    df_raw = pd.DataFrame(raw_predictions)
    target_positions = config.get('target_positions', [1,3,5,7,9])
    consolidated = []
    for sn, group in df_raw.groupby('SN'):
        row = {'SN': sn}
        review=False
        for pos in target_positions:
            rec = group[group['Position']==pos]
            if not rec.empty:
                row[f'pos {pos} predict'] = rec.iloc[0]['Prediction']
                row[f'pos {pos} confidence'] = f"{rec.iloc[0]['Confidence']}%"
                if 'Review' in rec.iloc[0]['Action']:
                    review=True
            else:
                row[f'pos {pos} predict'] = 'Missing'
                row[f'pos {pos} confidence'] = 'N/A'
        row['Action Required'] = 'Review (Flagged for Double Check)' if review else 'Pass (Confirmed OK)'
        consolidated.append(row)
    df_final = pd.DataFrame(consolidated)

    # Save full CSV/Excel
    full_csv = output_path / config.get('output_csv','consolidated_batch_predictions.csv')
    full_xlsx = output_path / config.get('output_excel','consolidated_batch_predictions.xlsx')
    df_final.to_csv(full_csv, index=False)
    df_final.to_excel(full_xlsx, index=False)

    # Apply filtering rules: any 'n' OR (>= SN_COUNT_REQUIRED sn with confidence >= SN_COUNT_THRESHOLD)
    paired = [(f'pos {p} predict', f'pos {p} confidence') for p in target_positions]
    # build pred and conf frames
    pred_frame = pd.DataFrame({pred: df_final[pred] for pred,_ in paired})
    conf_frame = pd.DataFrame({conf: pd.to_numeric(df_final[conf].str.replace('%',''), errors='coerce') for _,conf in paired})
    has_n = pred_frame.eq('n').any(axis=1)
    sn_count_thresh = config.get('sn_count_threshold_percent', 35)
    sn_count_required = config.get('sn_count_required', 3)
    sn_single_thresh = config.get('sn_single_threshold_percent', 40)
    sn_mask_count = pd.DataFrame({pred: (pred_frame[pred]=='sn') & (conf_frame[conf]>=sn_count_thresh) for pred,conf in paired}).sum(axis=1) >= sn_count_required
    sn_mask_single = pd.DataFrame({pred: (pred_frame[pred]=='sn') & (conf_frame[conf]>=sn_single_thresh) for pred,conf in paired}).any(axis=1)
    selected_mask = has_n | sn_mask_count | sn_mask_single
    df_filtered = df_final[selected_mask].copy()

    filtered_csv = output_path / config.get('filtered_csv','predictions_with_n_or_three_high_conf_sn.csv')
    filtered_xlsx = output_path / config.get('filtered_excel','predictions_with_n_or_three_high_conf_sn.xlsx')
    df_filtered.to_csv(filtered_csv, index=False)
    df_filtered.to_excel(filtered_xlsx, index=False)

    # Save filtered images (copy originals)
    for _, row in df_filtered.iterrows():
        sn = row['SN']
        for pos in target_positions:
            imgname = f"{sn}-{pos}"
            # find matching in df_raw
            match = df_raw[(df_raw['SN']==sn) & (df_raw['Position']==pos)]
            if not match.empty:
                src = Path(match.iloc[0]['image_path'])
                dst_dir = out_filtered / str(sn)
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst_dir / src.name)

    return {
        'full_csv': str(full_csv),
        'full_xlsx': str(full_xlsx),
        'filtered_csv': str(filtered_csv),
        'filtered_xlsx': str(filtered_xlsx),
        'filtered_images_folder': str(out_filtered)
    }

def get_grid_cell_from_name(filename):
    name, _ = os.path.splitext(filename)
    parts = name.split('-')
    if len(parts) > 1 and parts[-1].isdigit():
        cell_num = int(parts[-1])
        if 1 <= cell_num <= 9:
            return cell_num
    return 1

def main():
    st.title('Blur Detection Batch Processor')
    st.markdown('Combine SAM3 segmentation + DINOv3-based classifier for batch predictions')

    st.sidebar.header('Paths & Config')
    input_dir = st.sidebar.text_input('Input image folder', value=str(cfg.get('default_input','/home/vilota/566-qa-2/621D/IMG')))
    output_dir = st.sidebar.text_input('Output folder (processed + csv/xlsx)', value=str(Path.home()/ '566-qa-2' / 'processed_output'))
    filtered_images = st.sidebar.text_input('Filtered images folder', value=str(Path.home()/ '566-qa-2' / 'filtered_images'))
    cfg['trained_head_path'] = st.sidebar.text_input('Trained head .pth', value=cfg.get('trained_head_path',''))
    cfg['dino_backbone_weights'] = st.sidebar.text_input('DINOv3 backbone weights (optional)', value=cfg.get('dino_backbone_weights',''))
    run = st.sidebar.button('Run Batch')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f'Using device: {device}')

    if run:
        try:
            with st.spinner('Processing images...'):
                results = process_and_predict(input_dir, output_dir, filtered_images, cfg, device)
            st.success('Processing complete')
            st.write('Files saved:')
            st.write(results)
        except Exception as e:
            st.error(str(e))

if __name__ == '__main__':
    main()
import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import yaml

# -----------------------------
# Load configuration
# -----------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path=CONFIG_PATH):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

# -----------------------------
# Utilities (crop / grid / CLAHE)
# -----------------------------
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        standardized = clahe.apply(gray)
        return cv2.cvtColor(standardized, cv2.COLOR_GRAY2BGR)
    else:
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
        return self.image[y0:y1, x0:x1].copy()

# -----------------------------
# SAM3 wrapper (attempt import)
# -----------------------------
class Sam3BuildingSegmenter:
    def __init__(self, model_path=None):
        self.current_image = None
        self.current_crop = None
        self.grid_selector = None
        self.processor = None
        self._init_model(model_path)

    def _init_model(self, model_path=None):
        try:
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            if model_path is None:
                model_path = f"{sam3_root}/sam3/assets/sam3.pt"

            self.model = build_sam3_image_model(checkpoint_path=model_path, bpe_path=bpe_path)
            self.processor = Sam3Processor(self.model)
        except Exception as e:
            raise ImportError("sam3 not available or failed to load: " + str(e))

    def load_image(self, image_path: str) -> bool:
        try:
            img_pil = Image.open(image_path).convert("RGB")
            self.current_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.grid_selector = CropGridSelector(self.current_image)
            return True
        except Exception:
            return False

    def select_crop_region(self, cell_num: int) -> bool:
        try:
            self.current_crop = self.grid_selector.get_crop(cell_num)
            return True
        except Exception:
            return False

    def add_text_prompt(self, text: str):
        try:
            crop_pil = Image.fromarray(cv2.cvtColor(self.current_crop, cv2.COLOR_BGR2RGB))
            state = self.processor.set_image(crop_pil)
            state = self.processor.set_text_prompt(text, state)
            return {"masks": state.get("masks", [])}
        except Exception:
            return None

# -----------------------------
# Directional crop (uses configurable left shift)
# -----------------------------
def directional_crop_and_pad(image, bboxes, target_size=(200, 200), bg_color=(0, 0, 0), shift_left_pixels=50):
    img_h, img_w = image.shape[:2]
    target_w, target_h = target_size
    far_bbox = max(bboxes, key=lambda b: b[2])
    f_x0, f_y0, f_x1, f_y1 = far_bbox
    anchor_right_x = f_x1
    anchor_center_y = (f_y0 + f_y1) // 2
    crop_x1 = anchor_right_x - shift_left_pixels
    crop_x0 = crop_x1 - target_w
    crop_y0 = anchor_center_y - (target_h // 2)
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

# -----------------------------
# Blur classifier (depends on dinov3)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class BlurClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes=4, laplacian_input_dim=196):
        super().__init__()
        self.backbone = backbone
        linear_input_dim = 2 * embed_dim
        self.laplacian_projector = nn.Sequential(
            nn.Linear(laplacian_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        fused_input_dim = linear_input_dim + 128
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            ResidualBlock(2048, 1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, patch_mask, laplacian_tensor):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        mask_weights = patch_mask.float().unsqueeze(-1)
        weighted_patches = patch_tokens * mask_weights
        summed_patches = weighted_patches.sum(dim=1)
        valid_patch_count = mask_weights.sum(dim=1) + 1e-6
        masked_patch_mean = summed_patches / valid_patch_count
        dino_feature = torch.cat([cls_token, masked_patch_mean], dim=1)
        pooled_laplacian = F.adaptive_avg_pool2d(laplacian_tensor, (14, 14))
        flat_laplacian = pooled_laplacian.view(pooled_laplacian.size(0), -1)
        laplacian_features = self.laplacian_projector(flat_laplacian)
        fused_input = torch.cat([dino_feature, laplacian_features], dim=1)
        logits = self.classifier_head(fused_input)
        return logits

# -----------------------------
# Helpers: patch mask and laplacian from numpy image
# -----------------------------
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
    gray = pil.convert('L')
    gray_t = torch.from_numpy(np.array(gray)).float().unsqueeze(0).unsqueeze(0) / 255.0
    lap_kernel = torch.tensor([[[[0,1,0],[0,-2,0],[0,1,0]]]], dtype=torch.float32)
    lap = F.conv2d(gray_t, lap_kernel, padding=1)
    return torch.abs(lap)

# -----------------------------
# Main processing and Streamlit UI
# -----------------------------
def process_and_predict(input_folder, output_folder, filtered_images_folder, config, device):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    out_filtered = Path(filtered_images_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    out_filtered.mkdir(parents=True, exist_ok=True)

    # Initialize SAM3 segmenter
    try:
        segmenter = Sam3BuildingSegmenter(model_path=config.get('sam3_checkpoint'))
    except Exception as e:
        raise RuntimeError('Failed to initialize SAM3: ' + str(e))

    # Initialize DINOv3 backbone if available
    try:
        from dinov3.hub.backbones import dinov3_vith16plus, Weights
        weights_arg = config.get('dino_backbone_weights') or Weights.LVD1689M
        backbone = dinov3_vith16plus(pretrained=True, weights=weights_arg)
    except Exception as e:
        raise RuntimeError('Failed to import or build dinov3 backbone: ' + str(e))

    embed_dim = backbone.embed_dim
    model = BlurClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=config.get('num_classes',4))
    model = model.to(device)

    head_path = config.get('trained_head_path')
    if not Path(head_path).exists():
        raise FileNotFoundError(f"Trained head not found: {head_path}")
    try:
        state = torch.load(head_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        model.classifier_head.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError('Failed to load classifier head: ' + str(e))

    import torchvision.transforms as T
    dino_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    valid_exts = {'.bmp', '.png', '.jpg', '.jpeg'}
    image_paths = [p for p in input_path.rglob('*') if p.suffix.lower() in valid_exts]
    raw_predictions = []

    for img_p in image_paths:
        if not segmenter.load_image(str(img_p)):
            continue
        cell =  get_grid_cell_from_name(img_p.name)
        if not segmenter.select_crop_region(cell):
            continue
        result = segmenter.add_text_prompt('building')
        if not result or len(result.get('masks',[]))==0:
            continue
        crop = segmenter.current_crop.copy()
        h_crop, w_crop = crop.shape[:2]
        bboxes = []
        for mask in result['masks'][:2]:
            mask_np = mask[0].cpu().numpy() if hasattr(mask[0],'cpu') else mask[0]
            if mask_np.ndim==3:
                mask_np = mask_np[0]
            mask_uint = (mask_np>0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask_resized>0)
            if xs.size==0: continue
            x0,y0 = int(xs.min()), int(ys.min())
            x1,y1 = int(xs.max())+1, int(ys.max())+1
            bboxes.append([x0,y0,x1,y1])
        if not bboxes: continue
        final_image = directional_crop_and_pad(crop, bboxes, target_size=tuple(config.get('target_size',(200,200))), shift_left_pixels=config.get('shift_left_pixels',50))

        input_tensor = dino_transform(Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        patch_mask = get_horizontal_patch_mask_from_array(final_image).unsqueeze(0).to(device)
        lap = get_laplacian_tensor_from_array(final_image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor, patch_mask, lap)
            probs = torch.softmax(logits, dim=1)[0]

        ok_idx = config.get('ok_class_index', 1)
        ok_prob = probs[ok_idx].item()
        if ok_prob >= config.get('confidence_gate', 0.9):
            final_pred = 'o'
            conf = ok_prob
            action = 'Pass (Confirmed OK)'
        else:
            defect_idxs = [i for i in range(len(probs)) if i!=ok_idx]
            defect_probs = probs[defect_idxs]
            max_idx = defect_idxs[int(torch.argmax(defect_probs).item())]
            final_pred = config.get('id_to_label', {0:'f',1:'o',2:'sn',3:'n'}).get(max_idx, str(max_idx))
            conf = probs[max_idx].item()
            action = 'Review (Flagged for Double Check)'

        raw_predictions.append({
            'SN': img_p.stem.replace('processed_','').split('-')[0],
            'Position': int(img_p.stem.replace('processed_','').split('-')[1]) if '-' in img_p.stem else 0,
            'Prediction': final_pred,
            'Confidence': round(conf*100,2),
            'Action': action,
            'image_path': str(img_p)
        })

        out_path = output_path / f"processed_{img_p.name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)).save(out_path)

    df_raw = pd.DataFrame(raw_predictions)
    target_positions = config.get('target_positions', [1,3,5,7,9])
    consolidated = []
    for sn, group in df_raw.groupby('SN'):
        row = {'SN': sn}
        review=False
        for pos in target_positions:
            rec = group[group['Position']==pos]
            if not rec.empty:
                row[f'pos {pos} predict'] = rec.iloc[0]['Prediction']
                row[f'pos {pos} confidence'] = f"{rec.iloc[0]['Confidence']}%"
                if 'Review' in rec.iloc[0]['Action']:
                    review=True
            else:
                row[f'pos {pos} predict'] = 'Missing'
                row[f'pos {pos} confidence'] = 'N/A'
        row['Action Required'] = 'Review (Flagged for Double Check)' if review else 'Pass (Confirmed OK)'
        consolidated.append(row)
    df_final = pd.DataFrame(consolidated)

    full_csv = output_path / config.get('output_csv','consolidated_batch_predictions.csv')
    full_xlsx = output_path / config.get('output_excel','consolidated_batch_predictions.xlsx')
    df_final.to_csv(full_csv, index=False)
    df_final.to_excel(full_xlsx, index=False)

    paired = [(f'pos {p} predict', f'pos {p} confidence') for p in target_positions]
    pred_frame = pd.DataFrame({pred: df_final[pred] for pred,_ in paired})
    conf_frame = pd.DataFrame({conf: pd.to_numeric(df_final[conf].str.replace('%',''), errors='coerce') for _,conf in paired})
    has_n = pred_frame.eq('n').any(axis=1)
    sn_count_thresh = config.get('sn_count_threshold_percent', 35)
    sn_count_required = config.get('sn_count_required', 3)
    sn_single_thresh = config.get('sn_single_threshold_percent', 40)
    sn_mask_count = pd.DataFrame({pred: (pred_frame[pred]=='sn') & (conf_frame[conf]>=sn_count_thresh) for pred,conf in paired}).sum(axis=1) >= sn_count_required
    sn_mask_single = pd.DataFrame({pred: (pred_frame[pred]=='sn') & (conf_frame[conf]>=sn_single_thresh) for pred,conf in paired}).any(axis=1)
    selected_mask = has_n | sn_mask_count | sn_mask_single
    df_filtered = df_final[selected_mask].copy()

    filtered_csv = output_path / config.get('filtered_csv','predictions_with_n_or_three_high_conf_sn.csv')
    filtered_xlsx = output_path / config.get('filtered_excel','predictions_with_n_or_three_high_conf_sn.xlsx')
    df_filtered.to_csv(filtered_csv, index=False)
    df_filtered.to_excel(filtered_xlsx, index=False)

    for _, row in df_filtered.iterrows():
        sn = row['SN']
        for pos in target_positions:
            match = df_raw[(df_raw['SN']==sn) & (df_raw['Position']==pos)]
            if not match.empty:
                src = Path(match.iloc[0]['image_path'])
                dst_dir = out_filtered / str(sn)
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst_dir / src.name)

    return {
        'full_csv': str(full_csv),
        'full_xlsx': str(full_xlsx),
        'filtered_csv': str(filtered_csv),
        'filtered_xlsx': str(filtered_xlsx),
        'filtered_images_folder': str(out_filtered)
    }

def get_grid_cell_from_name(filename):
    name, _ = os.path.splitext(filename)
    parts = name.split('-')
    if len(parts) > 1 and parts[-1].isdigit():
        cell_num = int(parts[-1])
        if 1 <= cell_num <= 9:
            return cell_num
    return 1

def main():
    st.title('Blur Detection Batch Processor')
    st.markdown('Combine SAM3 segmentation + DINOv3-based classifier for batch predictions')

    st.sidebar.header('Paths & Config')
    input_dir = st.sidebar.text_input('Input image folder', value=str(cfg.get('default_input','/home/vilota/566-qa-2/621D/IMG')))
    output_dir = st.sidebar.text_input('Output folder (processed + csv/xlsx)', value=str(Path.home()/ '566-qa-2' / 'processed_output'))
    filtered_images = st.sidebar.text_input('Filtered images folder', value=str(Path.home()/ '566-qa-2' / 'filtered_images'))
    cfg['trained_head_path'] = st.sidebar.text_input('Trained head .pth', value=cfg.get('trained_head_path',''))
    cfg['dino_backbone_weights'] = st.sidebar.text_input('DINOv3 backbone weights (optional)', value=cfg.get('dino_backbone_weights',''))
    run = st.sidebar.button('Run Batch')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f'Using device: {device}')

    if run:
        try:
            with st.spinner('Processing images...'):
                results = process_and_predict(input_dir, output_dir, filtered_images, cfg, device)
            st.success('Processing complete')
            st.write('Files saved:')
            st.write(results)
        except Exception as e:
            st.error(str(e))

if __name__ == '__main__':
    main()
