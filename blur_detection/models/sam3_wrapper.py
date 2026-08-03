import cv2
import numpy as np
import torch
from PIL import Image
from core.config import cfg
from utils.image_utils import CropGridSelector

class Sam3BuildingSegmenter:
    def __init__(self, model_path=None, bpe_path=None):
        self.processor = None
        self._init_model(model_path, bpe_path)

    def _init_model(self, model_path=None, bpe_path=None):
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model_path = model_path or cfg.get("sam3_checkpoint")
            bpe_path = bpe_path or cfg.get("bpe_path")

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            self.model = build_sam3_image_model(
                checkpoint_path=model_path,
                bpe_path=bpe_path,
            )

            self.model = self.model.to("cuda")
            self.model.eval()

            # Compile the backbone encoder to optimize performance if enabled in config
            if cfg.get("compile_sam3_backbone", False):
                self.model.backbone = torch.compile(self.model.backbone, mode="reduce-overhead")

            # Initialize Sam3Processor with resolution from config (default to 1008)
            resolution = cfg.get("sam3_resolution", 1008)
            self.processor = Sam3Processor(self.model, resolution=resolution)

        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            raise ImportError("sam3 not available or failed to load: " + str(e))

    def load_image(self, image_path: str) -> np.ndarray:
        try:
            return cv2.imread(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def select_crop_region(self, image: np.ndarray, cell_num: int, config: dict = None) -> np.ndarray:
        try:
            grid_selector = CropGridSelector(image, config=config)
            return grid_selector.get_crop(cell_num)
        except Exception as e:
            print(f"Error selecting crop: {e}")
            return None

    def add_text_prompt(self, crop: np.ndarray, text: str):
        try:
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(crop_pil)
                state = self.processor.set_text_prompt(text, state)
                return {"masks": state.get("masks", [])}
        except Exception as e:
            print(f"Error in text prompt: {e}")
            return None