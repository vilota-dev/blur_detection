import yaml
from pathlib import Path

# -----------------------------
# Path Configurations
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent 
MODELS_DIR = SCRIPT_DIR / "assets"

DEFAULT_MODEL_PATHS = {
    "bpe_path": str(MODELS_DIR / "bpe_simple_vocab_16e6.txt.gz"),
    "trained_head_path": str(MODELS_DIR / "dino_classifier_head_multiclass_epoch_58.pth"),
    "dino_backbone_weights": str(MODELS_DIR / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    "sam3_checkpoint": str(MODELS_DIR / "sam3.pt"),
}

CONFIG_PATH = SCRIPT_DIR / "config.yaml"

def load_config(path=CONFIG_PATH):
    cfg = {}
    if path.exists():
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                cfg.update(loaded)
    for k, v in DEFAULT_MODEL_PATHS.items():
        cfg.setdefault(k, v)
    return cfg

cfg = load_config()