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

def resolve_path(p_str):
    if not p_str:
        return p_str
    p = Path(p_str)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    
    script_rel = SCRIPT_DIR / p
    if script_rel.exists():
        return str(script_rel.resolve())
    
    p_s = str(p_str)
    if p_s.startswith("blur_detection/"):
        stripped = SCRIPT_DIR / p_s.replace("blur_detection/", "", 1)
        if stripped.exists():
            return str(stripped.resolve())
            
    by_filename = MODELS_DIR / p.name
    if by_filename.exists():
        return str(by_filename.resolve())

    return str((SCRIPT_DIR / p).resolve())

def load_config(path=CONFIG_PATH):
    cfg = {}
    if path.exists():
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                cfg.update(loaded)
    for k, v in DEFAULT_MODEL_PATHS.items():
        cfg.setdefault(k, v)
        
    path_keys = ["bpe_path", "trained_head_path", "dino_backbone_weights", "sam3_checkpoint"]
    for k in path_keys:
        if k in cfg:
            cfg[k] = resolve_path(cfg[k])

    return cfg

cfg = load_config()