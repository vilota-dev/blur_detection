import streamlit as st
import torch
from pathlib import Path

from core.config import DEFAULT_MODEL_PATHS
from core.pipeline import process_and_predict
from models.sam3_wrapper import Sam3BuildingSegmenter
from models.dino_classifier import BlurClassifier

@st.cache_resource(show_spinner="Loading SAM3 Model (this only happens once)...")
def load_sam3_model(sam3_ckpt, bpe_path):
    return Sam3BuildingSegmenter(model_path=sam3_ckpt, bpe_path=bpe_path)

@st.cache_resource(show_spinner="Loading DINOv3 Classifier (this only happens once)...")
def load_dino_model(dino_weights, head_path, num_classes, _device):
    from dinov3.hub.backbones import dinov3_vith16plus
    backbone = dinov3_vith16plus(pretrained=True, weights=dino_weights)
    embed_dim = backbone.embed_dim
    model = BlurClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=num_classes).to(_device)
    
    if not Path(head_path).exists():
        raise FileNotFoundError(f"Trained head not found: {head_path}")
    
    state = torch.load(head_path, map_location=_device)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    model.classifier_head.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def render_batch_tab():
    st.header("Batch Process")
    
    # UI Inputs (CSV/XLSX paths are handled internally to map to filtered_images)
    input_dir = st.text_input("Input image folder", value=str(st.session_state.app_config.get("default_input", "/home/vilota/566-qa-2/621D/IMG")))
    output_dir = st.text_input("Processed Images Output folder", value=str(Path.home() / "566-qa-2" / "processed_output"))
    filtered_images = st.text_input("Filtered output folder (Images & Datasets)", value=str(Path.home() / "566-qa-2" / "filtered_images"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: **{device}**")

    if st.button("Run Batch", type="primary"):
        try:
            with st.spinner("Loading models..."):
                sam_ckpt = st.session_state.app_config.get("sam3_checkpoint", DEFAULT_MODEL_PATHS["sam3_checkpoint"])
                bpe = st.session_state.app_config.get("bpe_path", DEFAULT_MODEL_PATHS["bpe_path"])
                dino_w = st.session_state.app_config.get("dino_backbone_weights", DEFAULT_MODEL_PATHS["dino_backbone_weights"])
                head_p = st.session_state.app_config.get("trained_head_path", DEFAULT_MODEL_PATHS["trained_head_path"])
                num_cls = st.session_state.app_config.get("num_classes", 4)
                
                segmenter = load_sam3_model(sam_ckpt, bpe)
                dino_model = load_dino_model(dino_w, head_p, num_cls, device)
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()

        try:
            results = process_and_predict(
                input_dir, output_dir, filtered_images, 
                st.session_state.app_config, device, segmenter, dino_model
            )
            st.success("Batch Processing Complete!")
            st.write("**Files saved:**")
            st.json(results)
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")