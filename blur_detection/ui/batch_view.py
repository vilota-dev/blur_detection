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
    
    input_dir = st.text_input("Input image folder", value=str(st.session_state.app_config.get("default_input", "/home/vilota/566-qa-2/600D/IMG")))
    output_dir = st.text_input("Processed Images Output folder", value=str(Path.home() / "566-qa-2" / "processed_output" / "600D"))
    filtered_images = st.text_input("Filtered output folder (Images & Datasets)", value=str(Path.home() / "566-qa-2" / "filtered_images" / "600D"))

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
            
            # --- Result Analysis Summary ---
            st.write("### Result Analysis")
            total = results.get("total_units", 0)
            filtered = results.get("units_flagged_for_review", 0)
            percentage = (filtered / total * 100) if total > 0 else 0.0
            process_time = results.get("process_time_seconds", 0)
            avg_time = results.get("avg_time_per_image_seconds", 0)

            mins, secs = divmod(process_time, 60)
            time_str = f"{int(mins)}m {secs:.1f}s" if mins > 0 else f"{process_time:.2f}s"

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Total Units (SNs)", total)
            with c2:
                st.metric("Filtered Units (Flagged)", filtered)
            with c3:
                st.metric("Filtered Percentage", f"{percentage:.2f}%")
            with c4:
                st.metric("Processing Time", time_str)
            with c5:
                st.metric("Avg Time/Image", f"{avg_time:.2f}s")
            
            # --- DETAILED REASON METRICS DISPLAY SHORTCUT ---
            st.write("#### Flagged Units Breakdown by Reason")
            flc_err = results.get("flc_error_warning_count", 0)
            rev_check = results.get("review_double_check_count", 0)
            
            col_reason1, col_reason2 = st.columns(2)
            with col_reason1:
                st.info(f"⚠️ **FLC Required (Error/Warning):** `{flc_err}` units\n\n*Reason: Pipeline processing failures, missing expected position views, name parsing exceptions, or SAM3 building localization timeouts.*")
            with col_reason2:
                st.warning(f"🔍 **Review (Flagged for Double Check):** `{rev_check}` units\n\n*Reason: Successful pipeline processing, but triggered by low model confidence scores or classification defect tags requiring operator validation.*")

            st.divider()
            # -----------------------------------
            
            st.write("**Files saved:**")
            # Hide raw metadata count hooks from json display output
            st.json({k: v for k, v in results.items() if k not in [
                "total_units", "units_flagged_for_review", "process_time_seconds", 
                "avg_time_per_image_seconds", "flc_error_warning_count", "review_double_check_count"
            ]})
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")