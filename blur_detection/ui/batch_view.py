import streamlit as st
import torch
from pathlib import Path

from core.config import DEFAULT_MODEL_PATHS
from core.pipeline import process_and_predict
from models.sam3_wrapper import Sam3BuildingSegmenter
from models.dino_classifier import BlurClassifier

LANG_BATCH = {
    "EN": {
        "header": "Batch Process",
        "in_dir": "Input image folder",
        "out_dir": "Processed Images Output folder",
        "filt_dir": "Filtered output folder (Images & Datasets)",
        "device": "Using device",
        "run_btn": "Run Batch",
        "loading": "Loading models...",
        "success": "Batch Processing Complete!",
        "analysis": "Result Analysis",
        "breakdown": "Flagged Units Breakdown by Reason",
        "saved_files": "Files saved:",
        "m1": "Total Units (SNs)", "m2": "Filtered Units (Flagged)", "m3": "Filtered Percentage", "m4": "Processing Time", "m5": "Avg Time/Image",
        "err_title": "⚠️ FLC Required (Error/Warning)",
        "err_desc": "*Reason: Pipeline processing failures, missing expected position views, name parsing exceptions, or SAM3 building localization timeouts.*",
        "warn_title": "🔍 Review (Flagged for Double Check)",
        "warn_desc": "*Reason: Successful pipeline processing, but triggered by low model confidence scores or classification defect tags requiring operator validation.*"
    },
    "ZH": {
        "header": "批量处理",
        "in_dir": "输入图像文件夹路径",
        "out_dir": "处理后图像输出文件夹路径",
        "filt_dir": "过滤输出文件夹 (图像与数据集)",
        "device": "当前运行设备",
        "run_btn": "运行批量处理",
        "loading": "模型加载中...",
        "success": "批处理执行完毕！",
        "analysis": "结果分析报告",
        "breakdown": "标记单元原因细分",
        "saved_files": "保存的文件：",
        "m1": "总单元数 (SN码)", "m2": "过滤单元数 (触发标记)", "m3": "拦截率", "m4": "耗时", "m5": "平均单张耗时",
        "err_title": "⚠️ 需要 FLC (错误/警告)",
        "err_desc": "*原因：流水线处理失败、缺失指定位置视图、命名解析异常或 SAM3 建筑定位超时。*",
        "warn_title": "🔍 待审查 (标记人工复核)",
        "warn_desc": "*原因：算法流水线正常运行，但模型置信度得分较低或触发缺陷分类标签，需要人工操作员确认。*"
    }
}

@st.cache_resource(show_spinner="Loading SAM3 Model...")
def load_sam3_model(sam3_ckpt, bpe_path):
    return Sam3BuildingSegmenter(model_path=sam3_ckpt, bpe_path=bpe_path)

@st.cache_resource(show_spinner="Loading DINOv3 Classifier...")
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
    lang = st.session_state.get("lang", "EN")
    ln = LANG_BATCH[lang]
    
    st.header(ln["header"])
    
    input_dir = st.text_input(ln["in_dir"], value=str(st.session_state.app_config.get("default_input", "/home/vilota/566-qa-2/600D/IMG")))
    output_dir = st.text_input(ln["out_dir"], value=str(Path.home() / "566-qa-2" / "processed_output" / "600D"))
    filtered_images = st.text_input(ln["filt_dir"], value=str(Path.home() / "566-qa-2" / "filtered_images" / "600D"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"{ln['device']}: **{device}**")

    if st.button(ln["run_btn"], type="primary"):
        try:
            with st.spinner(ln["loading"]):
                sam_ckpt = st.session_state.app_config.get("sam3_checkpoint", DEFAULT_MODEL_PATHS["sam3_checkpoint"])
                bpe = st.session_state.app_config.get("bpe_path", DEFAULT_MODEL_PATHS["bpe_path"])
                dino_w = st.session_state.app_config.get("dino_backbone_weights", DEFAULT_MODEL_PATHS["dino_backbone_weights"])
                head_p = st.session_state.app_config.get("trained_head_path", DEFAULT_MODEL_PATHS["trained_head_path"])
                num_cls = st.session_state.app_config.get("num_classes", 4)
                
                segmenter = load_sam3_model(sam_ckpt, bpe)
                dino_model = load_dino_model(dino_w, head_p, num_cls, device)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        try:
            results = process_and_predict(
                input_dir, output_dir, filtered_images, 
                st.session_state.app_config, device, segmenter, dino_model
            )
            
            st.success(ln["success"])
            st.write(f"### {ln['analysis']}")
            
            total = results.get("total_units", 0)
            filtered = results.get("units_flagged_for_review", 0)
            percentage = (filtered / total * 100) if total > 0 else 0.0
            process_time = results.get("process_time_seconds", 0)
            avg_time = results.get("avg_time_per_image_seconds", 0)

            mins, secs = divmod(process_time, 60)
            time_str = f"{int(mins)}m {secs:.1f}s" if mins > 0 else f"{process_time:.2f}s"

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric(ln["m1"], total)
            with c2: st.metric(ln["m2"], filtered)
            with c3: st.metric(ln["m3"], f"{percentage:.2f}%")
            with c4: st.metric(ln["m4"], time_str)
            with c5: st.metric(ln["m5"], f"{avg_time:.2f}s")
            
            st.write(f"#### {ln['breakdown']}")
            flc_err = results.get("flc_error_warning_count", 0)
            rev_check = results.get("review_double_check_count", 0)
            
            col_reason1, col_reason2 = st.columns(2)
            with col_reason1:
                st.info(f"{ln['err_title']}: `{flc_err}`\n\n{ln['err_desc']}")
            with col_reason2:
                st.warning(f"{ln['warn_title']}: `{rev_check}`\n\n{ln['warn_desc']}")

            st.divider()
            st.write(f"**{ln['saved_files']}**")
            st.json({k: v for k, v in results.items() if k not in [
                "total_units", "units_flagged_for_review", "process_time_seconds", 
                "avg_time_per_image_seconds", "flc_error_warning_count", "review_double_check_count"
            ]})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")