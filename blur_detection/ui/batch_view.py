import streamlit as st
import torch
import yaml
from pathlib import Path

from core.config import DEFAULT_MODEL_PATHS, CONFIG_PATH
from core.pipeline import process_and_predict
from models.sam3_wrapper import Sam3BuildingSegmenter
from models.dino_classifier import BlurClassifier

LANG_BATCH = {
    "EN": {
        "header": "Batch Process Execution Pipeline",
        "info": "Configure your input data sources and output destinations below. Changes to paths will automatically sync to system configuration presets.",
        "in_dir": "📁 Input Raw Image Folder Path",
        "out_dir": "🎯 Result Output Root Folder Path",
        "device": "Current computing device context",
        "run_btn": "🚀 Run Batch Pipeline Process",
        "loading": "Loading neural network architectures & loading state_dicts securely...",
        "success": "Batch Processing Execution Completed Successfully!",
        "analysis": "Operational Analytics Report Summary",
        "breakdown": "Flagged Review Units Breakdown by Pipeline Reason",
        "saved_files": "Exported Manifest Mapping Reports:",
        "m1": "Total Units (SNs)", "m2": "Filtered Units (Flagged)", "m3": "Filtered Percentage", "m4": "Processing Time", "m5": "Avg Time/Image",
        "err_title": "⚠️ FLC Required (Error/Warning)",
        "err_desc": "*Reason: Pipeline processing failures, missing expected position views, name parsing exceptions, or SAM3 building localization timeouts.*",
        "warn_title": "🔍 Review (Flagged for Double Check)",
        "warn_desc": "*Reason: Successful pipeline processing, but triggered by low model confidence scores or classification defect tags requiring operator validation.*"
    },
    "ZH": {
        "header": "批量数据处理控制台",
        "info": "在下方配置您的输入数据源和输出目标路径。路径变更将自动同步写入本地系统配置预设中。",
        "in_dir": "📁 原始输入图像文件夹路径",
        "out_dir": "🎯 预测结果输出根文件夹路径",
        "device": "当前运行设备上下文环境",
        "run_btn": "运行批量流清洗处理",
        "loading": "正在后台加载核心算法模型并执行参数矩阵拓扑映射...",
        "success": "批处理执行完毕！结构化文件生成成功。",
        "analysis": "结果分析报告指标看板",
        "breakdown": "标记缺陷单元原因细分统计",
        "saved_files": "已成功导出并落盘的数据资产清单：",
        "m1": "总单元数 (SN码)", "m2": "过滤单元数 (触发标记)", "m3": "拦截率", "m4": "耗时", "m5": "平均单张耗时",
        "err_title": "⚠️ 需要 FLC (错误/警告)",
        "err_desc": "*原因：流水线处理失败、缺失指定位置视图、命名解析异常或 SAM3 建筑定位超时。*",
        "warn_title": "🔍 待审查 (标记人工复核)",
        "warn_desc": "*原因：算法流水线正常运行，但模型置信度得分较低或触发缺陷分类标签，需要人工操作员确认。*"
    }
}

@st.cache_resource(show_spinner="Loading SAM3 Grounding Segmentation weights...")
def load_sam3_model(sam3_ckpt, bpe_path):
    return Sam3BuildingSegmenter(model_path=sam3_ckpt, bpe_path=bpe_path)

@st.cache_resource(show_spinner="Loading DINOv3 Backbone and fusing classifier heads...")
def load_dino_model(dino_weights, head_path, num_classes, _device):
    from dinov3.hub.backbones import dinov3_vith16plus
    backbone = dinov3_vith16plus(pretrained=True, weights=dino_weights)
    embed_dim = backbone.embed_dim
    model = BlurClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=num_classes).to(_device)
    
    if not Path(head_path).exists():
        raise FileNotFoundError(f"Trained head not found: {head_path}")
    
    state = torch.load(head_path, map_location=_device)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    
    # --- CRITICAL FIX FOR STATE_DICT MISMATCH ---
    # Strip 'classifier_head.' prefix if saved with wrapper namespace, 
    # or keep naked keys to line up with the sequential module accurately.
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("classifier_head.", "")
        cleaned_state_dict[new_key] = v

    # Target the inner sequential classification module with strict structural validation
    model.classifier_head.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    return model

def render_batch_tab():
    lang = st.session_state.get("lang", "EN")
    ln = LANG_BATCH[lang]
    
    app_cfg = st.session_state.app_config
    
    # Resolve default baseline paths from standard configuration metrics
    default_in = app_cfg.get("default_input", "/data/input")
    default_out_root = app_cfg.get("default_output_root", "/data/output")
    
    st.header(ln["header"])
    st.info(ln["info"])
    
    # Two-path minimalistic layout restriction
    input_dir = st.text_input(ln["in_dir"], value=str(default_in))
    output_root_dir = st.text_input(ln["out_dir"], value=str(default_out_root))

    # --- AUTOMATED CONFIGURATION SAVE & CROSS-TAB TRACKING SYNCHRONIZER ---
    if (input_dir != app_cfg.get("default_input") or output_root_dir != app_cfg.get("default_output_root")):
        app_cfg["default_input"] = input_dir
        app_cfg["default_output_root"] = output_root_dir
        st.session_state.app_config = app_cfg

        # Direct cross-tab tracking adjustments targeting Processed and Dataset boundaries
        st.session_state["review_csv_path"] = str(Path(output_root_dir) / "dataset_output" / "consolidated_batch_predictions.csv")
        st.session_state["review_img_folder"] = str(Path(output_root_dir) / "processed_images")

        # Overwrite physical disk configuration state variables instantly
        try:
            with open(CONFIG_PATH, "w") as f:
                yaml.safe_dump(app_cfg, f, default_flow_style=False)
            with open(CONFIG_PATH, "r") as f:
                st.session_state.yaml_text = f.read()
        except Exception as e:
            st.error(f"Failed to auto-save path presets to configuration yaml layer: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"{ln['device']}: **{device}**")

    if st.button(ln["run_btn"], type="primary"):
        try:
            with st.spinner(ln["loading"]):
                sam_ckpt = app_cfg.get("sam3_checkpoint", DEFAULT_MODEL_PATHS["sam3_checkpoint"])
                bpe = app_cfg.get("bpe_path", DEFAULT_MODEL_PATHS["bpe_path"])
                dino_w = app_cfg.get("dino_backbone_weights", DEFAULT_MODEL_PATHS["dino_backbone_weights"])
                head_p = app_cfg.get("trained_head_path", DEFAULT_MODEL_PATHS["trained_head_path"])
                num_cls = app_cfg.get("num_classes", 4)
                
                segmenter = load_sam3_model(sam_ckpt, bpe)
                dino_model = load_dino_model(dino_w, head_p, num_cls, device)
        except Exception as e:
            st.error(f"Model Initialization Failure: {e}")
            st.stop()

        try:
            # Process using your updated directory tracking signature argument values
            results = process_and_predict(
                input_folder=input_dir, 
                output_root=output_root_dir, 
                config=app_cfg, 
                device=device, 
                segmenter=segmenter, 
                model=dino_model
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
            st.error(f"Pipeline processing execution aborted: {str(e)}")