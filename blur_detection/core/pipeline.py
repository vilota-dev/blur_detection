import os
import shutil
import gc
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as T
import time
import json

from utils.image_utils import (
    directional_crop_and_pad, 
    get_horizontal_patch_mask_from_array, 
    get_laplacian_tensor_from_array
)
from utils.file_utils import get_grid_cell_from_name, extract_sn_and_pos

# --- 中文状态持久化映射字典 ---
STATUS_MAPPING_ZH = {
    "Pass (Program Pass)": "自动通过 (Pass)",
    "Pass (Human Reviewed)": "人工确认通过 (Pass)",
    "Review (Flagged for Double Check)": "待人工审查 (Review)",
    "FLC Required (Human Reviewed)": "人工确认复核 (FLC)",
    "FLC Required (Error/Warning)": "异常复核 (FLC)",
    "NG (Human Reviewed)": "人工确认报废 (NG)"
}

def process_and_predict(input_folder, output_root, config, device, segmenter, model):
    start_time = time.time()

    input_path = Path(input_folder)
    output_root_path = Path(output_root)
    
    # Define organized internal directory structure
    processed_images_dir = output_root_path / "processed_images"
    dataset_output_dir = output_root_path / "dataset_output"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_folder}")

    processed_images_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    dino_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    valid_exts = {".bmp", ".png", ".jpg", ".jpeg"}
    image_paths = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts]
    
    if not image_paths:
        raise RuntimeError(f"No images found in {input_folder}")

    raw_predictions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_images = len(image_paths)

    ID_TO_LABEL = {0: 'f', 1: 'o', 2: 'sn', 3: 'n'}
    pos_shifts = config.get("position_shifts", {})

    for idx, img_p in enumerate(image_paths):
        status_text.text(f"Processing {idx + 1}/{total_images}: {img_p.name}")
        
        try:
            sn, position = extract_sn_and_pos(img_p.name)
        except Exception as e:
            raw_predictions.append({
                "SN": "Unknown", "Position": "Unknown", "Prediction": "Error",
                "Confidence": 0.0, "Action": "Review", 
                "image_path": str(img_p), "Error Description": f"Filename identification failed: {e}"
            })
            continue

        if not segmenter.load_image(str(img_p)):
            raw_predictions.append({
                "SN": sn, "Position": position, "Prediction": "Error",
                "Confidence": 0.0, "Action": "Review", 
                "image_path": str(img_p), "Error Description": "Image could not be loaded or identified"
            })
            continue
            
        cell = get_grid_cell_from_name(img_p.name)
        if cell is None or not segmenter.select_crop_region(cell):
            raw_predictions.append({
                "SN": sn, "Position": position, "Prediction": "Error",
                "Confidence": 0.0, "Action": "Review", 
                "image_path": str(img_p), "Error Description": "Grid crop region selection failed"
            })
            continue

        result = segmenter.add_text_prompt("building")
        
        if not result or len(result.get("masks", [])) == 0:
            raw_predictions.append({
                "SN": sn, "Position": position, "Prediction": "Error",
                "Confidence": 0.0, "Action": "Review", 
                "image_path": str(img_p), "Error Description": "SAM3 segmentation failed to find building masks"
            })
            continue

        crop = segmenter.current_crop.copy()
        h_crop, w_crop = crop.shape[:2]
        bboxes = []
        
        for mask in result["masks"]:
            mask_np = mask[0].cpu().numpy() if hasattr(mask[0], "cpu") else mask[0]
            if mask_np.ndim == 3: mask_np = mask_np[0]
            mask_uint = (mask_np > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask_resized > 0)
            if xs.size == 0: continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
            bboxes.append([x0, y0, x1, y1])
            if len(bboxes) == 2:
                break

        if len(bboxes) < 2:
            raw_predictions.append({
                "SN": sn, "Position": position, "Prediction": "Error",
                "Confidence": 0.0, "Action": "Review", 
                "image_path": str(img_p), "Error Description": f"SAM3 failed to detect two buildings (found {len(bboxes)})"
            })
            continue

        dx, dy = pos_shifts.get(position, pos_shifts.get(str(position), [-50, 0]))

        final_image = directional_crop_and_pad(
            crop, bboxes,
            target_size=tuple(config.get("target_size", (200, 200))),
            dx=dx,
            dy=dy
        )

        input_tensor = dino_transform(Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        patch_mask = get_horizontal_patch_mask_from_array(final_image).unsqueeze(0).to(device)
        lap = get_laplacian_tensor_from_array(final_image).to(device)

        with torch.no_grad():
            logits = model(input_tensor, patch_mask, lap)
            probs = torch.softmax(logits, dim=1)[0]

        ok_idx = config.get("ok_class_index", 1)
        ok_prob = probs[ok_idx].item()
        
        if ok_prob >= config.get("confidence_gate", 0.9):
            final_pred = "o"
            conf = ok_prob
        else:
            defect_idxs = [i for i in range(len(probs)) if i != ok_idx]
            defect_probs = probs[defect_idxs]
            max_idx = defect_idxs[int(torch.argmax(defect_probs).item())]
            final_pred = ID_TO_LABEL.get(max_idx, str(max_idx))
            conf = probs[max_idx].item()

            sn_thresh_raw = (
                config.get("sn_count_threshold_percent") or 
                config.get("sn_single_threshold_percent", 45)
            )

            sn_thresh = sn_thresh_raw / 100.0 if sn_thresh_raw >= 1.0 else sn_thresh_raw

            if final_pred == 'sn' and conf < sn_thresh:
                final_pred = 'o'
                conf = ok_prob

        raw_predictions.append({
            "SN": sn, "Position": position, "Prediction": final_pred,
            "Confidence": round(conf * 100, 2), "Action": "Evaluate", "image_path": str(img_p),
            "Error Description": ""
        })

        unit_img_dir = processed_images_dir / str(sn)
        unit_img_dir.mkdir(parents=True, exist_ok=True)
        out_path = unit_img_dir / img_p.name
        Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)).save(out_path)

        del input_tensor, patch_mask, lap, logits, probs
        if idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        progress_bar.progress((idx + 1) / total_images)
        
    status_text.text("Image processing complete. Generating reports...")

    df_raw = pd.DataFrame(raw_predictions)
    if df_raw.empty:
        raise RuntimeError("No predictions generated. Check input folder and model outputs.")

    target_positions = config.get("target_positions", [1, 3, 5, 7, 9])
    consolidated = []
    
    for sn, group in df_raw.groupby("SN"):
        row = {"SN": sn}
        errors = []
        has_missing_position = False
        
        for _, rec in group.iterrows():
            if rec.get("Error Description"):
                errors.append(f"Pos {rec['Position']}: {rec['Error Description']}")

        for pos in target_positions:
            rec = group[group["Position"] == pos]
            if not rec.empty:
                row[f"pos {pos} predict"] = rec.iloc[0]["Prediction"]
                row[f"pos {pos} confidence"] = f"{rec.iloc[0]['Confidence']}%"
            else:
                row[f"pos {pos} predict"] = "Missing"
                row[f"pos {pos} confidence"] = "N/A"
                has_missing_position = True
                errors.append(f"Position {pos} is missing from the dataset")
                
        row["Error Description"] = "; ".join(errors) if errors else "None"

        if row["Error Description"] != "None" or has_missing_position:
            row["Status"] = "FLC Required (Error/Warning)"
        else:
            row["Status"] = "Pass (Program Pass)"
            
        consolidated.append(row)

    df_final = pd.DataFrame(consolidated)

    paired_cols = [
        (f'pos {pos} predict', f'pos {pos} confidence')
        for pos in target_positions
        if f'pos {pos} predict' in df_final.columns and f'pos {pos} confidence' in df_final.columns
    ]
    
    if paired_cols:
        pred_frame = pd.DataFrame({p_col: df_final[p_col].astype(str).str.strip().str.lower() for p_col, _ in paired_cols})

        if config.get("filter_on_n", True):
            has_n_mask = pred_frame.eq('n').any(axis=1)
        else:
            has_n_mask = pd.Series(False, index=df_final.index)

        if config.get("filter_on_sn_single", True):
            any_sn_single_mask = pred_frame.eq('sn').any(axis=1)
        else:
            any_sn_single_mask = pd.Series(False, index=df_final.index)

        sn_count_req = config.get("sn_count_required")
        if config.get("filter_on_sn_count", True) and sn_count_req is not None:
            sn_mask_count = pred_frame.eq('sn').sum(axis=1) >= sn_count_req
        else:
            sn_mask_count = pd.Series(False, index=df_final.index)

        trigger_review_mask = has_n_mask | any_sn_single_mask | sn_mask_count
        df_final.loc[trigger_review_mask & (df_final["Status"] == "Pass (Program Pass)"), "Status"] = "Review (Flagged for Double Check)"

        # ----------------------------------------------------
        # 1. 核心保持：先用原始状态生成并保存 metadata.json (不修改 JSON)
        # ----------------------------------------------------
        for _, row in df_final.iterrows():
            sn = row["SN"]
            dst_dir = processed_images_dir / str(sn)
            dst_dir.mkdir(parents=True, exist_ok=True)

            unit_json_data = {
                "SN": sn,
                "Status": row["Status"],  # 保持原厂英文状态不变写入 JSON
                "Error Description": row.get('Error Description', 'None'),
                "positions": {}
            }

            for pos in target_positions:
                pred_val = row.get(f"pos {pos} predict", "o")
                conf_val = row.get(f"pos {pos} confidence", "0.0%")

                unit_json_data["positions"][f"pos {pos}"] = {
                    "model_predict": pred_val,
                    "model_confidence": conf_val,
                    "human_annotation": pred_val
                }

            json_file_path = dst_dir / "metadata.json"
            with open(json_file_path, "w") as jf:
                json.dump(unit_json_data, jf, indent=4)

        # ----------------------------------------------------
        # 2. 状态映射变更：对 Dataframe 执行中文翻译后再导出 CSV / Excel
        # ----------------------------------------------------
        df_final["Status"] = df_final["Status"].map(lambda x: STATUS_MAPPING_ZH.get(x, x))

        # 编译经过过滤的分流报表（此时 Status 列已经是中文）
        review_mask = df_final["Status"].isin(["异常复核 (FLC)", "待人工审查 (Review)"])
        df_filtered = df_final[review_mask].copy()

        filtered_csv = dataset_output_dir / config.get("filtered_csv", "predictions_filtered.csv")
        filtered_xlsx = dataset_output_dir / config.get("filtered_excel", "predictions_filtered.xlsx")
        df_filtered.to_csv(filtered_csv, index=False)
        df_filtered.to_excel(filtered_xlsx, index=False)

    full_csv = dataset_output_dir / config.get("output_csv", "consolidated_batch_predictions.csv")
    full_xlsx = dataset_output_dir / config.get("output_excel", "consolidated_batch_predictions.xlsx")
    df_final.to_csv(full_csv, index=False)
    df_final.to_excel(full_xlsx, index=False)

    status_text.text("Reports successfully generated!")
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / total_images if total_images > 0 else 0

    flc_err_count = int((df_final["Status"] == "异常复核 (FLC)").sum()) if paired_cols else 0
    rev_check_count = int((df_final["Status"] == "待人工审查 (Review)").sum()) if paired_cols else 0

    return {
        "full_csv": str(full_csv),
        "full_xlsx": str(full_xlsx),
        "filtered_csv": str(filtered_csv) if paired_cols else "Skipped (No valid pairs)",
        "filtered_xlsx": str(filtered_xlsx) if paired_cols else "Skipped",
        "processed_images_folder": str(processed_images_dir),
        "total_units": len(df_final),
        "units_flagged_for_review": len(df_filtered) if paired_cols else 0,
        "flc_error_warning_count": flc_err_count,
        "review_double_check_count": rev_check_count,
        "process_time_seconds": round(elapsed_time, 2),
        "avg_time_per_image_seconds": round(avg_time, 2)
    }