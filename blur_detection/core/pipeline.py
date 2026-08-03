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
from concurrent.futures import ThreadPoolExecutor

# --- 中文状态持久化映射字典 ---
STATUS_MAPPING_ZH = {
    "Pass (Program Pass)": "自动通过 (Pass)",
    "Pass (Human Reviewed)": "人工确认通过 (Pass)",
    "Review (Flagged for Double Check)": "待人工审查 (Review)",
    "FLC Required (Human Reviewed)": "人工确认复核 (FLC)",
    "FLC Required (Error/Warning)": "异常复核 (FLC)",
    "NG (Human Reviewed)": "人工确认报废 (NG)"
}

def async_save_image(final_image, out_path):
    Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)).save(out_path)

def segment_single_image(img_p, segmenter, config, pos_shifts, dino_transform):
    try:
        sn, position = extract_sn_and_pos(img_p.name)
    except Exception as e:
        return {
            "success": False,
            "sn": "Unknown", "position": "Unknown", "img_p": img_p,
            "error_desc": f"Filename identification failed: {e}"
        }

    image = segmenter.load_image(str(img_p))
    if image is None:
        return {
            "success": False,
            "sn": sn, "position": position, "img_p": img_p,
            "error_desc": "Image could not be loaded or identified"
        }

    cell = get_grid_cell_from_name(img_p.name)
    if cell is None:
        return {
            "success": False,
            "sn": sn, "position": position, "img_p": img_p,
            "error_desc": "Grid crop region selection failed"
        }
    
    crop = segmenter.select_crop_region(image, cell, config=config)
    if crop is None:
        return {
            "success": False,
            "sn": sn, "position": position, "img_p": img_p,
            "error_desc": "Grid crop region selection failed"
        }

    result = segmenter.add_text_prompt(crop, "building")
    if not result or len(result.get("masks", [])) == 0:
        return {
            "success": False,
            "sn": sn, "position": position, "img_p": img_p,
            "error_desc": "SAM3 segmentation failed to find building masks"
        }

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

    # Free the SAM3 masks immediately (VRAM Optimization)
    del result
    result = None

    if len(bboxes) < 2:
        return {
            "success": False,
            "sn": sn, "position": position, "img_p": img_p,
            "error_desc": f"SAM3 failed to detect two buildings (found {len(bboxes)})"
        }

    dx, dy = pos_shifts.get(position, pos_shifts.get(str(position), [-50, 0]))

    final_image = directional_crop_and_pad(
        crop, bboxes,
        target_size=tuple(config.get("target_size", (200, 200))),
        dx=dx,
        dy=dy
    )

    # Preprocess DINOv3 inputs with Pinned Memory (Item 3: Pinned Memory)
    input_tensor = dino_transform(Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).pin_memory()
    patch_mask = get_horizontal_patch_mask_from_array(final_image).unsqueeze(0).pin_memory()
    lap = get_laplacian_tensor_from_array(final_image).pin_memory()

    return {
        "success": True,
        "sn": sn, "position": position, "img_p": img_p,
        "final_image": final_image,
        "input_tensor": input_tensor,
        "patch_mask": patch_mask,
        "lap": lap
    }

def process_active_batch(batch_items, model, device, config, id_to_label, raw_predictions, processed_images_dir, executor):
    # Stack tensors along batch dimension (Item 2: Batching)
    batched_input = torch.cat([item["input_tensor"] for item in batch_items], dim=0).to(device, non_blocking=True)
    batched_patch_mask = torch.cat([item["patch_mask"] for item in batch_items], dim=0).to(device, non_blocking=True)
    batched_lap = torch.cat([item["lap"] for item in batch_items], dim=0).to(device, non_blocking=True)

    # Run DINOv3 batched inference using mixed precision (Item 2: Batching)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batched_input, batched_patch_mask, batched_lap)
        probs_all = torch.softmax(logits, dim=1)

    # Post-process each item in the batch
    for i, item in enumerate(batch_items):
        sn = item["sn"]
        position = item["position"]
        img_p = item["img_p"]
        final_image = item["final_image"]
        probs = probs_all[i]

        ok_idx = config.get("ok_class_index", 1)
        ok_prob = probs[ok_idx].item()
        
        if ok_prob >= config.get("confidence_gate", 0.9):
            final_pred = "o"
            conf = ok_prob
        else:
            defect_idxs = [idx for idx in range(len(probs)) if idx != ok_idx]
            defect_probs = probs[defect_idxs]
            max_idx = defect_idxs[int(torch.argmax(defect_probs).item())]
            final_pred = id_to_label.get(max_idx, str(max_idx))
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
            "idx": item["idx"],
            "SN": sn, "Position": position, "Prediction": final_pred,
            "Confidence": round(conf * 100, 2),
            "f confidence": round(probs[0].item() * 100, 2),
            "o confidence": round(probs[1].item() * 100, 2),
            "sn confidence": round(probs[2].item() * 100, 2),
            "n confidence": round(probs[3].item() * 100, 2),
            "Action": "Evaluate", "image_path": str(img_p),
            "Error Description": ""
        })

        # Asynchronously save image in a background thread (Item 1: CPU-GPU overlap)
        unit_img_dir = processed_images_dir / str(sn)
        unit_img_dir.mkdir(parents=True, exist_ok=True)
        out_path = unit_img_dir / img_p.name
        
        executor.submit(async_save_image, final_image, out_path)

    # Clean up GPU references immediately
    del batched_input, batched_patch_mask, batched_lap, logits, probs_all

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

    # ThreadPoolExecutor for background file writes (Item 1: CPU-GPU overlap)
    file_executor = ThreadPoolExecutor(max_workers=4)
    
    batch_items = []
    max_batch_size = 8

    # Start SAM3 segmentations in parallel (Item 1 & 2: Task Parallelism & Prefetching)
    sam3_threads = config.get("sam3_threads", 4)
    with ThreadPoolExecutor(max_workers=sam3_threads) as sam_executor:
        # Submit all images to run in parallel threads
        futures = [
            sam_executor.submit(segment_single_image, img_p, segmenter, config, pos_shifts, dino_transform)
            for img_p in image_paths
        ]

        # Gather results in the exact original order of image_paths (Ordered Prefetching)
        for idx, img_p in enumerate(image_paths):
            status_text.text(f"Processing {idx + 1}/{total_images}: {img_p.name}")
            
            # Wait for this specific image's SAM3 segmentation to complete
            res = futures[idx].result()
            
            sn = res["sn"]
            position = res["position"]

            if not res["success"]:
                # Record error predicting immediately
                raw_predictions.append({
                    "idx": idx,
                    "SN": sn, "Position": position, "Prediction": "Error",
                    "Confidence": 0.0,
                    "f confidence": 0.0, "o confidence": 0.0, "sn confidence": 0.0, "n confidence": 0.0,
                    "Action": "Review", 
                    "image_path": str(img_p), "Error Description": res["error_desc"]
                })
                progress_bar.progress((idx + 1) / total_images)
                continue

            # Flush current batch if SN changes
            if batch_items and batch_items[0]["sn"] != sn:
                process_active_batch(batch_items, model, device, config, ID_TO_LABEL, raw_predictions, processed_images_dir, file_executor)
                batch_items = []

            # Add to the active DINOv3 batch queue (Item 2: Batching)
            batch_items.append({
                "idx": idx,
                "sn": sn,
                "position": position,
                "img_p": img_p,
                "final_image": res["final_image"],
                "input_tensor": res["input_tensor"],
                "patch_mask": res["patch_mask"],
                "lap": res["lap"]
            })

            if len(batch_items) >= max_batch_size:
                process_active_batch(batch_items, model, device, config, ID_TO_LABEL, raw_predictions, processed_images_dir, file_executor)
                batch_items = []

            if (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            progress_bar.progress((idx + 1) / total_images)

    # Process remaining batch items
    if batch_items:
        process_active_batch(batch_items, model, device, config, ID_TO_LABEL, raw_predictions, processed_images_dir, file_executor)
        batch_items = []

    # Shutdown background worker pool and wait for all image writes to complete
    file_executor.shutdown(wait=True)
    
    # Final VRAM cache flush
    torch.cuda.empty_cache()
    gc.collect()
        
    status_text.text("Image processing complete. Generating reports...")

    raw_predictions.sort(key=lambda x: x.pop("idx", 0))
    df_raw = pd.DataFrame(raw_predictions)
    if df_raw.empty:
        raise RuntimeError("No predictions generated. Check model outputs.")

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
                row[f"pos {pos} f confidence"] = f"{rec.iloc[0]['f confidence']}%"
                row[f"pos {pos} o confidence"] = f"{rec.iloc[0]['o confidence']}%"
                row[f"pos {pos} sn confidence"] = f"{rec.iloc[0]['sn confidence']}%"
                row[f"pos {pos} n confidence"] = f"{rec.iloc[0]['n confidence']}%"
            else:
                row[f"pos {pos} predict"] = "Missing"
                row[f"pos {pos} confidence"] = "N/A"
                row[f"pos {pos} f confidence"] = "N/A"
                row[f"pos {pos} o confidence"] = "N/A"
                row[f"pos {pos} sn confidence"] = "N/A"
                row[f"pos {pos} n confidence"] = "N/A"
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
                f_conf = row.get(f"pos {pos} f confidence", "0.0%")
                o_conf = row.get(f"pos {pos} o confidence", "0.0%")
                sn_conf = row.get(f"pos {pos} sn confidence", "0.0%")
                n_conf = row.get(f"pos {pos} n confidence", "0.0%")

                unit_json_data["positions"][f"pos {pos}"] = {
                    "model_predict": pred_val,
                    "model_confidence": conf_val,
                    "model_f_confidence": f_conf,
                    "model_o_confidence": o_conf,
                    "model_sn_confidence": sn_conf,
                    "model_n_confidence": n_conf,
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

        filtered_json = dataset_output_dir / "predictions_filtered.json"
        filtered_xlsx = dataset_output_dir / "predictions_filtered.xlsx"
        df_filtered.to_json(filtered_json, orient="records", indent=4)
        df_filtered.to_excel(filtered_xlsx, index=False)

    full_json = dataset_output_dir / "consolidated_batch_predictions.json"
    full_xlsx = dataset_output_dir / "consolidated_batch_predictions.xlsx"
    df_final.to_json(full_json, orient="records", indent=4)
    df_final.to_excel(full_xlsx, index=False)

    raw_json = dataset_output_dir / "raw_predictions.json"
    raw_xlsx = dataset_output_dir / "raw_predictions.xlsx"
    df_raw.to_json(raw_json, orient="records", indent=4)
    df_raw.to_excel(raw_xlsx, index=False)

    status_text.text("Reports successfully generated!")
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / total_images if total_images > 0 else 0

    flc_err_count = int((df_final["Status"] == "异常复核 (FLC)").sum()) if paired_cols else 0
    rev_check_count = int((df_final["Status"] == "待人工审查 (Review)").sum()) if paired_cols else 0

    return {
        "full_json": str(full_json),
        "full_xlsx": str(full_xlsx),
        "raw_json": str(raw_json),
        "raw_xlsx": str(raw_xlsx),
        "filtered_json": str(filtered_json) if paired_cols else "Skipped (No valid pairs)",
        "filtered_xlsx": str(filtered_xlsx) if paired_cols else "Skipped",
        "processed_images_folder": str(processed_images_dir),
        "total_units": len(df_final),
        "units_flagged_for_review": len(df_filtered) if paired_cols else 0,
        "flc_error_warning_count": flc_err_count,
        "review_double_check_count": rev_check_count,
        "process_time_seconds": round(elapsed_time, 2),
        "avg_time_per_image_seconds": round(avg_time, 2)
    }