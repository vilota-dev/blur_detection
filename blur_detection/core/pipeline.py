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

from utils.image_utils import (
    directional_crop_and_pad, 
    get_horizontal_patch_mask_from_array, 
    get_laplacian_tensor_from_array
)
from utils.file_utils import get_grid_cell_from_name, extract_sn_and_pos

def process_and_predict(input_folder, output_folder, filtered_images_folder, config, device, segmenter, model):
    start_time = time.time()

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    out_filtered = Path(filtered_images_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_folder}")

    output_path.mkdir(parents=True, exist_ok=True)
    out_filtered.mkdir(parents=True, exist_ok=True)

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
        sn, position = extract_sn_and_pos(img_p.name)

        if not segmenter.load_image(str(img_p)):
            continue
            
        cell = get_grid_cell_from_name(img_p.name)
        if not segmenter.select_crop_region(cell):
            continue

        result = segmenter.add_text_prompt("building")
        
        if not result or len(result.get("masks", [])) == 0:
            raw_predictions.append({
                "SN": sn, "Position": position, "Prediction": "Seg Failed",
                "Confidence": 0.0, "Action": "Review (Segmentation Failed)", "image_path": str(img_p)
            })
            continue

        crop = segmenter.current_crop.copy()
        h_crop, w_crop = crop.shape[:2]
        bboxes = []
        for mask in result["masks"][:2]:
            mask_np = mask[0].cpu().numpy() if hasattr(mask[0], "cpu") else mask[0]
            if mask_np.ndim == 3: mask_np = mask_np[0]
            mask_uint = (mask_np > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask_resized > 0)
            if xs.size == 0: continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
            bboxes.append([x0, y0, x1, y1])

        if not bboxes:
            raw_predictions.append({
                "SN": sn, "Position": position, "Prediction": "No Bbox",
                "Confidence": 0.0, "Action": "Review (No Building Found)", "image_path": str(img_p)
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
            action = "Pass (Confirmed OK)"
        else:
            defect_idxs = [i for i in range(len(probs)) if i != ok_idx]
            defect_probs = probs[defect_idxs]
            max_idx = defect_idxs[int(torch.argmax(defect_probs).item())]
            final_pred = ID_TO_LABEL.get(max_idx, str(max_idx))
            conf = probs[max_idx].item()
            action = "Review (Flagged for Double Check)"

        raw_predictions.append({
            "SN": sn, "Position": position, "Prediction": final_pred,
            "Confidence": round(conf * 100, 2), "Action": action, "image_path": str(img_p)
        })

        out_path = output_path / f"processed_{img_p.name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
        review = False
        for pos in target_positions:
            rec = group[group["Position"] == pos]
            if not rec.empty:
                row[f"pos {pos} predict"] = rec.iloc[0]["Prediction"]
                row[f"pos {pos} confidence"] = f"{rec.iloc[0]['Confidence']}%"
                if "Review" in rec.iloc[0]["Action"]:
                    review = True
            else:
                row[f"pos {pos} predict"] = "Missing"
                row[f"pos {pos} confidence"] = "N/A"
        row["Action Required"] = "Review (Flagged for Double Check)" if review else "Pass (Confirme"
        "d OK)"
        consolidated.append(row)

    df_final = pd.DataFrame(consolidated)

    full_csv = out_filtered / config.get("output_csv", "consolidated_batch_predictions.csv")
    full_xlsx = out_filtered / config.get("output_excel", "consolidated_batch_predictions.xlsx")
    df_final.to_csv(full_csv, index=False)
    df_final.to_excel(full_xlsx, index=False)

    paired_cols = [
        (f'pos {pos} predict', f'pos {pos} confidence')
        for pos in target_positions
        if f'pos {pos} predict' in df_final.columns and f'pos {pos} confidence' in df_final.columns
    ]
    
    if paired_cols:
        pred_frame = pd.DataFrame({pred_col: df_final[pred_col].astype(str).str.strip().str.lower() for pred_col, _ in paired_cols})
        conf_frame = pd.DataFrame({conf_col: pd.to_numeric(df_final[conf_col].astype(str).str.replace('%', '', regex=False).str.strip(), errors='coerce') for _, conf_col in paired_cols})

        has_n_mask = pred_frame.eq('n').any(axis=1)

        sn_single_thresh = config.get("sn_single_threshold_percent", 45)
        any_sn_single_mask = pd.DataFrame({
            pred_col: (pred_frame[pred_col] == 'sn') & (conf_frame[conf_col] >= sn_single_thresh)
            for pred_col, conf_col in paired_cols
        }).any(axis=1)

        sn_count_thresh = config.get("sn_count_threshold_percent")
        sn_count_req = config.get("sn_count_required")
        
        if sn_count_thresh is not None and sn_count_req is not None:
            sn_mask_count = (
                pd.DataFrame({
                    pred_col: (pred_frame[pred_col] == 'sn') & (conf_frame[conf_col] >= sn_count_thresh)
                    for pred_col, conf_col in paired_cols
                }).sum(axis=1) >= sn_count_req
            )
            selected_mask = has_n_mask | any_sn_single_mask | sn_mask_count
        else:
            selected_mask = has_n_mask | any_sn_single_mask

        df_filtered = df_final[selected_mask].copy()

        filtered_csv = out_filtered / config.get("filtered_csv", "predictions_filtered.csv")
        filtered_xlsx = out_filtered / config.get("filtered_excel", "predictions_filtered.xlsx")
        df_filtered.to_csv(filtered_csv, index=False)
        df_filtered.to_excel(filtered_xlsx, index=False)

        for _, row in df_filtered.iterrows():
            sn = row["SN"]
            for pos in target_positions:
                match = df_raw[(df_raw["SN"] == sn) & (df_raw["Position"] == pos)]
                if not match.empty and match.iloc[0]["Prediction"] not in ["Seg Failed", "No Bbox"]:
                    src = Path(match.iloc[0]["image_path"])
                    
                    # --- Target the cropped/processed image from the output folder ---
                    processed_src = output_path / f"processed_{src.name}"
                    
                    dst_dir = out_filtered / str(sn)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    
                    if processed_src.exists():
                        # Saving as 'src.name' preserves the clean "1234-1.png" layout inside the SN folder
                        shutil.copy(processed_src, dst_dir / src.name)

    status_text.text("Reports successfully generated!")

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / total_images if total_images > 0 else 0

    return {
        "full_csv": str(full_csv),
        "full_xlsx": str(full_xlsx),
        "filtered_csv": str(filtered_csv) if paired_cols else "Skipped (No valid pairs)",
        "filtered_xlsx": str(filtered_xlsx) if paired_cols else "Skipped",
        "filtered_images_folder": str(out_filtered),
        "total_units": len(df_final),
        "units_flagged_for_review": len(df_filtered) if paired_cols else 0,
        "process_time_seconds": round(elapsed_time, 2),
        "avg_time_per_image_seconds": round(avg_time, 2)
    }