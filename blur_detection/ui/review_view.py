import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import json

LANG_REVIEW = {
    "EN": {
        "header": "Image Review & Annotation",
        "info": "Load your consolidated batch predictions to review cropped images and add human annotations.",
        "p_csv": "Path to Predictions CSV",
        "p_folder": "Path to Processed Images Folder",
        "valid_csv": "Please provide a valid CSV path to begin.",
        "refresh": "Refresh Data",
        "filter_lbl": "🔍 Filter Checklist / Review Units by Status",
        "filter_help": "Select specific statuses to customize the active review queue layout dynamically.",
        "no_match": "No units match the selected status filter. Please adjust your checklist criteria.",
        "record": "Record",
        "sn": "Serial Number (SN):",
        "status": "Current Status:",
        "flc_y": "⚠️ Need FLC",
        "flc_n": "✅ Don't Need FLC",
        "prev": "⬅️ Previous",
        "next": "Next ➡️",
        "grid_title": "🖼️ Interactive Operational Grid",
        "missing_img": "Missing Image",
        "sys_log": "ℹ️ System logs: Live changes automatically compiled to include all units at:"
    },
    "ZH": {
        "header": "图像数据审查与标注面板",
        "info": "加载整合的批处理预测 CSV 数据集，以审查裁剪图块并直接附加人工校准结论。",
        "p_csv": "预测结果 CSV 路径",
        "p_folder": "已过滤图像文件夹路径",
        "valid_csv": "请提供有效的 CSV 路径以启动面板。",
        "refresh": "刷新数据",
        "filter_lbl": "🔍 依据状态过滤审查队列明细",
        "filter_help": "选中特定的业务状态，动态定制当前活动审查队列的布局。",
        "no_match": "没有单元匹配选中的状态过滤条件。请重新调整选择指标。",
        "record": "当前记录条数",
        "sn": "产品序列号 (SN):",
        "status": "当前业务状态:",
        "flc_y": "⚠️ 需要 FLC 异常复核",
        "flc_n": "✅ 正常无需 FLC",
        "prev": "⬅️ 上一页",
        "next": "下一页 ➡️",
        "grid_title": "🖼️ 交互式多视角全景操作网格",
        "missing_img": "图像文件丢失",
        "sys_log": "ℹ️ 系统日志：实时变更结果已自动汇编保存至："
    }
}

def render_review_tab():
    lang = st.session_state.get("lang", "EN")
    ln = LANG_REVIEW[lang]

    st.header(ln["header"])
    st.info(ln["info"])

    # --- ADVANCED CSS FOR STRETCHED HEIGHTS AND DETACHED VIEWPORT CENTERING ---
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: stretch !important;
            overflow: visible !important;
        }
        div[data-testid="column"] {
            display: flex !important;
            flex-direction: column !important;
            overflow: visible !important;
        }
        div[data-testid="stImage"] { 
            overflow: visible !important; 
            position: relative; 
            display: flex !important; 
            justify-content: flex-start !important; 
            align-items: center; 
            width: 100% !important;
            margin-left: -5px !important;
        }
        div[data-testid="stImage"] > div {
            display: flex !important; 
            justify-content: flex-start !important;
            width: 100% !important;
            overflow: visible !important;
        }
        button[title="View fullscreen"], div[data-testid="stImage"] button { visibility: hidden !important; display: none !important; pointer-events: none !important; }
        div[data-testid="stImage"] img { 
            border-radius: 4px; 
            image-rendering: crisp-edges; 
            image-rendering: pixelated; 
            cursor: zoom-in; 
            pointer-events: auto !important; 
            height: 22vh !important; 
            width: 100% !important; 
            object-fit: contain !important; 
            object-position: left center !important;
            z-index: 1; 
        }
        div[data-testid="stImage"] img:hover { 
            position: fixed !important; 
            top: 50vh !important;       
            left: 50vw !important;
            transform: translate(-50%, -50%) !important; 
            width: 80vw !important;      
            height: 80vh !important;
            max-width: 80vw !important;
            max-height: 80vh !important;
            object-fit: contain !important; 
            margin: 0px !important;
            z-index: 999999 !important; 
            box-shadow: 0px 0px 150px rgba(0, 0, 0, 0.95) !important; 
            background-color: #1e1e1e !important; 
            padding: 16px !important; 
            border: 1px solid #444 !important; 
        }
        .vertical-button-box {
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
        }
        .vertical-button-box div[data-testid="stVerticalBlock"] { 
            gap: 2px !important; 
            display: flex; 
            flex-direction: column; 
            justify-content: space-between; 
            height: 100% !important; 
            min-height: 22vh !important;
        }
        .vertical-button-box button { flex-grow: 1 !important; padding: 0px !important; height: 100% !important; font-size: 0.85rem !important; line-height: 1.2 !important; }
        div[data-testid="stVerticalBlock"] > div { padding-bottom: 2px !important; padding-top: 2px !important; }
        </style>
    """, unsafe_allow_html=True)

    # --- DYNAMIC TARGET RESOLUTION ROUTINES ---
    # Fetch unified default positions computed directly from the app config session modifications
    output_root_val = st.session_state.app_config.get("default_output_root", "")
    
    fallback_csv = str(Path(output_root_val) / "dataset_output" / "predictions_filtered.csv") if output_root_val else ""
    fallback_folder = str(Path(output_root_val) / "processed_images") if output_root_val else ""

    # Map inputs to use tracking session variables as top priorities, cascading to fallbacks if empty
    default_review_csv = st.session_state.get("review_csv_path", fallback_csv)
    default_review_img = st.session_state.get("review_img_folder", fallback_folder)

    # --- TOP INPUTS AND REFRESH ROW ---
    col_csv, col_img, col_ref = st.columns([4.5, 4.5, 1])
    with col_csv:
        csv_path = st.text_input(ln["p_csv"], value=default_review_csv)
    with col_img:
        img_folder = st.text_input(ln["p_folder"], value=default_review_img)
    with col_ref:
        st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        if st.button(ln["refresh"], width='stretch'):
            if 'pred_data_cache' in st.session_state:
                del st.session_state.pred_data_cache
            if 'live_statuses' in st.session_state:
                del st.session_state.live_statuses
            st.rerun()

    # Retain mutations in the active context session
    st.session_state["review_csv_path"] = csv_path
    st.session_state["review_img_folder"] = img_folder

    out_name = "human_annotated_predictions.csv"

    if not csv_path or not Path(csv_path).exists():
        st.warning(ln["valid_csv"])
        return

    if 'pred_data_cache' not in st.session_state:
        try:
            st.session_state.pred_data_cache = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return

    df_master = st.session_state.pred_data_cache
    base_dir = Path(csv_path).parent
    master_csv_path = base_dir / "consolidated_batch_predictions.csv"
    
    if master_csv_path.exists() and len(df_master) < 5: 
        try:
            df_master = pd.read_csv(master_csv_path)
            st.session_state.pred_data_cache = df_master
        except:
            pass

    if 'annotations' not in st.session_state: st.session_state.annotations = {}
    if 'live_statuses' not in st.session_state: st.session_state.live_statuses = {}
    if 'review_idx' not in st.session_state: st.session_state.review_idx = 0

    for _, row in df_master.iterrows():
        sn = str(row['SN'])
        if sn not in st.session_state.live_statuses:
            json_p = Path(img_folder) / sn / "metadata.json"
            if json_p.exists():
                try:
                    with open(json_p, "r") as jf:
                        d = json.load(jf)
                    st.session_state.live_statuses[sn] = d.get("Status", row.get("Status", "Review (Flagged for Double Check)"))
                except:
                    st.session_state.live_statuses[sn] = row.get("Status", "Review (Flagged for Double Check)")
            else:
                st.session_state.live_statuses[sn] = row.get("Status", "Review (Flagged for Double Check)")

    all_statuses = sorted(list(set(st.session_state.live_statuses.values()).union({
        "Pass (Program Pass)", "Pass (Human Reviewed)", "Review (Flagged for Double Check)", 
        "FLC Required (Human Reviewed)", "FLC Required (Error/Warning)"
    })))
    
    selected_statuses = st.multiselect(ln["filter_lbl"], options=all_statuses, default=all_statuses, help=ln["filter_help"])

    filtered_rows = [row for _, row in df_master.iterrows() if st.session_state.live_statuses[str(row['SN'])] in selected_statuses]
    df_filtered_view = pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame(columns=df_master.columns)

    if df_filtered_view.empty:
        st.warning(ln["no_match"])
        return

    if st.session_state.review_idx >= len(df_filtered_view):
        st.session_state.review_idx = 0

    current_row = df_filtered_view.iloc[st.session_state.review_idx]
    current_sn = str(current_row['SN'])
    unit_folder = Path(img_folder) / current_sn
    json_path = unit_folder / "metadata.json"

    error_desc = str(current_row.get('Error Description', 'None')).strip()
    has_pipeline_error = error_desc != "" and error_desc.lower() != "none"
    positions = [1, 3, 5, 7, 9]

    # --- Criteria-Based Logic Evaluation Engine ---
    def evaluate_flc_rules():
        """Evaluates active position arrays and determines logic status metrics."""
        labels = [st.session_state.annotations[current_sn][f'pos {p}'] for p in positions]
        if labels.count('n') >= 1 or labels.count('sn') >= 3:
            return "Yes"
        return "No"

    # --- SYNCHRONIZED INITIALIZATION ---
    if current_sn not in st.session_state.annotations:
        p_data = {'need_flc': 'No', 'flc_manually_overridden': False, 'touched': False}
        
        if json_path.exists():
            try:
                with open(json_path, "r") as jf:
                    j_data = json.load(jf)
                for p in positions:
                    p_data[f'pos {p}'] = j_data["positions"].get(f"pos {p}", {}).get("human_annotation", "o")
                saved_status = j_data.get("Status", "")
                if saved_status in ["FLC Required (Human Reviewed)", "Pass (Human Reviewed)"]:
                    p_data['touched'] = True
            except Exception:
                json_path.unlink(missing_ok=True)

        if f'pos 1' not in p_data:
            valid_classes = {'o', 'f', 'sn', 'n'}
            for p in positions:
                pred_val = str(current_row.get(f'pos {p} predict', 'o')).lower().strip()
                p_data[f'pos {p}'] = pred_val if pred_val in valid_classes else 'o'
            
            initial_status = st.session_state.live_statuses.get(current_sn, "")
            if initial_status in ["FLC Required (Human Reviewed)", "Pass (Human Reviewed)"]:
                p_data['touched'] = True

        st.session_state.annotations[current_sn] = p_data
        
        if not p_data['touched']:
            if has_pipeline_error or "FLC Required" in st.session_state.live_statuses.get(current_sn, ""):
                st.session_state.annotations[current_sn]['need_flc'] = "Yes"
            else:
                st.session_state.annotations[current_sn]['need_flc'] = "No"
        else:
            st.session_state.annotations[current_sn]['need_flc'] = evaluate_flc_rules()

    # --- LOGIC RUNTIME ENGINE SYNCHRONIZATION ---
    current_flc = st.session_state.annotations[current_sn]['need_flc']
    if not st.session_state.annotations[current_sn]['touched']:
        live_action_status = "FLC Required (Error/Warning)" if has_pipeline_error else st.session_state.live_statuses.get(current_sn, "Review (Flagged for Double Check)")
    else:
        live_action_status = "FLC Required (Human Reviewed)" if current_flc == "Yes" else "Pass (Human Reviewed)"

    def auto_save_and_compile_master():
        st.session_state.live_statuses[current_sn] = live_action_status
        unit_folder.mkdir(parents=True, exist_ok=True)
        updated_json_payload = {"SN": current_sn, "Status": live_action_status, "Error Description": error_desc, "positions": {}}
        for p in positions:
            updated_json_payload["positions"][f"pos {p}"] = {
                "model_predict": str(current_row.get(f'pos {p} predict', 'o')).lower(),
                "model_confidence": str(current_row.get(f'pos {p} confidence', 'N/A')),
                "human_annotation": st.session_state.annotations[current_sn][f'pos {p}']
            }
        with open(json_path, "w") as jf: json.dump(updated_json_payload, jf, indent=4)

        annotated_rows = []
        for index, row in df_master.iterrows():
            sn_loop = str(row['SN'])
            loop_json_path = Path(img_folder) / sn_loop / "metadata.json"
            loop_error = str(row.get('Error Description', 'None')).strip()
            loop_has_error = loop_error != "" and loop_error.lower() != "none"
            
            if loop_json_path.exists():
                try:
                    with open(loop_json_path, "r") as jf: d = json.load(jf)
                    anno = {f'pos {p}': d["positions"][f"pos {p}"]["human_annotation"] for p in positions}
                    action_status = d.get("Status", "Review (Flagged for Double Check)")
                except Exception:
                    cached = st.session_state.annotations.get(sn_loop, st.session_state.annotations[current_sn])
                    anno = {f'pos {p}': cached.get(f'pos {p}', 'o') for p in positions}
                    action_status = "FLC Required (Error/Warning)" if loop_has_error else "Review (Flagged for Double Check)"
            else:
                if sn_loop in st.session_state.annotations:
                    cached = st.session_state.annotations[sn_loop]
                    anno = {f'pos {p}': cached.get(f'pos {p}', 'o') for p in positions}
                    if not cached['touched']: 
                        action_status = "FLC Required (Error/Warning)" if loop_has_error else st.session_state.live_statuses.get(sn_loop, row.get("Status", "Review (Flagged for Double Check)"))
                    else: 
                        action_status = "FLC Required (Human Reviewed)" if cached['need_flc'] == "Yes" else "Pass (Human Reviewed)"
                else:
                    anno = {'pos 1': 'o', 'pos 3': 'o', 'pos 5': 'o', 'pos 7': 'o', 'pos 9': 'o'}
                    action_status = "FLC Required (Error/Warning)" if loop_has_error else st.session_state.live_statuses.get(sn_loop, row.get("Status", "Pass (Program Pass)"))

            correct_o = sum(1 for p in positions if str(row.get(f'pos {p} predict', '')).lower() == 'o' and anno[f'pos {p}'] == 'o')
            incorrect_o = sum(1 for p in positions if str(row.get(f'pos {p} predict', '')).lower() == 'o' and anno[f'pos {p}'] != 'o')

            annotated_rows.append({
                "SN": sn_loop, "pos 1": anno['pos 1'], "pos 3": anno['pos 3'], "pos 5": anno['pos 5'], "pos 7": anno['pos 7'], "pos 9": anno['pos 9'],
                "pos 1 predict": row.get('pos 1 predict', ''), "pos 1 confidence": row.get('pos 1 confidence', ''),
                "pos 3 predict": row.get('pos 3 predict', ''), "pos 3 confidence": row.get('pos 3 confidence', ''),
                "pos 5 predict": row.get('pos 5 predict', ''), "pos 5 confidence": row.get('pos 5 confidence', ''),
                "pos 7 predict": row.get('pos 7 predict', ''), "pos 7 confidence": row.get('pos 7 confidence', ''),
                "pos 9 predict": row.get('pos 9 predict', ''), "pos 9 confidence": row.get('pos 9 confidence', ''),
                "Status": action_status, "number of correct \"o\"": correct_o, "number of incorrect \"o\"": incorrect_o
            })
        if annotated_rows: pd.DataFrame(annotated_rows).to_csv(Path(img_folder) / out_name, index=False)

    def render_control_deck(location_key):
        st.write(f"### {ln['record']}: {st.session_state.review_idx + 1} / {len(df_filtered_view)}")
        st.write(f"**{ln['sn']}** `{current_sn}` | **{ln['status']}** `{live_action_status}`")

        f_col1, f_col2, _ = st.columns([1.5, 1.5, 7])
        with f_col1:
            if st.button(ln["flc_y"], key=f"flc_y_{location_key}_{current_sn}", type="primary" if current_flc == "Yes" else "secondary", width='stretch'):
                st.session_state.annotations[current_sn]['need_flc'] = "Yes"
                st.session_state.annotations[current_sn]['touched'] = True
                st.session_state.annotations[current_sn]['flc_manually_overridden'] = True
                auto_save_and_compile_master()
                st.rerun()
        with f_col2:
            if st.button(ln["flc_n"], key=f"flc_n_{location_key}_{current_sn}", type="primary" if current_flc == "No" else "secondary", width='stretch'):
                st.session_state.annotations[current_sn]['need_flc'] = "No"
                st.session_state.annotations[current_sn]['touched'] = True
                st.session_state.annotations[current_sn]['flc_manually_overridden'] = True
                auto_save_and_compile_master()
                st.rerun()

        nav_col1, nav_col2, _ = st.columns([1.5, 1.5, 7])
        with nav_col1:
            if st.button(ln["prev"], key=f"nav_prev_{location_key}", width='stretch') and st.session_state.review_idx > 0:
                st.session_state.annotations[current_sn]['touched'] = True
                auto_save_and_compile_master()
                st.session_state.review_idx -= 1
                st.rerun()
        with nav_col2:
            if st.button(ln["next"], key=f"nav_next_{location_key}", width='stretch') and st.session_state.review_idx < len(df_filtered_view) - 1:
                st.session_state.annotations[current_sn]['touched'] = True
                auto_save_and_compile_master()
                st.session_state.review_idx += 1
                st.rerun()
                
    render_control_deck(location_key="top")
    st.divider()

    classes = ['o', 'f', 'sn', 'n']
    grid_layout = [[1, None, 3], [None, 5, None], [7, None, 9]]
    st.subheader(ln["grid_title"])
    
    for row in grid_layout:
        cols = st.columns(3)
        for col_idx, pos in enumerate(row):
            with cols[col_idx]:
                if pos is not None:
                    # Perform image file glob checks in the standardized folder setup structure
                    img_files = (list(unit_folder.glob(f"{current_sn}-{pos}.*")) + 
                                 list(unit_folder.glob(f"processed_{current_sn}-{pos}.*")) +
                                 list(unit_folder.glob(f"*{current_sn}-{pos}.*")))

                    current_selection = st.session_state.annotations[current_sn][f'pos {pos}']
                    sub_c1, sub_c2 = st.columns([1.2, 8.8])
                    
                    with sub_c1:
                        st.markdown('<div style="height: 38px;"></div>', unsafe_allow_html=True)
                        st.markdown('<div class="vertical-button-box">', unsafe_allow_html=True)
                        for cls_label in classes:
                            if st.button(cls_label.upper(), key=f"lbl_{current_sn}_{pos}_{cls_label}", type="primary" if (current_selection == cls_label) else "secondary", width='stretch'):
                                st.session_state.annotations[current_sn][f'pos {pos}'] = cls_label
                                st.session_state.annotations[current_sn]['touched'] = True
                                
                                st.session_state.annotations[current_sn]['flc_manually_overridden'] = False
                                st.session_state.annotations[current_sn]['need_flc'] = evaluate_flc_rules()
                                
                                auto_save_and_compile_master(); st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with sub_c2:
                        pred_val = str(current_row.get(f'pos {pos} predict', 'N/A')).lower()
                        conf_val = str(current_row.get(f'pos {pos} confidence', 'N/A'))
                        st.markdown(f"""
                            <div style="margin-bottom: 2px; line-height: 1.3; text-align: left;">
                                <span style="font-weight: 600; font-size: 1rem; color: inherit;">Pos {pos}</span><br>
                                <span style="font-size: 0.85rem; color: #888888;">
                                    AI: <code style="background-color: rgba(128,128,128,0.1); padding: 2px 4px; border-radius: 4px;">{pred_val}</code> ({conf_val})
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
                        if img_files: 
                            st.image(Image.open(img_files[0]), width='stretch')
                        else: 
                            st.error(ln["missing_img"])
                else:
                    st.write("")

    st.divider()
    render_control_deck(location_key="bottom")
    st.divider()
    st.caption(f"{ln['sys_log']} `{Path(img_folder) / out_name}`")