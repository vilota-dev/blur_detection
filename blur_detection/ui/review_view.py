import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import json

def render_review_tab():
    st.header("Image Review & Annotation")
    st.info("Load your consolidated batch predictions to review cropped images and add human annotations.")

    # Advanced CSS for Side-by-Side Sizing, Nearest-Neighbor, Hover-Zoom, and Hidden Nav Shortcuts
    st.markdown("""
        <style>
        div[data-testid="stImage"] {
            overflow: visible !important;
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        button[title="View fullscreen"], 
        div[data-testid="stImage"] button {
            visibility: hidden !important;
            display: none !important;
            pointer-events: none !important;
        }
        div[data-testid="stImage"] img {
            border-radius: 4px;
            image-rendering: crisp-edges;
            image-rendering: pixelated;
            cursor: zoom-in;
            pointer-events: auto !important;
            height: 22vh !important;
            width: 100% !important;
            object-fit: contain;
            z-index: 1;
        }
        div[data-testid="stImage"] img:hover {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) scale(1) !important;
            width: 70vw !important;
            height: auto !important;
            max-height: 80vh !important;
            object-fit: contain !important;
            z-index: 99999 !important;
            box-shadow: 0px 0px 100px rgba(0, 0, 0, 0.95) !important;
            background-color: #1e1e1e !important;
            padding: 16px !important;
            border: 1px solid #444 !important;
        }
        .vertical-button-box div[data-testid="stVerticalBlock"] {
            gap: 2px !important;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 22vh !important;
        }
        .vertical-button-box button {
            flex-grow: 1 !important;
            padding: 0px !important;
            height: 100% !important;
            font-size: 0.85rem !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            padding-bottom: 2px !important;
            padding-top: 2px !important;
        }
        .hidden-trigger-deck {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col_csv, col_img = st.columns(2)
    with col_csv:
        csv_path = st.text_input("Path to Predictions CSV", value=str(Path.home() / "566-qa-2/filtered_images/600D/predictions_filtered.csv"))
    with col_img:
        img_folder = st.text_input("Path to Filtered Images Folder", value=str(Path.home() / "566-qa-2/filtered_images/600D"))

    out_name = "human_annotated_predictions.csv"

    if not Path(csv_path).exists():
        st.warning("Please provide a valid CSV path to begin.")
        return

    @st.cache_data(show_spinner=False)
    def load_pred_data(path):
        return pd.read_csv(path)

    c_space, c_ref = st.columns([5, 1])
    with c_ref:
        if st.button("Refresh Data", width='stretch'):
            load_pred_data.clear()
            if 'live_statuses' in st.session_state:
                del st.session_state.live_statuses
            st.rerun()
            
    try:
        base_dir = Path(csv_path).parent
        master_csv_path = base_dir / "consolidated_batch_predictions.csv"
        # Source from master database if available to unlock review coverage on Pass/Clean entries
        if master_csv_path.exists():
            df_master = load_pred_data(master_csv_path)
        else:
            df_master = load_pred_data(csv_path)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'live_statuses' not in st.session_state:
        st.session_state.live_statuses = {}
    if 'review_idx' not in st.session_state:
        st.session_state.review_idx = 0

    # Initialize live statuses for all records from JSON metadata fallback fields
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

    # --- STATUS CHECKLIST FILTER CONTROL UNIFIED INTERFACE ---
    all_statuses = sorted(list(set(st.session_state.live_statuses.values())))
    selected_statuses = st.multiselect(
        "🔍 Filter Checklist / Review Units by Status",
        options=all_statuses,
        default=all_statuses,
        help="Select specific statuses to customize the active review queue layout dynamically."
    )

    # Compile the interactive view matching selection states
    filtered_rows = [row for _, row in df_master.iterrows() if st.session_state.live_statuses[str(row['SN'])] in selected_statuses]
    df_filtered_view = pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame(columns=df_master.columns)

    if df_filtered_view.empty:
        st.warning("No units match the selected status filter. Please adjust your checklist criteria.")
        return

    total_sns = len(df_filtered_view)
    if st.session_state.review_idx >= total_sns:
        st.session_state.review_idx = 0

    current_row = df_filtered_view.iloc[st.session_state.review_idx]
    current_sn = str(current_row['SN'])
    
    unit_folder = Path(img_folder) / current_sn
    json_path = unit_folder / "metadata.json"

    error_desc = str(current_row.get('Error Description', 'None')).strip()
    has_pipeline_error = error_desc != "" and error_desc.lower() != "none"

    # --- ATOMIC HOOK: Sync Session State with Local JSON Database ---
    if current_sn not in st.session_state.annotations:
        if json_path.exists():
            try:
                with open(json_path, "r") as jf:
                    j_data = json.load(jf)
                
                saved_status = j_data.get("Status", "Review (Flagged for Double Check)")
                if has_pipeline_error:
                    initial_flc = "Yes"
                elif saved_status == "FLC Required (Human Reviewed)":
                    initial_flc = "Yes"
                else:
                    initial_flc = "No"
                
                st.session_state.annotations[current_sn] = {
                    'pos 1': j_data["positions"].get("pos 1", {}).get("human_annotation", "o"),
                    'pos 3': j_data["positions"].get("pos 3", {}).get("human_annotation", "o"),
                    'pos 5': j_data["positions"].get("pos 5", {}).get("human_annotation", "o"),
                    'pos 7': j_data["positions"].get("pos 7", {}).get("human_annotation", "o"),
                    'pos 9': j_data["positions"].get("pos 9", {}).get("human_annotation", "o"),
                    'need_flc': initial_flc,
                    'touched': True if saved_status not in ["Review (Flagged for Double Check)", "FLC Required (Error/Warning)"] else False
                }
            except Exception:
                json_path.unlink(missing_ok=True)

        if current_sn not in st.session_state.annotations:
            valid_classes = {'o', 'f', 'sn', 'n'}
            def get_default_annotation(pos_num):
                pred_val = str(current_row.get(f'pos {pos_num} predict', 'o')).lower().strip()
                return pred_val if pred_val in valid_classes else 'o'

            initial_status = st.session_state.live_statuses.get(current_sn, "Review (Flagged for Double Check)")
            st.session_state.annotations[current_sn] = {
                'pos 1': get_default_annotation(1),
                'pos 3': get_default_annotation(3),
                'pos 5': get_default_annotation(5),
                'pos 7': get_default_annotation(7),
                'pos 9': get_default_annotation(9),
                'need_flc': "Yes" if (has_pipeline_error or initial_status == "FLC Required (Human Reviewed)") else "No",
                'touched': True if initial_status in ["FLC Required (Human Reviewed)", "Pass (Human Reviewed)"] else False
            }

    # --- EXPORT PIPELINE AGGREGATOR ENGINE ---
    def auto_save_and_compile_master():
        """Saves current interactive parameters directly to individual unit JSON and master CSV mapping."""
        positions = [1, 3, 5, 7, 9]
        current_flc = st.session_state.annotations[current_sn]['need_flc']
        is_touched = st.session_state.annotations[current_sn]['touched']
        
        if has_pipeline_error:
            live_status = "FLC Required (Error/Warning)"
        elif not is_touched:
            live_status = "Review (Flagged for Double Check)"
        else:
            live_status = "FLC Required (Human Reviewed)" if current_flc == "Yes" else "Pass (Human Reviewed)"

        st.session_state.live_statuses[current_sn] = live_status

        # 1. Update unique unit JSON Database
        unit_folder.mkdir(parents=True, exist_ok=True)
        updated_json_payload = {
            "SN": current_sn,
            "Status": live_status,
            "Error Description": error_desc,
            "positions": {}
        }
        for p in positions:
            updated_json_payload["positions"][f"pos {p}"] = {
                "model_predict": str(current_row.get(f'pos {p} predict', 'o')).lower(),
                "model_confidence": str(current_row.get(f'pos {p} confidence', 'N/A')),
                "human_annotation": st.session_state.annotations[current_sn][f'pos {p}']
            }
        with open(json_path, "w") as jf:
            json.dump(updated_json_payload, jf, indent=4)

        # 2. Recompile Unified Master Output Log File (Outputs ALL units processed in batch)
        annotated_rows = []
        for index, row in df_master.iterrows():
            sn_loop = str(row['SN'])
            loop_json_path = Path(img_folder) / sn_loop / "metadata.json"
            loop_error = str(row.get('Error Description', 'None')).strip()
            loop_has_error = loop_error != "" and loop_error.lower() != "none"
            
            if loop_json_path.exists():
                try:
                    with open(loop_json_path, "r") as jf:
                        d = json.load(jf)
                    anno = {f'pos {p}': d["positions"][f"pos {p}"]["human_annotation"] for p in positions}
                    action_status = d.get("Status", "Review (Flagged for Double Check)")
                    if loop_has_error:
                        action_status = "FLC Required (Error/Warning)"
                except Exception:
                    cached = st.session_state.annotations.get(sn_loop, st.session_state.annotations[current_sn])
                    anno = {f'pos {p}': cached[f'pos {p}'] for p in positions}
                    action_status = "FLC Required (Error/Warning)" if loop_has_error else ("FLC Required (Human Reviewed)" if cached['need_flc'] == "Yes" else "Pass (Human Reviewed)")
            else:
                if sn_loop in st.session_state.annotations:
                    cached = st.session_state.annotations[sn_loop]
                    anno = {f'pos {p}': cached[f'pos {p}'] for p in positions}
                    if loop_has_error:
                        action_status = "FLC Required (Error/Warning)"
                    elif not cached['touched']:
                        action_status = "Review (Flagged for Double Check)"
                    else:
                        action_status = "FLC Required (Human Reviewed)" if cached['need_flc'] == "Yes" else "Pass (Human Reviewed)"
                else:
                    anno = {'pos 1': 'o', 'pos 3': 'o', 'pos 5': 'o', 'pos 7': 'o', 'pos 9': 'o'}
                    if loop_has_error:
                        action_status = "FLC Required (Error/Warning)"
                    else:
                        action_status = st.session_state.live_statuses.get(sn_loop, row.get("Status", "Pass (Program Pass)"))

            correct_o = sum(1 for p in positions if str(row.get(f'pos {p} predict', '')).lower() == 'o' and anno[f'pos {p}'] == 'o')
            incorrect_o = sum(1 for p in positions if str(row.get(f'pos {p} predict', '')).lower() == 'o' and anno[f'pos {p}'] != 'o')

            annotated_rows.append({
                "SN": sn_loop,
                "pos 1": anno['pos 1'], "pos 3": anno['pos 3'], "pos 5": anno['pos 5'], "pos 7": anno['pos 7'], "pos 9": anno['pos 9'],
                "pos 1 predict": row.get('pos 1 predict', ''), "pos 1 confidence": row.get('pos 1 confidence', ''),
                "pos 3 predict": row.get('pos 3 predict', ''), "pos 3 confidence": row.get('pos 3 confidence', ''),
                "pos 5 predict": row.get('pos 5 predict', ''), "pos 5 confidence": row.get('pos 5 confidence', ''),
                "pos 7 predict": row.get('pos 7 predict', ''), "pos 7 confidence": row.get('pos 7 confidence', ''),
                "pos 9 predict": row.get('pos 9 predict', ''), "pos 9 confidence": row.get('pos 9 confidence', ''),
                "Status": action_status,
                "number of correct \"o\"": correct_o, "number of incorrect \"o\"": incorrect_o
            })
        
        if annotated_rows:
            pd.DataFrame(annotated_rows).to_csv(Path(img_folder) / out_name, index=False)

    # --- HIDDEN KEYBOARD ROUTING NAVIGATION BRIDGE ---
    st.markdown('<div class="hidden-trigger-deck">', unsafe_allow_html=True)
    if st.button("BRIDGE_PREV", key="btn_bridge_prev"):
        if st.session_state.review_idx > 0:
            auto_save_and_compile_master()  
            st.session_state.review_idx -= 1
            st.rerun()
    if st.button("BRIDGE_NEXT", key="btn_bridge_next"):
        if st.session_state.review_idx < total_sns - 1:
            auto_save_and_compile_master()  
            st.session_state.review_idx += 1
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    def render_control_deck(location_key):
        st.write(f"### Record: {st.session_state.review_idx + 1} / {total_sns}")
        
        current_flc = st.session_state.annotations[current_sn].get('need_flc', 'No')
        is_touched = st.session_state.annotations[current_sn].get('touched', False)
        
        if has_pipeline_error:
            live_action_status = "FLC Required (Error/Warning)"
        elif not is_touched:
            live_action_status = "Review (Flagged for Double Check)"
        else:
            live_action_status = "FLC Required (Human Reviewed)" if current_flc == "Yes" else "Pass (Human Reviewed)"
        
        st.write(f"**Serial Number (SN):** `{current_sn}` | **Current Status:** `{live_action_status}`")
        if has_pipeline_error:
            st.error(f"⚠️ **Pipeline Failure/Warning:** {error_desc}")

        f_col1, f_col2, _ = st.columns([1.5, 1.5, 7])
        with f_col1:
            if st.button("⚠️ Need FLC", key=f"flc_y_{location_key}_{current_sn}", type="primary" if (is_touched and current_flc == "Yes") else "secondary", width='stretch'):
                st.session_state.annotations[current_sn]['need_flc'] = "Yes"
                st.session_state.annotations[current_sn]['touched'] = True
                auto_save_and_compile_master()
                st.rerun()
        with f_col2:
            if st.button("✅ Don't Need FLC", key=f"flc_n_{location_key}_{current_sn}", type="primary" if (is_touched and current_flc == "No" and not has_pipeline_error) else "secondary", width='stretch', disabled=has_pipeline_error):
                st.session_state.annotations[current_sn]['need_flc'] = "No"
                st.session_state.annotations[current_sn]['touched'] = True
                auto_save_and_compile_master()
                st.rerun()

        nav_col1, nav_col2, _ = st.columns([1.5, 1.5, 7])
        with nav_col1:
            if st.button("⬅️ Previous", key=f"nav_prev_{location_key}", width='stretch') and st.session_state.review_idx > 0:
                auto_save_and_compile_master()
                st.session_state.review_idx -= 1
                st.rerun()
        with nav_col2:
            if st.button("Next ➡️", key=f"nav_next_{location_key}", width='stretch') and st.session_state.review_idx < total_sns - 1:
                auto_save_and_compile_master()
                st.session_state.review_idx += 1
                st.rerun()

    render_control_deck(location_key="top")
    st.divider()

    positions = [1, 3, 5, 7, 9]
    classes = ['o', 'f', 'sn', 'n']
    grid_layout = [[1, None, 3], [None, 5, None], [7, None, 9]]

    st.subheader("🖼️ Interactive Operational Grid")
    
    for row in grid_layout:
        cols = st.columns(3)
        for col_idx, pos in enumerate(row):
            with cols[col_idx]:
                if pos is not None:
                    st.markdown(f"**Pos {pos}**")
                    st.caption(f"AI: `{str(current_row.get(f'pos {pos} predict', 'N/A')).lower()}` ({str(current_row.get(f'pos {pos} confidence', 'N/A'))})")
                    
                    sn_folder = Path(img_folder) / current_sn
                    img_files = []
                    if sn_folder.exists():
                        img_files = list(sn_folder.glob(f"{current_sn}-{pos}.*")) + list(sn_folder.glob(f"processed_{current_sn}-{pos}.*"))
                    
                    # Smart Fallback Strategy: dynamically query master processed output directory if target view crop is uncopied
                    if not img_files:
                        processed_out_dir = Path(img_folder).parent.parent / "processed_output" / Path(img_folder).name
                        if processed_out_dir.exists():
                            img_files = list(processed_out_dir.glob(f"{current_sn}-{pos}.*")) + list(processed_out_dir.glob(f"processed_{current_sn}-{pos}.*"))

                    current_selection = st.session_state.annotations[current_sn][f'pos {pos}']

                    if pos in [1, 7]:
                        sub_c1, sub_c2 = st.columns([3, 7])
                        with sub_c1:
                            st.markdown('<div class="vertical-button-box">', unsafe_allow_html=True)
                            for cls_label in classes:
                                if st.button(cls_label.upper(), key=f"lbl_{current_sn}_{pos}_{cls_label}", type="primary" if (current_selection == cls_label) else "secondary", width='stretch'):
                                    st.session_state.annotations[current_sn][f'pos {pos}'] = cls_label
                                    st.session_state.annotations[current_sn]['touched'] = True
                                    
                                    active_labels = [st.session_state.annotations[current_sn][f'pos {p}'] for p in positions]
                                    if has_pipeline_error or (active_labels.count('n') >= 1 or active_labels.count('sn') >= 3):
                                        st.session_state.annotations[current_sn]['need_flc'] = "Yes"
                                    else:
                                        st.session_state.annotations[current_sn]['need_flc'] = "No"
                                        
                                    auto_save_and_compile_master()
                                    st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                        with sub_c2:
                            if img_files: st.image(Image.open(img_files[0]), width='stretch')
                            else: st.error("Missing Image")
                    else:
                        sub_c1, sub_c2 = st.columns([7, 3])
                        with sub_c1:
                            if img_files: st.image(Image.open(img_files[0]), width='stretch')
                            else: st.error("Missing Image")
                        with sub_c2:
                            st.markdown('<div class="vertical-button-box">', unsafe_allow_html=True)
                            for cls_label in classes:
                                if st.button(cls_label.upper(), key=f"lbl_{current_sn}_{pos}_{cls_label}", type="primary" if (current_selection == cls_label) else "secondary", width='stretch'):
                                    st.session_state.annotations[current_sn][f'pos {pos}'] = cls_label
                                    st.session_state.annotations[current_sn]['touched'] = True
                                    
                                    active_labels = [st.session_state.annotations[current_sn][f'pos {p}'] for p in positions]
                                    if has_pipeline_error or (active_labels.count('n') >= 1 or active_labels.count('sn') >= 3):
                                        st.session_state.annotations[current_sn]['need_flc'] = "Yes"
                                    else:
                                        st.session_state.annotations[current_sn]['need_flc'] = "No"
                                        
                                    auto_save_and_compile_master()
                                    st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.write("")

    st.divider()
    render_control_deck(location_key="bottom")
    st.divider()

    st.caption(f"ℹ️ System logs: Live changes automatically compiled to include all units at: `{Path(img_folder) / out_name}`")

    # --- TOP LEVEL DOM SHORTCUT BRIDGE (CHROME ON UBUNTU FIX) ---
    st.markdown("""
        <script>
        (function() {
            const targetWindow = window.top || window;
            if (targetWindow.ubuntuChromeFixApplied) return;
            targetWindow.ubuntuChromeFixApplied = true;

            targetWindow.addEventListener('keydown', function(e) {
                const activeEl = targetWindow.document.activeElement;
                const localActiveEl = document.activeElement;
                if (activeEl && ['input', 'textarea'].includes(activeEl.tagName.toLowerCase()) || activeEl.isContentEditable) return;
                if (localActiveEl && ['input', 'textarea'].includes(localActiveEl.tagName.toLowerCase()) || localActiveEl.isContentEditable) return;

                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                    const rootDoc = window.top.document;
                    const nestedIframes = Array.from(rootDoc.querySelectorAll('iframe'));
                    let targetButton = null;
                    const searchLabel = (e.key === 'ArrowLeft') ? 'BRIDGE_PREV' : 'BRIDGE_NEXT';

                    for (let frame of nestedIframes) {
                        try {
                            const innerButtons = Array.from(frame.contentWindow.document.querySelectorAll('button'));
                            targetButton = innerButtons.find(btn => btn.innerText.trim() === searchLabel);
                            if (targetButton) break;
                        } catch(err) {}
                    }

                    if (targetButton) {
                        e.preventDefault();
                        e.stopPropagation();
                        targetButton.click();
                    }
                }
            }, true);
        })();
        </script>
    """, unsafe_allow_html=True)