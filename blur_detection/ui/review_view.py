import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

def render_review_tab():
    st.header("Image Review & Annotation")
    st.info("Load your consolidated batch predictions to review cropped images and add human annotations.")

    # CSS for crisp layout, layering, and centering the hover magnification window
    st.markdown("""
        <style>
        /* Allow images to break out of their standard column boundaries */
        div[data-testid="stImage"] {
            overflow: visible !important;
            position: relative;
        }
        
        div[data-testid="stImage"] img {
            border-radius: 4px;
            image-rendering: pixelated; 
        }
        
        /* THE CENTERING AND RESIZING FIX: 
           When hovered, detach the image from the columns and position it 
           at the exact middle of the screen at 80% width */
        div[data-testid="stImage"]:hover img {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) !important;
            width: 80vw !important;
            height: auto !important;
            max-height: 80vh !important;
            object-fit: contain;
            z-index: 99999 !important;
            box-shadow: 0px 0px 50px rgba(0, 0, 0, 0.85);
            background-color: #1e1e1e;
            padding: 16px;
            border: 1px solid #444;
        }
        
        /* THE BLINKING FIX:
           Creates an invisible protective boundary shield around the column on hover.
           This locks the focus state and completely prevents the edge-flicker race condition. */
        div[data-testid="stImage"]:hover::before {
            content: "";
            position: absolute;
            top: -60px;
            left: -60px;
            right: -300px;
            bottom: -300px;
            background: transparent;
            z-index: 99998;
            pointer-events: auto;
        }
        
        /* Dense padding rules for your grading button layout blocks */
        div[data-testid="stBlock"] button {
            padding: 2px 4px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col_csv, col_img = st.columns(2)
    with col_csv:
        csv_path = st.text_input("Path to Predictions CSV", value=str(Path.home() / "566-qa-2/filtered_images/predictions_filtered.csv"))
    with col_img:
        img_folder = st.text_input("Path to Filtered Images Folder", value=str(Path.home() / "566-qa-2/filtered_images"))

    if not Path(csv_path).exists():
        st.warning("Please provide a valid CSV path to begin.")
        return

    # Load Data
    @st.cache_data(show_spinner=False)
    def load_pred_data(path):
        return pd.read_csv(path)

    c_space, c_ref = st.columns([5, 1])
    with c_ref:
        if st.button("Refresh Data", width="stretch"):
            load_pred_data.clear()  # Clear the cache to force reload on next call
            st.rerun()
    try:
        df_preds = load_pred_data(csv_path)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    # Initialize Annotation Storage in Session State
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'review_idx' not in st.session_state:
        st.session_state.review_idx = 0

    total_sns = len(df_preds)
    
    # Navigation Controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])
    with nav_col1:
        if st.button("⬅️ Previous") and st.session_state.review_idx > 0:
            st.session_state.review_idx -= 1
            st.rerun()
    with nav_col2:
        if st.button("Next ➡️") and st.session_state.review_idx < total_sns - 1:
            st.session_state.review_idx += 1
            st.rerun()
    with nav_col3:
        st.write(f"### Record: {st.session_state.review_idx + 1} / {total_sns}")

    # Current SN Data Configuration
    current_row = df_preds.iloc[st.session_state.review_idx]
    current_sn = str(current_row['SN'])
    
    # Initialize dictionary structure for current SN if it doesn't exist
    if current_sn not in st.session_state.annotations:
        valid_classes = {'o', 'f', 'sn', 'n'}
        
        # Helper to safely match prediction strings or fall back to 'o' if errors occurred
        def get_default_annotation(pos_num):
            pred_val = str(current_row.get(f'pos {pos_num} predict', 'o')).lower().strip()
            return pred_val if pred_val in valid_classes else 'o'

        st.session_state.annotations[current_sn] = {
            'pos 1': get_default_annotation(1),
            'pos 3': get_default_annotation(3),
            'pos 5': get_default_annotation(5),
            'pos 7': get_default_annotation(7),
            'pos 9': get_default_annotation(9),
            'need_flc': 'No'
        }

    # Display Unit Header Meta Info
    st.write(f"**Serial Number (SN):** `{current_sn}` | **Action Required:** `{current_row.get('Action Required', 'N/A')}`")

    # --- FLC Option Buttons ---
    st.write("**FLC Requirement Status:**")
    flc_val = st.session_state.annotations[current_sn].get('need_flc', 'No')
    
    f_col1, f_col2, _ = st.columns([1.5, 1.5, 7])
    with f_col1:
        if st.button("⚠️ Need FLC", key=f"flc_y_{current_sn}", type="primary" if flc_val == "Yes" else "secondary", use_container_width=True):
            st.session_state.annotations[current_sn]['need_flc'] = "Yes"
            st.rerun()
    with f_col2:
        if st.button("✅ Don't Need FLC", key=f"flc_n_{current_sn}", type="primary" if flc_val == "No" else "secondary", use_container_width=True):
            st.session_state.annotations[current_sn]['need_flc'] = "No"
            st.rerun()

    st.divider()

    # Define the 5 camera positions
    positions = [1, 3, 5, 7, 9]
    cols = st.columns(5)
    classes = ['o', 'f', 'sn', 'n']
    
    # Display Images and Interactive Grid Buttons
    for idx, pos in enumerate(positions):
        with cols[idx]:
            st.markdown(f"### Pos {pos}")
            
            # Extract historical pipeline context
            pred = str(current_row.get(f'pos {pos} predict', 'N/A')).lower()
            conf = str(current_row.get(f'pos {pos} confidence', 'N/A'))
            
            # Build structured paths for exact boundaries
            sn_folder = Path(img_folder) / current_sn
            img_files = []
            if sn_folder.exists():
                img_files = list(sn_folder.glob(f"{current_sn}-{pos}.*")) + \
                            list(sn_folder.glob(f"processed_{current_sn}-{pos}.*"))
            
            if img_files:
                st.image(Image.open(img_files[0]), width='content')
            else:
                st.error("Image Missing")
                
            st.info(f"**AI Pred:** `{pred}` ({conf})")
            
            # Render Human Annotation Matrix Row
            st.write("Human Assessment:")
            current_selection = st.session_state.annotations[current_sn][f'pos {pos}']
            
            # Dynamic horizontal button layout for classes
            btn_grid = st.columns(4)
            for b_idx, cls_label in enumerate(classes):
                with btn_grid[b_idx]:
                    is_active = (current_selection == cls_label)
                    if st.button(
                        cls_label.upper(), 
                        key=f"lbl_{current_sn}_{pos}_{cls_label}", 
                        type="primary" if is_active else "secondary",
                        use_container_width=True
                    ):
                        st.session_state.annotations[current_sn][f'pos {pos}'] = cls_label
                        st.rerun()

    st.divider()
    
    # Save & Export Functionality
    st.write("#### Export Metrics Configuration")
    out_name = st.text_input("Output Filename", value="human_annotated_predictions.csv")
    
    if st.button("Save All Annotations to CSV", type="primary"):
        annotated_rows = []
        
        for index, row in df_preds.iterrows():
            sn = str(row['SN'])
            if sn in st.session_state.annotations:
                anno = st.session_state.annotations[sn]
                
                # Calculate metric counts for verification checks
                correct_o = 0
                incorrect_o = 0
                for p in positions:
                    model_pred = str(row.get(f'pos {p} predict', '')).lower()
                    human_anno = anno[f'pos {p}']
                    
                    if model_pred == 'o' and human_anno == 'o':
                        correct_o += 1
                    elif model_pred == 'o' and human_anno != 'o':
                        incorrect_o += 1

                annotated_rows.append({
                    "SN": sn,
                    "pos 1": anno['pos 1'],
                    "pos 3": anno['pos 3'],
                    "pos 5": anno['pos 5'],
                    "pos 7": anno['pos 7'],
                    "pos 9": anno['pos 9'],
                    "Need FLC": anno.get('need_flc', 'No'), # Appended clean custom column mapping
                    "pos 1 predict": row.get('pos 1 predict', ''),
                    "pos 1 confidence": row.get('pos 1 confidence', ''),
                    "pos 3 predict": row.get('pos 3 predict', ''),
                    "pos 3 confidence": row.get('pos 3 confidence', ''),
                    "pos 5 predict": row.get('pos 5 predict', ''),
                    "pos 5 confidence": row.get('pos 5 confidence', ''),
                    "pos 7 predict": row.get('pos 7 predict', ''),
                    "pos 7 confidence": row.get('pos 7 confidence', ''),
                    "pos 9 predict": row.get('pos 9 predict', ''),
                    "pos 9 confidence": row.get('pos 9 confidence', ''),
                    "Action Required": row.get('Action Required', ''),
                    "number of correct \"o\"": correct_o,
                    "number of incorrect \"o\"": incorrect_o
                })
        
        if annotated_rows:
            df_export = pd.DataFrame(annotated_rows)
            out_path = Path(img_folder) / out_name
            df_export.to_csv(out_path, index=False)
            st.success(f"✅ Successfully compiled and exported {len(annotated_rows)} annotations to `{out_path}`")
        else:
            st.warning("No session variables tracked to complete an export action yet.")