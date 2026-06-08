import re
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path

from core.config import CONFIG_PATH, SCRIPT_DIR
from utils.file_utils import load_performance_data, save_performance_data

def render_dev_tab():
    sub_config, sub_merge, sub_perf = st.tabs(["Config", "Dataset Merger", "Performance Comparison"])

    # ==========================================
    # --- SUB-TAB 1: Config ---
    # ==========================================
    with sub_config:
        st.subheader("Edit Configuration")
        st.info("Modifying this updates the configuration state immediately and auto-regenerates your filtered reports.")
        
        new_yaml = st.text_area("config.yaml", value=st.session_state.yaml_text, height=400)
        
        # --- Mirror the exact threshold-masking logic from core/pipeline.py ---
        def trigger_auto_refilter():
            filt_dir = Path(st.session_state.get('filtered_dir_path', Path.home() / "566-qa-2/filtered_images"))
            consolidated_file = filt_dir / "consolidated_batch_predictions.csv"
            filtered_file_csv = filt_dir / "predictions_filtered.csv"
            filtered_file_xlsx = filt_dir / "predictions_filtered.xlsx"
            
            if consolidated_file.exists():
                try:
                    df_all = pd.read_csv(consolidated_file)
                    config = st.session_state.app_config
                    
                    target_positions = config.get("target_positions", [1, 3, 5, 7, 9])
                    paired_cols = [
                        (f'pos {pos} predict', f'pos {pos} confidence')
                        for pos in target_positions
                        if f'pos {pos} predict' in df_all.columns and f'pos {pos} confidence' in df_all.columns
                    ]
                    
                    if paired_cols:
                        # Rebuild position dataframes for evaluation
                        pred_frame = pd.DataFrame({pred_col: df_all[pred_col].astype(str).str.strip().str.lower() for pred_col, _ in paired_cols})
                        conf_frame = pd.DataFrame({conf_col: pd.to_numeric(df_all[conf_col].astype(str).str.replace('%', '', regex=False).str.strip(), errors='coerce') for _, conf_col in paired_cols})

                        # 1. Check for explicit 'n' failures
                        has_n_mask = pred_frame.eq('n').any(axis=1)

                        # 2. Check for single position threshold gates
                        sn_single_thresh = config.get("sn_single_threshold_percent", 45)
                        any_sn_single_mask = pd.DataFrame({
                            pred_col: (pred_frame[pred_col] == 'sn') & (conf_frame[conf_col] >= sn_single_thresh)
                            for pred_col, conf_col in paired_cols
                        }).any(axis=1)

                        # 3. Check for multi-count requirements
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

                        # Slice out the newly matching structures
                        df_filtered = df_all[selected_mask].copy()
                        
                        # Overwrite outputs dynamically
                        df_filtered.to_csv(filtered_file_csv, index=False, encoding='utf-8-sig')
                        df_filtered.to_excel(filtered_file_xlsx, index=False)
                        
                        st.success(f"🔄 Re-filtered dataset auto-regenerated! Saved to: `{filtered_file_csv.name}`")
                        st.metric("New Filtered Row Count", len(df_filtered))
                    else:
                        st.error("Could not trace valid coordinate matching parameters to filter dataset.")
                except Exception as e:
                    st.error(f"Failed to auto-generate updated filtered files: {e}")
            else:
                st.info("💡 Note: No existing `consolidated_batch_predictions.csv` found to re-filter yet.")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Apply Config to Session"):
                try:
                    parsed_config = yaml.safe_load(new_yaml)
                    if isinstance(parsed_config, dict):
                        st.session_state.app_config.update(parsed_config)
                        st.session_state.yaml_text = new_yaml
                        st.success("Configuration applied to session!")
                        # Trigger immediate regeneration
                        trigger_auto_refilter()
                except Exception as e:
                    st.error(f"Invalid YAML format: {e}")
        with col2:
            if st.button("Save & Overwrite config.yaml"):
                try:
                    with open(CONFIG_PATH, "w") as f:
                        f.write(new_yaml)
                    st.session_state.yaml_text = new_yaml
                    parsed_config = yaml.safe_load(new_yaml)
                    st.session_state.app_config.update(parsed_config)
                    st.success("Saved to disk and applied!")
                    # Trigger immediate regeneration
                    trigger_auto_refilter()
                except Exception as e:
                    st.error(f"Failed to save: {e}")

    # ==========================================
    # --- SUB-TAB 2: Dataset Merger ---
    # ==========================================
    with sub_merge:
        st.subheader("Dataset Merger & Cleaning")
        
        # 1. Output Path Configuration
        st.write("#### 1. Output Configuration")
        default_out = str(Path.home() / "566-qa-2" / "merge" / "remapped_merged_dataset.csv")
        full_output_path_str = st.text_input("Full Output File Path", value=default_out)

        # 2. Value Mapping Editor
        st.write("#### 2. Value Mapping Configuration")
        st.info("Define how raw values map to final classes. Any row containing a value *not* listed in the 'Original Value' column will be filtered out.")
        
        default_mapping = pd.DataFrame({
            "Original Value": ["xf", "mf", "sf", "o", "sn", "mn", "xn", "f", "n"],
            "Mapped Value":   ["f",  "f",  "o",  "o", "sn", "n", "n", "f", "n"]
        })
        
        edited_mapping = st.data_editor(
            default_mapping, 
            num_rows="dynamic", 
            width='stretch',
            key="mapping_editor"
        )
        
        # Convert editor dataframe to dictionary and list for processing
        mapping_dict = dict(zip(edited_mapping["Original Value"], edited_mapping["Mapped Value"]))
        allowed_letters = edited_mapping["Original Value"].tolist()

        # 3. File Upload & Processing
        st.write("#### 3. Upload Datasets")
        uploaded_files = st.file_uploader(
            "Upload CSV Datasets", 
            type=['csv'], 
            accept_multiple_files=True,
            key="merge_csvs"
        )
        
        if st.button("Merge & Clean Datasets", type="primary"):
            if uploaded_files:
                with st.spinner("Processing datasets..."):
                    merged_dataframes = []
                    
                    column_to_drop = [
                        'Capture Time', 'Grade', 'NG', 'Short SN Pairing', 
                        '???.1', '???.2', '???_Focus', '???_Focus.1', 'AA?_??S', 'AA?_??S2', 
                        'AA?_??T', 'AA?_??T2', 'AA?_??S.1', 'AA?_??T.1', 'AA?_??S.2', 'AA?_??T.2', 
                        'AA?_??S.3', 'AA?_??T.3', 'AA?_??S.4', 'AA?_??T.4', 'AA?_Tilt-X', 
                        'AA?_Tilt-Y', 'AA?_OC-X', 'AA?_OC-Y', 'AA?_???X', 'AA?_???Y', 
                        'AA?_??0.5F-S', 'AA?_??0.5F-T', 'AA?_??0.5F-S.1', 'AA?_??0.5F-T.1', 
                        'AA?_??0.5F-S.2', 'AA?_??0.5F-T.2', 'AA?_??0.5F-S.3', 'AA?_??0.5F-T.3', 
                        'AA?_????', 'AA?_????.1', 'AA?_????.2', 'AA?_????.3', 'AA?_??0.5??', 
                        'AA?_??0.5??.1', 'AA?_??0.5??.2', 'AA?_??0.5??.3'
                    ]
                    
                    pos_columns = ['pos 1', 'pos 3', 'pos 5', 'pos 7', 'pos 9']

                    # Read and extract
                    for file_obj in uploaded_files:
                        # Extract 3 digits + 1 letter from the filename
                        match = re.search(r'(\d{3}[A-Za-z])', file_obj.name)
                        source_id = match.group(1) if match else "Unknown"
                        
                        try:
                            df = pd.read_csv(file_obj)
                            
                            # Assign the extracted filename ID to the column
                            df['Source_ID'] = source_id
                            
                            # Add 'Full SN' back to the drop list since we aren't using it for extraction
                            cols_to_drop_now = column_to_drop + ['Full SN']
                            
                            df = df.rename(columns={
                                "????": "Capture Time", 
                                "??": "Grade", 
                                "NG??": "NG", 
                                "???": "SN", 
                                "??.1": "Short SN Pairing"
                            })
                            
                            df = df.drop(columns=cols_to_drop_now, errors='ignore')
                            df = df.fillna("o")
                            
                            for col in pos_columns:
                                if col in df.columns:
                                    df[col] = df[col].astype(str).str.split('/').str[0].str.strip()
                            
                            if 'SN' in df.columns:
                                df['SN'] = df['SN'].astype(str).str.split('/').str[0].str.strip().str.upper().str.replace(r'\.0$', '', regex=True)
                            
                            merged_dataframes.append(df)
                            
                        except Exception as e:
                            st.error(f"Error processing {file_obj.name}: {e}")

                    if merged_dataframes:
                        final_df = pd.concat(merged_dataframes, ignore_index=True)
                        initial_count = len(final_df)
                        
                        existing_pos_cols = [col for col in pos_columns if col in final_df.columns]
                        
                        # Filter out unlisted original values
                        for col in existing_pos_cols:
                            final_df = final_df[final_df[col].isin(allowed_letters)]
                            
                        filtered_count = len(final_df)
                        
                        # Remap values to f, o, n, sn based on user configuration
                        for col in existing_pos_cols:
                            final_df[col] = final_df[col].replace(mapping_dict)

                        final_df['need_flc'] = "No"

                        # Save locally
                        full_output_path = Path(full_output_path_str)
                        full_output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        final_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')

                        # --- Display Results ---
                        st.success(f"Processing Complete! Saved to disk at: `{full_output_path}`")
                        
                        # --- Source ID Distribution ---
                        st.write("#### Rows per Source ID")
                        source_counts = final_df['Source_ID'].value_counts().reset_index()
                        source_counts.columns = ["Source ID", "Row Count"]
                        st.dataframe(source_counts, width='stretch', hide_index=True)
                        
                        # -----------------------------------
                        
                        if existing_pos_cols:
                            st.write("#### Final Value Distribution (All Positions Combined)")
                            counts = final_df[existing_pos_cols].stack().value_counts().reset_index()
                            counts.columns = ["Value", "Total Count"]
                            
                            c_table, c_chart = st.columns([1, 2])
                            with c_table:
                                st.dataframe(counts, width='stretch', hide_index=True)
                            with c_chart:
                                st.bar_chart(counts.set_index("Value"))

                        st.write("#### Data Preview")
                        st.dataframe(final_df.head(100), width='stretch')

                        # Provide direct download
                        csv_data = final_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button(
                            label="Download Merged Dataset",
                            data=csv_data,
                            file_name=Path(full_output_path_str).name,
                            mime="text/csv",
                        )
                    else:
                        st.error("No valid dataframes could be constructed from the uploaded files.")
            else:
                st.warning("Please upload at least one CSV file to proceed.")

    # ==========================================
    # --- SUB-TAB 3: Performance Comparison ---
    # ==========================================
    with sub_perf:
        st.subheader("Model Performance Comparison")
        perf_file = SCRIPT_DIR / "performance_log.csv"
        
        df_perf = load_performance_data(str(perf_file))
        
        if not df_perf.empty:
            st.write("#### Metrics Visualization")
            chart_data = df_perf.set_index("Dataset Name")[["Macro F1 Score", "Accuracy", "Blur Detection Rate"]]
            st.line_chart(chart_data)
        else:
            st.info("No data available for visualization. Add a record below.")
            
        st.divider()

        col_title, col_toggle = st.columns([4, 1])
        with col_title:
            st.write("#### Performance Log")
        with col_toggle:
            edit_mode = st.toggle("✏️ Enable Editing Mode")

        if edit_mode:
            st.info("Edit cells directly. To delete a row, check the box on the far left and press 'Delete'. Changes save automatically.")
            
            edited_df = st.data_editor(
                df_perf,
                num_rows="dynamic",
                width='stretch',
                key="perf_editor"
            )
            
            if not edited_df.equals(df_perf):
                save_performance_data(edited_df, str(perf_file))
                st.success("Changes saved successfully!")
                st.rerun() 
        else:
            st.dataframe(df_perf, width='stretch', hide_index=True)

        st.divider()

        st.write("#### Add New Training Record")
        with st.form("add_perf_record", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_ds = st.text_input("Dataset Name")
                new_size = st.number_input("Size", min_value=0, step=100)
            with c2:
                new_f1 = st.number_input("Macro F1 Score", min_value=0.0, max_value=1.0, format="%.4f")
                new_acc = st.number_input("Accuracy", min_value=0.0, max_value=1.0, format="%.4f")
            with c3:
                new_bdr = st.number_input("Blur Detection Rate", min_value=0.0, max_value=1.0, format="%.4f")
            
            new_remarks = st.text_input("Remarks (Differences)")
            
            submitted = st.form_submit_button("Save Record")
            if submitted:
                if new_ds.strip() == "":
                    st.error("Dataset Name cannot be empty.")
                else:
                    new_row = pd.DataFrame([{
                        "Dataset Name": new_ds,
                        "Size": new_size,
                        "Remarks": new_remarks,
                        "Macro F1 Score": new_f1,
                        "Accuracy": new_acc,
                        "Blur Detection Rate": new_bdr
                    }])
                    df_perf = pd.concat([df_perf, new_row], ignore_index=True)
                    save_performance_data(df_perf, str(perf_file))
                    st.success("Record successfully added!")
                    st.rerun()