import re
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path

from core.config import CONFIG_PATH, SCRIPT_DIR
from utils.file_utils import load_performance_data, save_performance_data

LANG_DEV = {
    "EN": {
        "sub_tab1": "Config", "sub_tab2": "Dataset Merger", "sub_tab3": "Performance Comparison",
        "edit_cfg": "Edit Configuration", "cfg_info": "Modifying this updates the configuration state immediately and auto-regenerates your filtered reports.",
        "apply_session": "Apply Config to Session", "save_disk": "Save & Overwrite config.yaml",
        "cfg_ok": "Configuration applied to session!", "cfg_save_ok": "Saved to disk and applied!",
        "refilter_ok": "🔄 Re-filtered dataset auto-regenerated!", "refilter_row": "New Filtered Row Count",
        "merger_title": "Dataset Merger & Cleaning", "out_cfg": "#### 1. Output Configuration", "out_path_lbl": "Full Output File Path",
        "val_cfg": "#### 2. Value Mapping Configuration", "val_info": "Define how raw values map to final classes. Any row containing a value *not* listed in the 'Original Value' column will be filtered out.",
        "upload_title": "#### 3. Upload Datasets", "upload_lbl": "Upload CSV Datasets", "merge_btn": "Merge & Clean Datasets",
        "perf_title": "Model Performance Comparison", "perf_viz": "#### Metrics Visualization", "perf_log": "#### Performance Log",
        "edit_mode_lbl": "✏️ Enable Editing Mode", "edit_mode_info": "Edit cells directly. To delete a row, check the box on the far left and press 'Delete'. Changes save automatically.",
        "add_record": "#### Add New Training Record", "save_rec_btn": "Save Record",
        "ds_name": "Dataset Name", "ds_size": "Size", "macro_f1": "Macro F1 Score", "accuracy": "Accuracy", "bdr": "Blur Detection Rate", "remarks": "Remarks (Differences)"
    },
    "ZH": {
        "sub_tab1": "算法参数配置", "sub_tab2": "数据集清洗合并", "sub_tab3": "模型性能看板",
        "edit_cfg": "编辑配置文件 (Yaml)", "cfg_info": "在此处修改参数将即时更新会话配置，并自动重新运行数据过滤引擎生成报告。",
        "apply_session": "应用配置到当前会话", "save_disk": "保存并覆盖本地 config.yaml",
        "cfg_ok": "配置已成功应用至当前环境！", "cfg_save_ok": "配置文件已写入磁盘并实时生效！",
        "refilter_ok": "🔄 过滤规则已触发，数据集完成重新汇编！", "refilter_row": "重新过滤后的总行数",
        "merger_title": "数据集多流归并与清洗引擎", "out_cfg": "#### 1. 输出路径配置", "out_path_lbl": "合并后的 CSV 导出完整路径",
        "val_cfg": "#### 2. 标签值映射配置矩阵", "val_info": "定义原始标记类到标准缺陷分类的映射关系。任何包含未在此列表中的'原始值'的记录行都将被过滤剔除。",
        "upload_title": "#### 3. 载入原始数据集", "upload_lbl": "多选上传 CSV 数据源文件", "merge_btn": "执行清洗合并流水线",
        "perf_title": "多版本模型性能交叉比对看板", "perf_viz": "#### 关键指标演进趋势", "perf_log": "#### 历史训练性能日志数据库",
        "edit_mode_lbl": "✏️ 开启配置编辑模式", "edit_mode_info": "可直接双击单元格修改数据。勾选最左侧复选框并按 'Delete' 键删除行。变更将实时存盘。",
        "add_record": "#### 录入全新模型训练指标", "save_rec_btn": "保存记录至本地",
        "ds_name": "数据集名称", "ds_size": "样本容量", "macro_f1": "宏平均 F1 分数 (Macro F1)", "accuracy": "准确率 (Accuracy)", "bdr": "模糊检测捕获率 (BDR)", "remarks": "核心优化备注 (实验差异)"
    }
}

def render_dev_tab():
    lang = st.session_state.get("lang", "EN")
    ln = LANG_DEV[lang]

    sub_config, sub_merge, sub_perf = st.tabs([ln["sub_tab1"], ln["sub_tab2"], ln["sub_tab3"]])

    # --- SUB-TAB 1: Config ---
    with sub_config:
        st.subheader(ln["edit_cfg"])
        st.info(ln["cfg_info"])
        
        new_yaml = st.text_area("config.yaml", value=st.session_state.yaml_text, height=400)
        
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
                    paired_cols = [(f'pos {pos} predict', f'pos {pos} confidence') for pos in target_positions if f'pos {pos} predict' in df_all.columns and f'pos {pos} confidence' in df_all.columns]
                    
                    if paired_cols:
                        pred_frame = pd.DataFrame({pred_col: df_all[pred_col].astype(str).str.strip().str.lower() for pred_col, _ in paired_cols})
                        conf_frame = pd.DataFrame({conf_col: pd.to_numeric(df_all[conf_col].astype(str).str.replace('%', '', regex=False).str.strip(), errors='coerce') for _, conf_col in paired_cols})
                        has_n_mask = pred_frame.eq('n').any(axis=1)
                        sn_single_thresh = config.get("sn_single_threshold_percent", 45)
                        any_sn_single_mask = pd.DataFrame({pred_col: (pred_frame[pred_col] == 'sn') & (conf_frame[conf_col] >= sn_single_thresh) for pred_col, conf_col in paired_cols}).any(axis=1)
                        sn_count_thresh = config.get("sn_count_threshold_percent")
                        sn_count_req = config.get("sn_count_required")
                        
                        if sn_count_thresh is not None and sn_count_req is not None:
                            sn_mask_count = (pd.DataFrame({pred_col: (pred_frame[pred_col] == 'sn') & (conf_frame[conf_col] >= sn_count_thresh) for pred_col, conf_col in paired_cols}).sum(axis=1) >= sn_count_req)
                            selected_mask = has_n_mask | any_sn_single_mask | sn_mask_count
                        else:
                            selected_mask = has_n_mask | any_sn_single_mask

                        df_filtered = df_all[selected_mask].copy()
                        df_filtered.to_csv(filtered_file_csv, index=False, encoding='utf-8-sig')
                        df_filtered.to_excel(filtered_file_xlsx, index=False)
                        
                        st.success(f"{ln['refilter_ok']} -> `{filtered_file_csv.name}`")
                        st.metric(ln["refilter_row"], len(df_filtered))
                except Exception as e:
                    st.error(f"Failed to auto-refilter: {e}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button(ln["apply_session"]):
                try:
                    parsed_config = yaml.safe_load(new_yaml)
                    if isinstance(parsed_config, dict):
                        st.session_state.app_config.update(parsed_config)
                        st.session_state.yaml_text = new_yaml
                        st.success(ln["cfg_ok"])
                        trigger_auto_refilter()
                except Exception as e: st.error(f"Invalid YAML: {e}")
        with col2:
            if st.button(ln["save_disk"]):
                try:
                    with open(CONFIG_PATH, "w") as f: f.write(new_yaml)
                    st.session_state.yaml_text = new_yaml
                    parsed_config = yaml.safe_load(new_yaml)
                    st.session_state.app_config.update(parsed_config)
                    st.success(ln["cfg_save_ok"])
                    trigger_auto_refilter()
                except Exception as e: st.error(f"Failed to save: {e}")

    # --- SUB-TAB 2: Dataset Merger ---
    with sub_merge:
        st.subheader(ln["merger_title"])
        st.write(ln["out_cfg"])
        default_out = str(Path.home() / "566-qa-2" / "merge" / "remapped_merged_dataset.csv")
        full_output_path_str = st.text_input(ln["out_path_lbl"], value=default_out)

        st.write(ln["val_cfg"])
        st.info(ln["val_info"])
        
        default_mapping = pd.DataFrame({
            "Original Value": ["xf", "mf", "sf", "o", "sn", "mn", "xn", "f", "n"],
            "Mapped Value":   ["f",  "f",  "o",  "o", "sn", "n", "n", "f", "n"]
        })
        
        edited_mapping = st.data_editor(default_mapping, num_rows="dynamic", width='stretch', key="mapping_editor")
        mapping_dict = dict(zip(edited_mapping["Original Value"], edited_mapping["Mapped Value"]))
        allowed_letters = edited_mapping["Original Value"].tolist()

        st.write(ln["upload_title"])
        uploaded_files = st.file_uploader(ln["upload_lbl"], type=['csv'], accept_multiple_files=True, key="merge_csvs")
        
        if st.button(ln["merge_btn"], type="primary"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    merged_dataframes = []
                    column_to_drop = [
                        'Capture Time', 'Grade', 'NG', 'Short SN Pairing', '???.1', '???.2', '???_Focus', '???_Focus.1', 
                        'AA?_??S', 'AA?_??S2', 'AA?_??T', 'AA?_??T2', 'AA?_??S.1', 'AA?_??T.1', 'AA?_??S.2', 'AA?_??T.2', 
                        'AA?_??S.3', 'AA?_??T.3', 'AA?_??S.4', 'AA?_??T.4', 'AA?_Tilt-X', 'AA?_Tilt-Y', 'AA?_OC-X', 'AA?_OC-Y', 
                        'AA?_???X', 'AA?_???Y', 'AA?_??0.5F-S', 'AA?_??0.5F-T', 'AA?_??0.5F-S.1', 'AA?_??0.5F-T.1', 
                        'AA?_??0.5F-S.2', 'AA?_??0.5F-T.2', 'AA?_??0.5F-S.3', 'AA?_??0.5F-T.3', 'AA?_????', 'AA?_????.1', 
                        'AA?_????.2', 'AA?_????.3', 'AA?_??0.5??', 'AA?_??0.5??.1', 'AA?_??0.5??.2', 'AA?_??0.5??.3'
                    ]
                    pos_columns = ['pos 1', 'pos 3', 'pos 5', 'pos 7', 'pos 9']

                    for file_obj in uploaded_files:
                        match = re.search(r'(\d{3}[A-Za-z])', file_obj.name)
                        source_id = match.group(1) if match else "Unknown"
                        try:
                            df = pd.read_csv(file_obj)
                            df['Source_ID'] = source_id
                            cols_to_drop_now = column_to_drop + ['Full SN']
                            df = df.rename(columns={"????": "Capture Time", "??": "Grade", "NG??": "NG", "???": "SN", "??.1": "Short SN Pairing"})
                            df = df.drop(columns=cols_to_drop_now, errors='ignore').fillna("o")
                            
                            for col in pos_columns:
                                if col in df.columns: df[col] = df[col].astype(str).str.split('/').str[0].str.strip()
                            if 'SN' in df.columns:
                                df['SN'] = df['SN'].astype(str).str.split('/').str[0].str.strip().str.upper().str.replace(r'\.0$', '', regex=True)
                            merged_dataframes.append(df)
                        except Exception as e: st.error(f"Error {file_obj.name}: {e}")

                    if merged_dataframes:
                        final_df = pd.concat(merged_dataframes, ignore_index=True)
                        existing_pos_cols = [col for col in pos_columns if col in final_df.columns]
                        for col in existing_pos_cols: final_df = final_df[final_df[col].isin(allowed_letters)]
                        for col in existing_pos_cols: final_df[col] = final_df[col].replace(mapping_dict)
                        final_df['need_flc'] = "No"

                        full_output_path = Path(full_output_path_str)
                        full_output_path.parent.mkdir(parents=True, exist_ok=True)
                        final_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')

                        st.success(f"Complete -> `{full_output_path}`")
                        st.dataframe(final_df.head(100), width='stretch')
            else: st.warning("Upload CSV datasets first.")

    # --- SUB-TAB 3: Performance Comparison ---
    with sub_perf:
        st.subheader(ln["perf_title"])
        perf_file = SCRIPT_DIR / "performance_log.csv"
        df_perf = load_performance_data(str(perf_file))
        
        if not df_perf.empty:
            st.write(ln["perf_viz"])
            chart_data = df_perf.set_index("Dataset Name")[["Macro F1 Score", "Accuracy", "Blur Detection Rate"]]
            st.line_chart(chart_data)
            
        st.divider()
        col_title, col_toggle = st.columns([4, 1])
        with col_title: st.write(ln["perf_log"])
        with col_toggle: edit_mode = st.toggle(ln["edit_mode_lbl"])

        if edit_mode:
            st.info(ln["edit_mode_info"])
            edited_df = st.data_editor(df_perf, num_rows="dynamic", width='stretch', key="perf_editor")
            if not edited_df.equals(df_perf):
                save_performance_data(edited_df, str(perf_file))
                st.success("Saved!")
                st.rerun()
        else:
            st.dataframe(df_perf, width='stretch', hide_index=True)

        st.divider()
        st.write(ln["add_record"])
        with st.form("add_perf_record", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_ds = st.text_input(ln["ds_name"])
                new_size = st.number_input(ln["ds_size"], min_value=0, step=100)
            with c2:
                new_f1 = st.number_input(ln["macro_f1"], min_value=0.0, max_value=1.0, format="%.4f")
                new_acc = st.number_input(ln["accuracy"], min_value=0.0, max_value=1.0, format="%.4f")
            with c3:
                new_bdr = st.number_input(ln["bdr"], min_value=0.0, max_value=1.0, format="%.4f")
            
            new_remarks = st.text_input(ln["remarks"])
            submitted = st.form_submit_button(ln["save_rec_btn"])
            
            if submitted:
                if new_ds.strip() == "": st.error("Name cannot be empty.")
                else:
                    new_row = pd.DataFrame([{"Dataset Name": new_ds, "Size": new_size, "Remarks": new_remarks, "Macro F1 Score": new_f1, "Accuracy": new_acc, "Blur Detection Rate": new_bdr}])
                    df_perf = pd.concat([df_perf, new_row], ignore_index=True)
                    save_performance_data(df_perf, str(perf_file))
                    st.success("Added!")
                    st.rerun()