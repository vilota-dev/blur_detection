import streamlit as st
import yaml
import pandas as pd

from core.config import CONFIG_PATH, SCRIPT_DIR
from utils.file_utils import load_performance_data, save_performance_data

def render_dev_tab():
    sub_config, sub_merge, sub_perf = st.tabs(["Config", "Dataset Merger", "Performance Comparison"])

    # ==========================================
    # --- SUB-TAB 1: Config ---
    # ==========================================
    with sub_config:
        st.subheader("Edit Configuration")
        st.info("Modifying this updates the behavior in the Batch Process tab immediately.")
        
        new_yaml = st.text_area("config.yaml", value=st.session_state.yaml_text, height=400)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Apply Config to Session"):
                try:
                    parsed_config = yaml.safe_load(new_yaml)
                    if isinstance(parsed_config, dict):
                        st.session_state.app_config.update(parsed_config)
                        st.session_state.yaml_text = new_yaml
                        st.success("Configuration applied for this session!")
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
                except Exception as e:
                    st.error(f"Failed to save: {e}")

    # ==========================================
    # --- SUB-TAB 2: Dataset Merger ---
    # ==========================================
    with sub_merge:
        st.subheader("Dataset Merger & Cleaning")
        col_csv1, col_csv2 = st.columns(2)
        with col_csv1:
            csv1 = st.file_uploader("Upload First CSV", type=['csv'], key="merge_csv1")
        with col_csv2:
            csv2 = st.file_uploader("Upload Second CSV", type=['csv'], key="merge_csv2")
        
        if st.button("Merge & Clean Datasets"):
            if csv1 is not None and csv2 is not None:
                st.info("Merge and data cleaning logic pending implementation.")
            else:
                st.warning("Please upload both CSV files to proceed.")

    # ==========================================
    # --- SUB-TAB 3: Performance Comparison ---
    # ==========================================
    with sub_perf:
        st.subheader("Model Performance Comparison")
        perf_file = SCRIPT_DIR / "performance_log.csv"
        
        # Load Data
        df_perf = load_performance_data(str(perf_file))
        
        # --- 1. Visualizations ---
        if not df_perf.empty:
            st.write("#### Metrics Visualization")
            # Set the Dataset Name as the index so Streamlit groups the lines automatically
            chart_data = df_perf.set_index("Dataset Name")[["Macro F1 Score", "Accuracy", "Blur Detection Rate"]]
            st.line_chart(chart_data)
        else:
            st.info("No data available for visualization. Add a record below.")
            
        st.divider()

        # --- 2. Interactive Data Table (View / Edit Mode) ---
        col_title, col_toggle = st.columns([4, 1])
        with col_title:
            st.write("#### Performance Log")
        with col_toggle:
            edit_mode = st.toggle("✏️ Enable Editing Mode")

        if edit_mode:
            st.info("Edit cells directly. To delete a row, check the box on the far left and press 'Delete'. Changes save automatically.")
            
            # st.data_editor replaces st.dataframe to allow dynamic interactions
            edited_df = st.data_editor(
                df_perf,
                num_rows="dynamic", # This enables the UI to add/delete rows
                width='stretch',
                key="perf_editor"
            )
            
            # Check if the user made any changes. If so, save and refresh the charts.
            if not edited_df.equals(df_perf):
                save_performance_data(edited_df, str(perf_file))
                st.success("Changes saved successfully!")
                st.rerun() 
        else:
            st.dataframe(df_perf, width='stretch', hide_index=True)

        st.divider()

        # --- 3. Add New Record Form ---
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