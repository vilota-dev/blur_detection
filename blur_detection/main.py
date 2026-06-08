import streamlit as st
import yaml

# --- Core Imports ---
from core.config import CONFIG_PATH, cfg

# --- UI Imports ---
from ui.batch_view import render_batch_tab
from ui.dev_view import render_dev_tab

def main():
    st.set_page_config(layout="wide", page_title="Blur Detection QA")
    st.title("Blur Detection Batch Processor")
    st.markdown("Combine SAM3 segmentation + DINOv3-based classifier for batch predictions")

    # Initialize configuration in session state so edits persist across interactions
    if 'app_config' not in st.session_state:
        st.session_state.app_config = cfg.copy()
    if 'yaml_text' not in st.session_state:
        try:
            with open(CONFIG_PATH, "r") as f:
                st.session_state.yaml_text = f.read()
        except:
            st.session_state.yaml_text = yaml.dump(cfg)

    # Main Tabs
    tab_batch, tab_dev = st.tabs(["Batch Process", "Config & Development"])

    with tab_batch:
        render_batch_tab()

    with tab_dev:
        render_dev_tab()

if __name__ == "__main__":
    main()