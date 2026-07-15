import streamlit as st
import yaml

# --- Core Imports ---
from core.config import CONFIG_PATH, cfg

# --- UI Imports ---
from ui.batch_view import render_batch_tab
from ui.dev_view import render_dev_tab
from ui.review_view import render_review_tab

# --- Translation Dictionary ---
LANG_DICT = {
    "EN": {
        "title": "Blur Detection Batch Processor",
        "tab_batch": "Batch Process",
        "tab_review": "Review & Annotate",
        "tab_dev": "Config & Development"
    },
    "ZH": {
        "title": "模糊检测批处理器",
        "tab_batch": "批量处理",
        "tab_review": "人工审查与标注",
        "tab_dev": "配置与开发调试"
    }
}

def main():
    st.set_page_config(layout="wide", page_title="Blur Detection QA")
    
    # Language Selection Component
    if 'lang' not in st.session_state:
        st.session_state.lang = cfg.get("default_language", "EN")
        
    col_title, col_lang = st.columns([8, 2])
    with col_lang:
        st.session_state.lang = st.selectbox("🌐 Language / 语言", options=["EN", "ZH"], index=1 if st.session_state.lang == "ZH" else 0)
    
    ln = LANG_DICT[st.session_state.lang]
    
    with col_title:
        st.title(ln["title"])

    # Initialize configuration in session state so edits persist across interactions
    if 'app_config' not in st.session_state:
        st.session_state.app_config = cfg.copy()
    if 'yaml_text' not in st.session_state:
        try:
            with open(CONFIG_PATH, "r") as f:
                st.session_state.yaml_text = f.read()
        except:
            st.session_state.yaml_text = yaml.dump(cfg)

    # Main Tabs using localized text
    tab_batch, tab_review, tab_dev = st.tabs([ln["tab_batch"], ln["tab_review"], ln["tab_dev"]])

    with tab_batch:
        render_batch_tab()

    with tab_dev:
        render_dev_tab()

    with tab_review:
        render_review_tab()

if __name__ == "__main__":
    main()