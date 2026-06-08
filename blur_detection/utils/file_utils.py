import os
from pathlib import Path
import pandas as pd

def get_grid_cell_from_name(filename):
    name, _ = os.path.splitext(filename)
    parts = name.split("-")
    if len(parts) > 1 and parts[-1].isdigit():
        cell_num = int(parts[-1])
        if 1 <= cell_num <= 9:
            return cell_num
    return 1

def extract_sn_and_pos(filename):
    """Helper to extract SN and Position safely"""
    stem = Path(filename).stem.replace("processed_", "")
    parts = stem.split("-")
    sn = parts[0]
    position = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return sn, position

def load_performance_data(filepath="performance_log.csv"):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return pd.DataFrame(columns=[
            "Dataset Name", "Size", "Remarks", 
            "Macro F1 Score", "Accuracy", "Blur Detection Rate"
        ])

def save_performance_data(df, filepath="performance_log.csv"):
    df.to_csv(filepath, index=False)