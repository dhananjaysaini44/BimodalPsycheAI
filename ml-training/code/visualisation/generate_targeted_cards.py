"""
Helper script to generate all individual figures, 4-panel visual cards, and markdown cards
for targeted audited models (Phase 12 CNN Classical Baselines & Controlled Classical Dummy).
"""
import sys
from pathlib import Path
import pandas as pd
import importlib

# Add visualisation module dir to path
sys.path.append(str(Path("ml-training/code/visualisation").resolve()))

mod_c = importlib.import_module("04_classification_visualizer")
mod_r = importlib.import_module("05_regression_visualizer")

df_reg = pd.read_csv("outputs/model_visualizations_v1/model_registry.csv")
p12_and_ctrl = df_reg[
    df_reg["model_id"].str.contains("cnn_classical") |
    (df_reg["model_id"].str.contains("controlled_classical") & df_reg["model_id"].str.contains("dummy"))
]

print(f"Generating full visual cards and markdown for {len(p12_and_ctrl)} targeted models...")

for idx, row in p12_and_ctrl.iterrows():
    row_dict = row.to_dict()
    m_id = row["model_id"]
    if row["task"] == "classification":
        res = mod_c.process_single_model_worker(row_dict)
        print(f"Classification processed: {m_id} -> {res}")
    else:
        res = mod_r.process_single_reg_model_worker(row_dict)
        print(f"Regression processed: {m_id} -> {res}")

print("Targeted card and visual figure generation complete!")
