"""
Audit Phases 1-4 for E-DAIC Classification Benchmark.

Covers:
- Phase 1: Workspace and Artifact Inventory
- Phase 2: Split Integrity and Leakage Audit
- Phase 3: Metric and Prediction Correctness Verification
- Phase 4: Five-Fold CV Stability Analysis and Plot Generation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "classification")
AUDIT_DIR = os.path.join(OUTPUTS_DIR, "audit")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LABELS_DIR = os.path.join(BASE_DIR, "labels")

os.makedirs(AUDIT_DIR, exist_ok=True)

def run_phase1_inventory():
    print("=" * 60)
    print("PHASE 1: Workspace and Artifact Inventory")
    print("=" * 60)
    
    inventory = {}
    
    # Check results.csv
    results_csv = os.path.join(OUTPUTS_DIR, "results.csv")
    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        inventory['results_csv_rows'] = len(df_results)
        inventory['results_csv_cols'] = list(df_results.columns)
        inventory['results_csv_unique_experiments'] = df_results['experiment_id'].nunique()
        inventory['results_csv_statuses'] = df_results['status'].value_counts().to_dict()
        inventory['results_csv_protocols'] = df_results['evaluation_protocol'].value_counts().to_dict()
        inventory['results_csv_models'] = df_results['model_name'].value_counts().to_dict()
        inventory['results_csv_representations'] = df_results['representation'].value_counts().to_dict()
        inventory['results_csv_pooling'] = df_results['pooling_strategy'].value_counts().to_dict()
    else:
        inventory['results_csv_exists'] = False

    # Check fold_results.csv
    fold_csv = os.path.join(OUTPUTS_DIR, "fold_results.csv")
    if os.path.exists(fold_csv):
        df_folds = pd.read_csv(fold_csv)
        inventory['fold_results_rows'] = len(df_folds)
        inventory['fold_results_unique_experiments'] = df_folds['experiment_id'].nunique()
        inventory['fold_results_folds_per_exp'] = df_folds.groupby('experiment_id')['fold'].count().value_counts().to_dict()
    else:
        inventory['fold_results_exists'] = False

    # Check directories
    subdirs = ['predictions', 'roc_curves', 'precision_recall_curves', 'confusion_matrices', 'curve_data', 'configs', 'comparisons']
    for sd in subdirs:
        sd_path = os.path.join(OUTPUTS_DIR, sd)
        if os.path.exists(sd_path):
            files = [f for f in os.listdir(sd_path) if os.path.isfile(os.path.join(sd_path, f))]
            inventory[f"{sd}_count"] = len(files)
        else:
            inventory[f"{sd}_count"] = 0

    # Check datasets in dataset/
    manifest_csv = os.path.join(DATASET_DIR, "manifest.csv")
    manifest_json = os.path.join(DATASET_DIR, "manifest.json")
    inventory['manifest_csv_exists'] = os.path.exists(manifest_csv)
    inventory['manifest_json_exists'] = os.path.exists(manifest_json)
    if os.path.exists(manifest_json):
        with open(manifest_json, 'r') as f:
            manifest_data = json.load(f)
            inventory['manifest_dataset_count'] = len(manifest_data)

    print(json.dumps(inventory, indent=2))
    return inventory


def run_phase2_split_integrity():
    print("\n" + "=" * 60)
    print("PHASE 2: Split Integrity and Leakage Audit")
    print("=" * 60)
    
    checks = {}
    
    # 1. Check official split files
    train_split_file = os.path.join(LABELS_DIR, "train_split.csv")
    dev_split_file = os.path.join(LABELS_DIR, "dev_split.csv")
    test_split_file = os.path.join(LABELS_DIR, "test_split.csv")
    
    df_train = pd.read_csv(train_split_file)
    df_dev = pd.read_csv(dev_split_file)
    
    train_ids = set(df_train['Participant_ID'].astype(int))
    dev_ids = set(df_dev['Participant_ID'].astype(int))
    
    checks['train_count'] = len(train_ids)
    checks['dev_count'] = len(dev_ids)
    checks['train_dev_overlap'] = len(train_ids.intersection(dev_ids))
    
    # Verify label distributions
    train_label_dist = df_train['PHQ_Binary'].value_counts().to_dict()
    dev_label_dist = df_dev['PHQ_Binary'].value_counts().to_dict()
    checks['train_labels'] = {str(k): int(v) for k, v in train_label_dist.items()}
    checks['dev_labels'] = {str(k): int(v) for k, v in dev_label_dist.items()}
    
    # 2. Check Predictions Participant Sets
    preds_dir = os.path.join(OUTPUTS_DIR, "predictions")
    pred_files = [f for f in os.listdir(preds_dir) if f.endswith(".csv")]
    
    cv_pred_issues = []
    dev_pred_issues = []
    
    for pf in pred_files:
        fpath = os.path.join(preds_dir, pf)
        df_p = pd.read_csv(fpath)
        p_ids = set(df_p['Participant_ID'].astype(int))
        
        if "__cv5" in pf:
            if len(df_p) != 163:
                cv_pred_issues.append(f"{pf}: expected 163 rows, got {len(df_p)}")
            if len(p_ids) != 163:
                cv_pred_issues.append(f"{pf}: duplicate IDs detected ({len(df_p)} vs {len(p_ids)})")
            if not p_ids.issubset(train_ids):
                cv_pred_issues.append(f"{pf}: non-train IDs found in CV predictions")
            if len(p_ids.intersection(dev_ids)) > 0:
                cv_pred_issues.append(f"{pf}: DEV IDs found in CV predictions!")
        elif "__dev" in pf:
            if len(df_p) != 56:
                dev_pred_issues.append(f"{pf}: expected 56 rows, got {len(df_p)}")
            if len(p_ids) != 56:
                dev_pred_issues.append(f"{pf}: duplicate IDs detected ({len(df_p)} vs {len(p_ids)})")
            if not p_ids.issubset(dev_ids):
                dev_pred_issues.append(f"{pf}: non-dev IDs found in Dev predictions")
            if len(p_ids.intersection(train_ids)) > 0:
                dev_pred_issues.append(f"{pf}: TRAIN IDs found in Dev predictions!")

    checks['cv_prediction_files_checked'] = sum(1 for f in pred_files if "__cv5" in f)
    checks['dev_prediction_files_checked'] = sum(1 for f in pred_files if "__dev" in f)
    checks['cv_prediction_issues'] = cv_pred_issues
    checks['dev_prediction_issues'] = dev_pred_issues
    
    # 3. Check Feature Datasets in dataset/
    dataset_issues = []
    manifest_csv = os.path.join(DATASET_DIR, "manifest.csv")
    df_manifest = pd.read_csv(manifest_csv)
        
    for _, item in df_manifest.iterrows():
        rel_path = item['output_path']
        fpath = os.path.join(BASE_DIR, rel_path)
        df_feat = pd.read_csv(fpath)
        
        # Check rows
        if len(df_feat) != 275:
            dataset_issues.append(f"{rel_path}: expected 275 rows, got {len(df_feat)}")
            
        # Check column names: ensure no label leakage in features
        cols = set(df_feat.columns)
        leakage_cols = {'PHQ_Score', 'PHQ_Binary', 'PHQ8_Binary', 'PHQ8_Score', 'Gender', 'age'}.intersection(cols)
        if len(leakage_cols) > 0:
            dataset_issues.append(f"{rel_path}: target/metadata leakage columns detected: {leakage_cols}")
            
        if 'Participant_ID' not in df_feat.columns:
            dataset_issues.append(f"{rel_path}: missing Participant_ID")
            
    checks['dataset_integrity_issues'] = dataset_issues
    print(json.dumps(checks, indent=2))
    return checks


def run_phase3_metric_verification():
    print("\n" + "=" * 60)
    print("PHASE 3: Metric and Prediction Correctness Verification")
    print("=" * 60)
    
    results_csv = os.path.join(OUTPUTS_DIR, "results.csv")
    df_results = pd.read_csv(results_csv)
    
    target_configs = [
        "boaw_egemaps__max__linear_svc__cv5",
        "raw_mfcc__mean__linear_svc__dev",
        "raw_mfcc__mean_std_min_max__linear_svc__dev",
        "raw_egemaps__mean__dummy__cv5",
        "raw_egemaps__mean__dummy__dev",
        "raw_mfcc__mean_std__random_forest__dev",
        "raw_mfcc__mean__ridge__dev",
        "raw_egemaps__mean__logreg__dev"
    ]
    
    recalc_comparisons = []
    
    for exp_id in target_configs:
        orig_row = df_results[df_results['experiment_id'] == exp_id]
        if orig_row.empty:
            continue
        orig_row = orig_row.iloc[0]
        
        pred_path = os.path.join(OUTPUTS_DIR, "predictions", f"{exp_id}__predictions.csv")
        if not os.path.exists(pred_path):
            continue
            
        df_pred = pd.read_csv(pred_path)
        y_true = df_pred['y_true'].to_numpy()
        y_pred = df_pred['y_pred'].to_numpy()
        continuous_score = df_pred['y_score'].to_numpy() if 'y_score' in df_pred.columns else None
        
        # Recalculate metrics
        rec_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_acc = accuracy_score(y_true, y_pred)
        rec_bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        rec_roc_auc = None
        rec_ap = None
        if continuous_score is not None and not np.isnan(continuous_score).all() and len(np.unique(continuous_score)) > 1:
            try:
                rec_roc_auc = roc_auc_score(y_true, continuous_score)
                rec_ap = average_precision_score(y_true, continuous_score)
            except Exception:
                pass
        elif len(np.unique(continuous_score)) <= 1:
            rec_roc_auc = 0.5
            rec_ap = np.mean(y_true)
            
        # Compare
        tol = 1e-4
        def check_diff(val1, val2):
            if pd.isna(val1) and pd.isna(val2):
                return True, 0.0
            if pd.isna(val1) or pd.isna(val2):
                return False, np.nan
            diff = abs(val1 - val2)
            return diff <= tol, diff
            
        f1_match, f1_diff = check_diff(orig_row['f1'], rec_f1)
        prec_match, prec_diff = check_diff(orig_row['precision'], rec_prec)
        rec_match, rec_diff = check_diff(orig_row['recall'], rec_rec)
        acc_match, acc_diff = check_diff(orig_row['accuracy'], rec_acc)
        bal_acc_match, bal_acc_diff = check_diff(orig_row['balanced_accuracy'], rec_bal_acc)
        
        orig_auc = orig_row['roc_auc']
        auc_match, auc_diff = check_diff(orig_auc, rec_roc_auc) if rec_roc_auc is not None else (True, 0.0)

        recalc_comparisons.append({
            'experiment_id': exp_id,
            'orig_f1': orig_row['f1'], 'rec_f1': rec_f1, 'f1_diff': f1_diff, 'f1_match': f1_match,
            'orig_prec': orig_row['precision'], 'rec_prec': rec_prec, 'prec_diff': prec_diff,
            'orig_rec': orig_row['recall'], 'rec_rec': rec_rec, 'rec_diff': rec_diff,
            'orig_acc': orig_row['accuracy'], 'rec_acc': rec_acc, 'acc_diff': acc_diff,
            'orig_bal_acc': orig_row['balanced_accuracy'], 'rec_bal_acc': rec_bal_acc, 'bal_acc_diff': bal_acc_diff,
            'orig_auc': orig_auc, 'rec_auc': rec_roc_auc, 'auc_diff': auc_diff, 'auc_match': auc_match
        })
        
    df_recalc = pd.DataFrame(recalc_comparisons)
    print(df_recalc.to_string())
    df_recalc.to_csv(os.path.join(AUDIT_DIR, "metric_verification_sample.csv"), index=False)
    return recalc_comparisons


def run_phase4_original_cv_stability():
    print("\n" + "=" * 60)
    print("PHASE 4: Original Five-Fold CV Stability Analysis")
    print("=" * 60)
    
    fold_csv = os.path.join(OUTPUTS_DIR, "fold_results.csv")
    df_folds = pd.read_csv(fold_csv)
    
    # Calculate stability metrics per experiment across the 5 folds
    grouped = df_folds.groupby('experiment_id')
    
    stability_records = []
    
    for exp_id, group in grouped:
        n_folds = len(group)
        rep = group['representation'].iloc[0]
        pooling = group['pooling_strategy'].iloc[0]
        model = group['model_name'].iloc[0]
        
        f1_vals = group['f1'].to_numpy()
        bal_acc_vals = group['balanced_accuracy'].to_numpy()
        auc_vals = group['roc_auc'].dropna().to_numpy()
        ap_vals = group['average_precision'].dropna().to_numpy()
        prec_vals = group['precision'].to_numpy()
        rec_vals = group['recall'].to_numpy()
        acc_vals = group['accuracy'].to_numpy()
        
        # Folds exceeding dummy (Dummy F1 = 0)
        folds_exceed_dummy = int(np.sum(f1_vals > 0.0))
        failed_folds = int(np.sum(group['status'] != 'SUCCESS')) if 'status' in group.columns else 0
        
        stability_records.append({
            'experiment_id': exp_id,
            'representation': rep,
            'pooling_strategy': pooling,
            'model_name': model,
            'n_folds': n_folds,
            'failed_folds': failed_folds,
            'f1_mean': float(np.mean(f1_vals)),
            'f1_std': float(np.std(f1_vals, ddof=1)) if n_folds > 1 else 0.0,
            'f1_min': float(np.min(f1_vals)),
            'f1_max': float(np.max(f1_vals)),
            'f1_range': float(np.max(f1_vals) - np.min(f1_vals)),
            'balanced_acc_mean': float(np.mean(bal_acc_vals)),
            'balanced_acc_std': float(np.std(bal_acc_vals, ddof=1)) if n_folds > 1 else 0.0,
            'roc_auc_mean': float(np.mean(auc_vals)) if len(auc_vals) > 0 else np.nan,
            'roc_auc_std': float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else 0.0,
            'average_precision_mean': float(np.mean(ap_vals)) if len(ap_vals) > 0 else np.nan,
            'average_precision_std': float(np.std(ap_vals, ddof=1)) if len(ap_vals) > 1 else 0.0,
            'precision_mean': float(np.mean(prec_vals)),
            'precision_std': float(np.std(prec_vals, ddof=1)) if n_folds > 1 else 0.0,
            'recall_mean': float(np.mean(rec_vals)),
            'recall_std': float(np.std(rec_vals, ddof=1)) if n_folds > 1 else 0.0,
            'accuracy_mean': float(np.mean(acc_vals)),
            'accuracy_std': float(np.std(acc_vals, ddof=1)) if n_folds > 1 else 0.0,
            'folds_exceeding_dummy': folds_exceed_dummy
        })
        
    df_stability = pd.DataFrame(stability_records)
    stability_csv = os.path.join(AUDIT_DIR, "cv_stability.csv")
    df_stability.to_csv(stability_csv, index=False)
    print(f"Saved fold stability metrics to {stability_csv}")
    
    # Model-level summary
    summary_records = []
    for model_name, m_group in df_stability.groupby('model_name'):
        summary_records.append({
            'model_name': model_name,
            'configs_evaluated': len(m_group),
            'f1_mean_overall': float(m_group['f1_mean'].mean()),
            'f1_std_mean': float(m_group['f1_std'].mean()),
            'balanced_acc_mean_overall': float(m_group['balanced_acc_mean'].mean()),
            'roc_auc_mean_overall': float(m_group['roc_auc_mean'].mean()),
            'mean_folds_exceeding_dummy': float(m_group['folds_exceeding_dummy'].mean()),
            'top_f1_config': m_group.sort_values(by='f1_mean', ascending=False)['experiment_id'].iloc[0],
            'top_f1_mean': float(m_group['f1_mean'].max())
        })
    df_summary = pd.DataFrame(summary_records).sort_values(by='top_f1_mean', ascending=False)
    summary_csv = os.path.join(AUDIT_DIR, "cv_stability_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved stability summary to {summary_csv}")
    print(df_summary.to_string())
    
    # Generate Phase 4 Diagnostic Plots
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_stability[df_stability['model_name'] != 'dummy'],
        x='f1_mean', y='f1_std', hue='model_name', style='representation', s=70, alpha=0.85
    )
    plt.title("5-Fold Cross-Validation: Mean F1 vs. Fold-Level Standard Deviation (Excl. Dummy)", fontsize=13, fontweight='bold')
    plt.xlabel("Mean Fold F1 Score", fontsize=11)
    plt.ylabel("Fold F1 Standard Deviation", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    p1 = os.path.join(AUDIT_DIR, "cv_f1_mean_vs_std.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    
    # Fold-wise metric distributions for top 5 CV configs
    top_5_configs = df_stability.sort_values(by='f1_mean', ascending=False).head(5)['experiment_id'].tolist()
    df_top_folds = df_folds[df_folds['experiment_id'].isin(top_5_configs)]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_top_folds, x='experiment_id', y='f1', palette='viridis')
    sns.stripplot(data=df_top_folds, x='experiment_id', y='f1', color='red', size=8, jitter=0.2)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.title("Fold F1 Distributions for Top 5 Cross-Validation Configurations", fontsize=13, fontweight='bold')
    plt.ylabel("F1 Score", fontsize=11)
    plt.grid(True, axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(AUDIT_DIR, "cv_leading_models_fold_dist.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    
    # Performance vs Variability Comparison across models
    plt.figure(figsize=(10, 5))
    df_plot_summary = df_summary[df_summary['model_name'] != 'dummy']
    plt.errorbar(
        df_plot_summary['model_name'], df_plot_summary['f1_mean_overall'],
        yerr=df_plot_summary['f1_std_mean'], fmt='o', color='#1f77b4',
        ecolor='#d62728', elinewidth=2, capsize=4, markersize=8
    )
    plt.title("Mean Performance & Fold Variability (F1 ± Fold Std) by Model", fontsize=13, fontweight='bold')
    plt.ylabel("Average Fold F1", fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3 = os.path.join(AUDIT_DIR, "cv_metrics_variability_comparison.png")
    plt.savefig(p3, dpi=300)
    plt.close()

    print(f"Generated Phase 4 plots: {p1}, {p2}, {p3}")
    return df_stability, df_summary

if __name__ == "__main__":
    run_phase1_inventory()
    run_phase2_split_integrity()
    run_phase3_metric_verification()
    run_phase4_original_cv_stability()
