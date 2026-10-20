"""
Investigate the RandomForest Anomaly on Development Split (Phase 7).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "classification")
AUDIT_DIR = os.path.join(OUTPUTS_DIR, "audit")
PREDS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

def audit_random_forest_anomaly():
    print("=" * 60)
    print("PHASE 7: RandomForest Anomaly Deep Dive")
    print("=" * 60)

    # 1. Inspect raw_mfcc__mean_std__random_forest__dev
    target_exp = "raw_mfcc__mean_std__random_forest__dev"
    pred_path = os.path.join(PREDS_DIR, f"{target_exp}__predictions.csv")
    df_pred = pd.read_csv(pred_path)
    
    y_true = df_pred['y_true'].to_numpy()
    y_pred = df_pred['y_pred'].to_numpy()
    y_score = df_pred['y_score'].to_numpy()

    print(f"Target: {target_exp}")
    print(f"Total Participants: {len(df_pred)}")
    print(f"True Class Counts: Negative (0) = {np.sum(y_true == 0)}, Positive (1) = {np.sum(y_true == 1)}")
    print(f"Predicted Class Counts: Negative (0) = {np.sum(y_pred == 0)}, Positive (1) = {np.sum(y_pred == 1)}")
    
    print("\nSummary of Predicted Continuous Probabilities (y_score):")
    print(pd.Series(y_score).describe())

    # Check threshold 0.5 vs predicted labels
    manual_preds_at_05 = (y_score >= 0.5).astype(int)
    matches_saved_preds = np.array_equal(y_pred, manual_preds_at_05)
    print(f"\nDo saved y_pred match (y_score >= 0.5)? {matches_saved_preds}")

    # Metrics at tau = 0.5
    f1_05 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec_05 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_05 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    bal_acc_05 = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    print(f"\nAt Default Threshold (tau = 0.5):")
    print(f"  F1-Score: {f1_05:.4f}")
    print(f"  Precision: {prec_05:.4f}")
    print(f"  Recall: {rec_05:.4f}")
    print(f"  Balanced Accuracy: {bal_acc_05:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")

    # Inspect all RF models on Development set to see if this is systematic
    results_csv = os.path.join(OUTPUTS_DIR, "results.csv")
    df_results = pd.read_csv(results_csv)
    df_rf_dev = df_results[(df_results['model_name'] == 'random_forest') & (df_results['evaluation_protocol'] == 'dev')]
    
    print("\nAll 24 RandomForest Models on Development Split:")
    cols_to_show = ['representation', 'pooling_strategy', 'f1', 'precision', 'recall', 'balanced_accuracy', 'roc_auc', 'average_precision', 'tp', 'fp']
    print(df_rf_dev[cols_to_show].to_string())

    # Check threshold sweep for target_exp to explore discriminative potential
    thresholds = np.linspace(0.05, 0.50, 46)
    sweep_results = []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        sweep_results.append({
            'threshold': t,
            'f1': f1_score(y_true, yp, pos_label=1, zero_division=0),
            'precision': precision_score(y_true, yp, pos_label=1, zero_division=0),
            'recall': recall_score(y_true, yp, pos_label=1, zero_division=0),
            'tp': cm[1, 1], 'fp': cm[0, 1], 'fn': cm[1, 0], 'tn': cm[0, 0]
        })
    df_sweep = pd.DataFrame(sweep_results)
    
    # Save sweep analysis
    sweep_csv = os.path.join(AUDIT_DIR, "rf_threshold_sensitivity_audit.csv")
    df_sweep.to_csv(sweep_csv, index=False)
    print(f"\nSaved threshold sensitivity table to {sweep_csv}")
    
    best_sweep = df_sweep.sort_values(by='f1', ascending=False).iloc[0]
    print(f"\nPotential with Prior-Calibrated Threshold (for diagnosis only, not tuning):")
    print(f"  Optimal Threshold: {best_sweep['threshold']:.2f}")
    print(f"  Max Potential F1: {best_sweep['f1']:.4f} (Recall: {best_sweep['recall']:.2f}, Precision: {best_sweep['precision']:.2f})")

    # Generate RF Anomaly Diagnostic Figure
    plt.figure(figsize=(12, 5))

    # Subplot 1: Score Distribution by Class
    plt.subplot(1, 2, 1)
    scores_neg = y_score[y_true == 0]
    scores_pos = y_score[y_true == 1]
    plt.hist(scores_neg, bins=15, alpha=0.6, label='Non-Depressed (0, n=44)', color='blue', density=True)
    plt.hist(scores_pos, bins=15, alpha=0.6, label='Depressed (1, n=12)', color='red', density=True)
    plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    plt.axvline(0.22, color='green', linestyle=':', linewidth=2, label='Training Prior (0.227)')
    plt.title(f"Score Separation: RF on raw_mfcc mean_std\n(ROC-AUC = {roc_auc:.3f})", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Positive Probability P(y=1)")
    plt.ylabel("Density")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Threshold vs F1 / Recall / Precision
    plt.subplot(1, 2, 2)
    plt.plot(df_sweep['threshold'], df_sweep['f1'], label='F1 Score', color='purple', linewidth=2)
    plt.plot(df_sweep['threshold'], df_sweep['recall'], label='Recall', color='red', linestyle='--')
    plt.plot(df_sweep['threshold'], df_sweep['precision'], label='Precision', color='blue', linestyle=':')
    plt.axvline(0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')
    plt.title("Performance vs. Decision Threshold", fontsize=11, fontweight='bold')
    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    p_rf = os.path.join(AUDIT_DIR, "rf_anomaly_analysis.png")
    plt.savefig(p_rf, dpi=300)
    plt.close()
    print(f"Saved RF anomaly diagnostic plot to {p_rf}")

if __name__ == "__main__":
    audit_random_forest_anomaly()
