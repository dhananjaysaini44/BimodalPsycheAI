from pathlib import Path
import sys
import pandas as pd

# Add code directory to sys.path if not present
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from core.config import OUTPUTS_DIR
except ImportError:
    try:
        from config import OUTPUTS_DIR
    except ImportError:
        OUTPUTS_DIR = CODE_DIR.parent / "outputs"


def summarize():
    clf_csv = OUTPUTS_DIR / "classification" / "results.csv"
    reg_csv = OUTPUTS_DIR / "regression" / "results.csv"

    if not clf_csv.exists() or not reg_csv.exists():
        print(f"Results CSVs not found in {OUTPUTS_DIR}")
        return

    df_clf = pd.read_csv(clf_csv)
    df_clf_dev = df_clf[df_clf['evaluation_protocol'] == 'dev'].set_index(['representation', 'pooling_strategy', 'model_name'])

    df_reg = pd.read_csv(reg_csv)
    df_reg_dev = df_reg[df_reg['evaluation_protocol'] == 'dev'].set_index(['representation', 'pooling_strategy', 'model_name'])

    print('=== CLASSIFICATION: BEST CONFIG PER MODEL FAMILY (CV vs DEV) ===')
    for m in df_clf['model_name'].unique():
        best_cv = df_clf[(df_clf['model_name'] == m) & (df_clf['evaluation_protocol'] == 'cv5')].sort_values(by='f1', ascending=False).iloc[0]
        dev_match = df_clf_dev.loc[(best_cv['representation'], best_cv['pooling_strategy'], m)]
        best_dev = df_clf[(df_clf['model_name'] == m) & (df_clf['evaluation_protocol'] == 'dev')].sort_values(by='f1', ascending=False).iloc[0]
        r_cv, p_cv = best_cv['representation'], best_cv['pooling_strategy']
        r_dev, p_dev = best_dev['representation'], best_dev['pooling_strategy']
        print(f"{m:15s} | Top CV: {r_cv:12s} {p_cv:16s} [CV F1: {best_cv['f1']:.3f}, Dev F1: {dev_match['f1']:.3f}, Dev AUC: {dev_match['roc_auc']:.3f}] | Top Dev: {r_dev:12s} {p_dev:16s} [Dev F1: {best_dev['f1']:.3f}, Dev AUC: {best_dev['roc_auc']:.3f}]")

    print('\n=== REGRESSION: BEST CONFIG PER MODEL FAMILY (CV vs DEV) ===')
    for m in df_reg['model_name'].unique():
        best_cv = df_reg[(df_reg['model_name'] == m) & (df_reg['evaluation_protocol'] == 'cv5')].sort_values(by='ccc', ascending=False).iloc[0]
        dev_match = df_reg_dev.loc[(best_cv['representation'], best_cv['pooling_strategy'], m)]
        best_dev = df_reg[(df_reg['model_name'] == m) & (df_reg['evaluation_protocol'] == 'dev')].sort_values(by='ccc', ascending=False).iloc[0]
        r_cv, p_cv = best_cv['representation'], best_cv['pooling_strategy']
        r_dev, p_dev = best_dev['representation'], best_dev['pooling_strategy']
        print(f"{m:18s} | Top CV: {r_cv:12s} {p_cv:16s} [CV CCC: {best_cv['ccc']:.3f}, Dev CCC: {dev_match['ccc']:.3f}, Dev RMSE: {dev_match['rmse']:.2f}] | Top Dev: {r_dev:12s} {p_dev:16s} [Dev CCC: {best_dev['ccc']:.3f}, Dev RMSE: {best_dev['rmse']:.2f}]")


if __name__ == '__main__':
    summarize()
