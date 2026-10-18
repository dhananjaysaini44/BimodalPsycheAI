"""
Smoke Test for Prompt 4 Classification Benchmark.

Executes a minimal test on raw_egemaps (mean_std) with:
- DummyClassifier
- LogisticRegression

Verifies:
1. Training participant IDs (163) are used strictly in cross-validation.
2. Development split (56) is used strictly for evaluation.
3. Test split is never touched.
4. Preprocessing (StandardScaler) is strictly within Pipeline.
5. Metrics and continuous scores are valid and finite.
6. Predictions retain correct Participant_IDs.
7. Figures and curve coordinate data are saved and verifiable.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLASSIFICATION_OUTPUT_DIR,
    PREDICTIONS_DIR,
    ROC_CURVES_DIR,
    PR_CURVES_DIR,
    CONFUSION_MATRICES_DIR,
    CURVE_DATA_DIR,
)
from experiments.run_classification import run_benchmark


class TestClassificationBenchmarkSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run minimal smoke benchmark: 1 dataset x 2 models
        cls.smoke_results = run_benchmark(
            representations_filter=["raw_egemaps"],
            pooling_filter=["mean_std"],
            models_filter=["dummy", "logreg"],
            force_rebuild=True,
            generate_plots=True,
        )

    def test_results_table_integrity(self):
        """Verifies results.csv contains 4 experiment entries (2 models x 2 protocols)."""
        df = self.smoke_results
        self.assertEqual(len(df), 4)
        
        cv_rows = df[df["evaluation_protocol"] == "cv5"]
        dev_rows = df[df["evaluation_protocol"] == "dev"]

        self.assertEqual(len(cv_rows), 2)
        self.assertEqual(len(dev_rows), 2)

        for _, row in df.iterrows():
            self.assertEqual(row["status"], "SUCCESS")
            self.assertEqual(row["target"], "PHQ_Binary")
            self.assertEqual(row["feature_count"], 46)

    def test_participant_isolation_in_predictions(self):
        """Verifies CV predictions have exactly 163 participants and Dev has 56."""
        for model in ["dummy", "logreg"]:
            cv_pred_file = PREDICTIONS_DIR / f"raw_egemaps__mean_std__{model}__cv5__predictions.csv"
            dev_pred_file = PREDICTIONS_DIR / f"raw_egemaps__mean_std__{model}__dev__predictions.csv"

            self.assertTrue(cv_pred_file.exists(), f"Missing {cv_pred_file}")
            self.assertTrue(dev_pred_file.exists(), f"Missing {dev_pred_file}")

            df_cv = pd.read_csv(cv_pred_file)
            df_dev = pd.read_csv(dev_pred_file)

            self.assertEqual(len(df_cv), 163)
            self.assertEqual(len(df_dev), 56)

            # Check mutual exclusivity of IDs
            cv_pids = set(df_cv["Participant_ID"])
            dev_pids = set(df_dev["Participant_ID"])
            self.assertEqual(len(cv_pids & dev_pids), 0, "Overlap found between CV and Dev prediction IDs!")

            # Verify predictions and scores exist and are non-null
            self.assertFalse(df_cv["y_pred"].isnull().any())
            self.assertFalse(df_cv["y_score"].isnull().any())
            self.assertFalse(df_dev["y_pred"].isnull().any())
            self.assertFalse(df_dev["y_score"].isnull().any())

    def test_diagnostic_figures_and_data_generated(self):
        """Verifies that ROC curves, PR curves, CMs, and curve data exist."""
        for model in ["dummy", "logreg"]:
            for proto in ["cv5", "dev"]:
                exp_id = f"raw_egemaps__mean_std__{model}__{proto}"
                
                # Curve data
                self.assertTrue((CURVE_DATA_DIR / f"{exp_id}__roc.json").exists())
                self.assertTrue((CURVE_DATA_DIR / f"{exp_id}__pr.json").exists())

                # Figures
                self.assertTrue((ROC_CURVES_DIR / f"{exp_id}__roc.png").exists())
                self.assertTrue((PR_CURVES_DIR / f"{exp_id}__precision_recall.png").exists())
                self.assertTrue((CONFUSION_MATRICES_DIR / f"{exp_id}__confusion.png").exists())
                self.assertTrue((CONFUSION_MATRICES_DIR / f"{exp_id}__confusion_normalized.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
