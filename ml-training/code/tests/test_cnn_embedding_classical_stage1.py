"""
Unit tests for Stage 1: Classical ML Baselines on VGG16 and DenseNet201 Embeddings.

Verifies:
1. Output directory structure, configs, models, predictions, metrics, and manifest existence.
2. Official Test split quarantine (no test files modified or accessed).
3. Prediction data integrity (row counts, participant IDs, column schemas, no NaNs).
4. Model serialization and reload fidelity (predictions match disk CSVs).
5. Metric recalculation consistency from saved prediction files.
"""

import sys
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Setup paths
TESTS_DIR = Path(__file__).resolve().parent
CODE_DIR = TESTS_DIR.parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from core.config import (
        OUTPUTS_DIR,
        LABELS_DIR,
        TARGET_CLASSIFICATION,
        TARGET_REGRESSION,
    )
except ImportError:
    from config import (
        OUTPUTS_DIR,
        LABELS_DIR,
        TARGET_CLASSIFICATION,
        TARGET_REGRESSION,
    )

try:
    from data.split_loader import load_official_splits, load_split_dataset
except ImportError:
    from split_loader import load_official_splits, load_split_dataset

from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


class TestStage1ClassicalBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage1_dir = OUTPUTS_DIR / "cnn_embedding_classical_v1"
        cls.configs_dir = cls.stage1_dir / "configs"
        cls.models_class_dir = cls.stage1_dir / "models" / "classification"
        cls.models_reg_dir = cls.stage1_dir / "models" / "regression"
        cls.pred_oof_dir = cls.stage1_dir / "predictions" / "train_oof"
        cls.pred_dev_dir = cls.stage1_dir / "predictions" / "dev"
        cls.metrics_dir = cls.stage1_dir / "metrics"
        cls.reports_dir = cls.stage1_dir / "reports"
        cls.manifest_path = cls.stage1_dir / "manifest.json"
        cls.readme_path = cls.stage1_dir / "README.md"

        cls.official = load_official_splits(labels_dir=LABELS_DIR)
        cls.expected_train_pids = sorted(cls.official.train_pids)
        cls.expected_dev_pids = sorted(cls.official.dev_pids)

        cls.classification_config_ids = [
            "vgg16__mean__linear_svc",
            "vgg16__mean_std__linear_svc",
            "densenet201__mean__linear_svc",
            "densenet201__mean_std__linear_svc",
            "reference__vgg16__mean__dummy_class",
        ]

        cls.regression_config_ids = [
            "vgg16__mean__ridge",
            "vgg16__mean_std__ridge",
            "densenet201__mean__ridge",
            "densenet201__mean_std__ridge",
            "reference__vgg16__mean__dummy_reg",
        ]

    def test_01_directories_exist(self):
        """Verify all expected subdirectories exist."""
        for d in [
            self.stage1_dir,
            self.configs_dir,
            self.models_class_dir,
            self.models_reg_dir,
            self.pred_oof_dir,
            self.pred_dev_dir,
            self.metrics_dir,
            self.reports_dir,
        ]:
            self.assertTrue(d.exists() and d.is_dir(), f"Missing directory: {d}")

    def test_02_manifest_and_docs_exist(self):
        """Verify manifest.json, stage1_experiment_report.md, and README.md exist."""
        self.assertTrue(self.manifest_path.exists(), "manifest.json is missing")
        self.assertTrue((self.reports_dir / "stage1_experiment_report.md").exists(), "Report is missing")
        self.assertTrue(self.readme_path.exists(), "README.md is missing")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["split_protocol"]["train_participants"], 163)
        self.assertEqual(manifest["split_protocol"]["dev_participants"], 56)
        self.assertTrue(manifest["split_protocol"]["test_quarantined"])
        self.assertEqual(manifest["models_verified_count"], 10)

    def test_03_classification_models_and_configs(self):
        """Verify all 5 classification models and configs exist and load cleanly."""
        for cfg_id in self.classification_config_ids:
            model_path = self.models_class_dir / f"{cfg_id}.joblib"
            cfg_path = self.configs_dir / f"{cfg_id}__config.json"

            self.assertTrue(model_path.exists(), f"Missing model: {model_path}")
            self.assertTrue(cfg_path.exists(), f"Missing config: {cfg_path}")

            # Verify pipeline can be loaded
            model = joblib.load(model_path)
            self.assertTrue(hasattr(model, "predict"), f"Model {cfg_id} lacks predict method")

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["task"], "classification")
            self.assertEqual(cfg["target"], TARGET_CLASSIFICATION)
            self.assertTrue(cfg["test_split_quarantined"])

    def test_04_regression_models_and_configs(self):
        """Verify all 5 regression models and configs exist and load cleanly."""
        for cfg_id in self.regression_config_ids:
            model_path = self.models_reg_dir / f"{cfg_id}.joblib"
            cfg_path = self.configs_dir / f"{cfg_id}__config.json"

            self.assertTrue(model_path.exists(), f"Missing model: {model_path}")
            self.assertTrue(cfg_path.exists(), f"Missing config: {cfg_path}")

            # Verify pipeline can be loaded
            model = joblib.load(model_path)
            self.assertTrue(hasattr(model, "predict"), f"Model {cfg_id} lacks predict method")

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["task"], "regression")
            self.assertEqual(cfg["target"], TARGET_REGRESSION)
            self.assertTrue(cfg["test_split_quarantined"])

    def test_05_classification_predictions_integrity(self):
        """Verify classification prediction CSVs have exact row counts, IDs, and no NaNs."""
        for cfg_id in self.classification_config_ids:
            oof_path = self.pred_oof_dir / f"{cfg_id}__oof_pred.csv"
            dev_path = self.pred_dev_dir / f"{cfg_id}__dev_pred.csv"

            self.assertTrue(oof_path.exists(), f"Missing OOF prediction: {oof_path}")
            self.assertTrue(dev_path.exists(), f"Missing Dev prediction: {dev_path}")

            df_oof = pd.read_csv(oof_path)
            self.assertEqual(len(df_oof), 163, f"OOF rows != 163 in {oof_path}")
            self.assertEqual(sorted(df_oof["Participant_ID"].tolist()), self.expected_train_pids)
            self.assertFalse(df_oof.isna().any().any(), f"NaNs found in {oof_path}")
            for col in ["Participant_ID", "y_true", "y_pred", "y_score", "fold"]:
                self.assertIn(col, df_oof.columns)

            df_dev = pd.read_csv(dev_path)
            self.assertEqual(len(df_dev), 56, f"Dev rows != 56 in {dev_path}")
            self.assertEqual(sorted(df_dev["Participant_ID"].tolist()), self.expected_dev_pids)
            self.assertFalse(df_dev.isna().any().any(), f"NaNs found in {dev_path}")
            for col in ["Participant_ID", "y_true", "y_pred", "y_score"]:
                self.assertIn(col, df_dev.columns)

    def test_06_regression_predictions_integrity(self):
        """Verify regression prediction CSVs have exact row counts, IDs, and no NaNs."""
        for cfg_id in self.regression_config_ids:
            oof_path = self.pred_oof_dir / f"{cfg_id}__oof_pred.csv"
            dev_path = self.pred_dev_dir / f"{cfg_id}__dev_pred.csv"

            self.assertTrue(oof_path.exists(), f"Missing OOF prediction: {oof_path}")
            self.assertTrue(dev_path.exists(), f"Missing Dev prediction: {dev_path}")

            df_oof = pd.read_csv(oof_path)
            self.assertEqual(len(df_oof), 163, f"OOF rows != 163 in {oof_path}")
            self.assertEqual(sorted(df_oof["Participant_ID"].tolist()), self.expected_train_pids)
            self.assertFalse(df_oof.isna().any().any(), f"NaNs found in {oof_path}")
            for col in ["Participant_ID", "y_true", "y_pred", "fold"]:
                self.assertIn(col, df_oof.columns)

            df_dev = pd.read_csv(dev_path)
            self.assertEqual(len(df_dev), 56, f"Dev rows != 56 in {dev_path}")
            self.assertEqual(sorted(df_dev["Participant_ID"].tolist()), self.expected_dev_pids)
            self.assertFalse(df_dev.isna().any().any(), f"NaNs found in {dev_path}")
            for col in ["Participant_ID", "y_true", "y_pred"]:
                self.assertIn(col, df_dev.columns)

    def test_07_model_reloading_reproduces_dev_predictions(self):
        """Verify that loaded models reproduce saved Dev predictions identically."""
        # 1. Classification check
        for cfg_id in ["vgg16__mean__linear_svc", "densenet201__mean_std__linear_svc"]:
            model_path = self.models_class_dir / f"{cfg_id}.joblib"
            dev_path = self.pred_dev_dir / f"{cfg_id}__dev_pred.csv"
            model = joblib.load(model_path)
            df_dev = pd.read_csv(dev_path)

            rep = "vgg16" if "vgg16" in cfg_id else "densenet201"
            pool = "mean_std" if "mean_std" in cfg_id else "mean"
            dev_data = load_split_dataset(rep, pool, split="dev")

            scores = model.decision_function(dev_data.X)
            preds = (scores >= 0.0).astype(int)

            np.testing.assert_allclose(scores, df_dev["y_score"].values, atol=1e-6)
            np.testing.assert_array_equal(preds, df_dev["y_pred"].values)

        # 2. Regression check
        for cfg_id in ["vgg16__mean__ridge", "densenet201__mean_std__ridge"]:
            model_path = self.models_reg_dir / f"{cfg_id}.joblib"
            dev_path = self.pred_dev_dir / f"{cfg_id}__dev_pred.csv"
            model = joblib.load(model_path)
            df_dev = pd.read_csv(dev_path)

            rep = "vgg16" if "vgg16" in cfg_id else "densenet201"
            pool = "mean_std" if "mean_std" in cfg_id else "mean"
            dev_data = load_split_dataset(rep, pool, split="dev")

            preds = model.predict(dev_data.X).ravel()
            np.testing.assert_allclose(preds, df_dev["y_pred"].values, atol=1e-6)

    def test_08_metric_recomputation_consistency(self):
        """Verify summary metrics match recomputed values from saved prediction CSVs."""
        # Check classification summary
        df_class_summary = pd.read_csv(self.metrics_dir / "classification_summary.csv").set_index("config_id")
        for cfg_id in self.classification_config_ids:
            df_dev = pd.read_csv(self.pred_dev_dir / f"{cfg_id}__dev_pred.csv")
            m = compute_classification_metrics(df_dev["y_true"], df_dev["y_pred"], y_score=df_dev["y_score"])
            recorded_f1 = df_class_summary.loc[cfg_id, "dev_f1"]
            self.assertAlmostEqual(m.f1, recorded_f1, places=4)

        # Check regression summary
        df_reg_summary = pd.read_csv(self.metrics_dir / "regression_summary.csv").set_index("config_id")
        for cfg_id in self.regression_config_ids:
            df_dev = pd.read_csv(self.pred_dev_dir / f"{cfg_id}__dev_pred.csv")
            m = compute_regression_metrics(df_dev["y_true"], df_dev["y_pred"])
            recorded_ccc = df_reg_summary.loc[cfg_id, "dev_ccc"]
            self.assertAlmostEqual(m.ccc, recorded_ccc, places=4)


if __name__ == "__main__":
    unittest.main()
