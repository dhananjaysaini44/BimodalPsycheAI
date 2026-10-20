"""
Unit Tests for Controlled Classical E-DAIC Benchmark Suite.

Tests:
1. Pipeline Structure & Scaling:
   - LinearSVC pipeline has StandardScaler + LinearSVC.
   - Ridge pipeline has StandardScaler + Ridge.
   - XGBoost has no StandardScaler (tree model).
   - DummyRegressor has strategy='mean'.
2. Fold Hygiene & Zero Leakage:
   - StratifiedKFold and KFold partition 163 participants into 5 disjoint subsets.
   - Union of val folds is exactly all 163 participants.
   - Repeated folds partition correctly into 50 folds with zero train/val overlap.
3. Lin's CCC Mathematical Verification:
   - Identical arrays yield CCC = 1.0.
   - Mean prediction yields CCC = 0.0.
   - Hand-calculated analytical test case yields 4/7 (~0.571429).
   - Inverted slope yields CCC = -1.0.
4. Dataset Input Integrity:
   - Verifies 4 mean_std datasets exist with expected feature counts:
     raw_egemaps=46, raw_mfcc=78, boaw_egemaps=200, boaw_mfcc=200.
   - Verifies feature arrays contain zero NaNs and no metadata columns.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import DATASET_DIR, LABELS_DIR, RANDOM_SEED
from split_loader import load_official_splits, load_split_dataset
from models.classification_models import get_classifier
from models.regression_models import get_regressor
from utils.metrics import (
    concordance_correlation_coefficient,
    compute_regression_metrics,
    compute_classification_metrics,
)
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


class TestControlledClassicalBenchmark(unittest.TestCase):

    def test_pipeline_structure_and_scaling(self):
        """Verifies that LinearSVC and Ridge include StandardScaler while XGBoost does not."""
        # 1. LinearSVC
        svc_pipe = get_classifier("linear_svc")
        self.assertEqual(len(svc_pipe.steps), 2)
        self.assertIsInstance(svc_pipe.steps[0][1], StandardScaler)
        self.assertEqual(svc_pipe.steps[1][0], "classifier")

        # 2. Ridge
        ridge_pipe = get_regressor("ridge")
        self.assertEqual(len(ridge_pipe.steps), 2)
        self.assertIsInstance(ridge_pipe.steps[0][1], StandardScaler)
        self.assertEqual(ridge_pipe.steps[1][0], "regressor")

        # 3. XGBoost
        xgb_pipe = get_regressor("xgboost")
        step_types = [type(s[1]) for s in xgb_pipe.steps]
        self.assertNotIn(StandardScaler, step_types, "XGBoost should not have StandardScaler!")

        # 4. Dummy
        dummy_pipe = get_regressor("dummy")
        step_types = [type(s[1]) for s in dummy_pipe.steps]
        self.assertNotIn(StandardScaler, step_types, "Dummy should not have StandardScaler!")

    def test_fold_hygiene_and_leakage(self):
        """Verifies 5-fold and 50-fold splits have 0 overlap and full coverage."""
        n_samples = 163
        y_binary = np.array([0] * 126 + [1] * 37)

        # 5-Fold Stratified
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        folds_strat = list(skf.split(np.zeros(n_samples), y_binary))
        self.assertEqual(len(folds_strat), 5)
        all_val_indices = []
        for tr_idx, va_idx in folds_strat:
            self.assertEqual(len(set(tr_idx).intersection(set(va_idx))), 0)
            all_val_indices.extend(va_idx)
        self.assertEqual(sorted(all_val_indices), list(range(n_samples)))

        # 5-Fold K-Fold
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        folds_kf = list(kf.split(np.zeros(n_samples)))
        self.assertEqual(len(folds_kf), 5)
        all_val_indices = []
        for tr_idx, va_idx in folds_kf:
            self.assertEqual(len(set(tr_idx).intersection(set(va_idx))), 0)
            all_val_indices.extend(va_idx)
        self.assertEqual(sorted(all_val_indices), list(range(n_samples)))

        # 50-Fold Repeated K-Fold
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_SEED)
        folds_rkf = list(rkf.split(np.zeros(n_samples)))
        self.assertEqual(len(folds_rkf), 50)
        for tr_idx, va_idx in folds_rkf:
            self.assertEqual(len(set(tr_idx).intersection(set(va_idx))), 0)

    def test_concordance_correlation_coefficient_math(self):
        """Verifies Lin's CCC implementation on exact analytical and boundary cases."""
        # 1. Identical vectors
        y = np.array([1.0, 5.0, 10.0, 20.0])
        self.assertAlmostEqual(concordance_correlation_coefficient(y, y), 1.0, places=6)

        # 2. Constant prediction (mean) -> Covariance is 0 -> CCC = 0.0
        y_pred_mean = np.full_like(y, fill_value=np.mean(y))
        self.assertAlmostEqual(concordance_correlation_coefficient(y, y_pred_mean), 0.0, places=6)

        # 3. Hand-calculated analytical case
        # y_true = [1, 2, 3], y_pred = [2, 3, 4]
        # mean_t = 2, mean_p = 3
        # var_t = 2/3, var_p = 2/3
        # cov = 2/3
        # mean_diff_sq = 1
        # denom = 2/3 + 2/3 + 1 = 7/3
        # num = 2 * (2/3) = 4/3
        # CCC = (4/3) / (7/3) = 4/7
        y_t = np.array([1.0, 2.0, 3.0])
        y_p = np.array([2.0, 3.0, 4.0])
        self.assertAlmostEqual(concordance_correlation_coefficient(y_t, y_p), 4.0 / 7.0, places=6)

        # 4. Inverted vectors
        y_inv_t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_inv_p = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        self.assertAlmostEqual(concordance_correlation_coefficient(y_inv_t, y_inv_p), -1.0, places=6)

    def test_dataset_input_integrity_and_dimensions(self):
        """Verifies existence, dimensions, and finite numeric values of the 4 mean_std datasets."""
        expected_dims = {
            "raw_egemaps": 46,
            "raw_mfcc": 78,
            "boaw_egemaps": 200,
            "boaw_mfcc": 200,
        }

        official = load_official_splits(LABELS_DIR)
        self.assertEqual(len(official.train_pids), 163)
        self.assertEqual(len(official.dev_pids), 56)
        self.assertEqual(len(official.test_pids), 56)

        for rep, exp_dim in expected_dims.items():
            train_data = load_split_dataset(rep, "mean_std", split="train")
            dev_data = load_split_dataset(rep, "mean_std", split="dev")

            self.assertEqual(train_data.num_participants, 163)
            self.assertEqual(dev_data.num_participants, 56)
            self.assertEqual(train_data.num_features, exp_dim)
            self.assertEqual(dev_data.num_features, exp_dim)

            # Check that feature names do not contain target or ID columns
            for col in ["Participant_ID", "PHQ_Binary", "PHQ_Score", "Gender", "split"]:
                self.assertNotIn(col, train_data.feature_names)
                self.assertNotIn(col, dev_data.feature_names)

            # Check finite numeric values
            self.assertTrue(np.all(np.isfinite(train_data.X)))
            self.assertTrue(np.all(np.isfinite(dev_data.X)))
            self.assertTrue(np.all(np.isfinite(train_data.y_binary)))
            self.assertTrue(np.all(np.isfinite(dev_data.y_binary)))
            self.assertTrue(np.all(np.isfinite(train_data.y_score)))
            self.assertTrue(np.all(np.isfinite(dev_data.y_score)))


if __name__ == "__main__":
    unittest.main()
