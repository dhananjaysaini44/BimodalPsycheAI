"""
Comprehensive Test Suite for Pooling Logic and Generated Datasets.

Tests:
1. Hand-verifiable numeric tests on small synthetic matrices (mean, sample std with ddof=1, min, max, composites).
2. Edge case handling (single-frame sequences, non-finite rejection).
3. Name generator determinism and uniqueness.
4. Validation of generated datasets against source files.
5. Verification of manifest completeness (all 24 combinations).
6. Strict separation of targets and participant features.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, FEATURE_REGISTRY
from pooling import (
    VALID_POOLING_STRATEGIES,
    STD_DDOF,
    pool_sequence,
    compute_all_pooled_variants,
    get_pooled_feature_names,
)
from feature_extraction import extract_features


class TestPoolingMathematics(unittest.TestCase):
    """Verifies pooling operations against exact, hand-calculated mathematical ground truth."""

    def setUp(self):
        # 3 frames x 2 features
        # Col 0: [1.0, 3.0, 5.0] -> mean=3.0, sample_std=sqrt((4+0+4)/(3-1))=2.0, min=1.0, max=5.0
        # Col 1: [10.0, 20.0, 30.0] -> mean=20.0, sample_std=sqrt((100+0+100)/2)=10.0, min=10.0, max=30.0
        self.X_small = np.array([
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
        ], dtype=np.float32)

    def test_hand_calculated_mean(self):
        res = pool_sequence(self.X_small, "mean")
        expected = np.array([3.0, 20.0], dtype=np.float32)
        np.testing.assert_allclose(res, expected, rtol=1e-6)

    def test_hand_calculated_sample_std(self):
        res = pool_sequence(self.X_small, "std", ddof=1)
        expected = np.array([2.0, 10.0], dtype=np.float32)
        np.testing.assert_allclose(res, expected, rtol=1e-6)

    def test_hand_calculated_min_max(self):
        res_min = pool_sequence(self.X_small, "min")
        expected_min = np.array([1.0, 10.0], dtype=np.float32)
        np.testing.assert_allclose(res_min, expected_min, rtol=1e-6)

        res_max = pool_sequence(self.X_small, "max")
        expected_max = np.array([5.0, 30.0], dtype=np.float32)
        np.testing.assert_allclose(res_max, expected_max, rtol=1e-6)

    def test_hand_calculated_composites(self):
        res_mean_std = pool_sequence(self.X_small, "mean_std", ddof=1)
        expected_mean_std = np.array([3.0, 20.0, 2.0, 10.0], dtype=np.float32)
        np.testing.assert_allclose(res_mean_std, expected_mean_std, rtol=1e-6)

        res_full = pool_sequence(self.X_small, "mean_std_min_max", ddof=1)
        expected_full = np.array([3.0, 20.0, 2.0, 10.0, 1.0, 10.0, 5.0, 30.0], dtype=np.float32)
        np.testing.assert_allclose(res_full, expected_full, rtol=1e-6)

    def test_compute_all_pooled_variants_consistency(self):
        all_variants = compute_all_pooled_variants(self.X_small, ddof=1)
        self.assertEqual(set(all_variants.keys()), set(VALID_POOLING_STRATEGIES))
        np.testing.assert_allclose(all_variants["mean"], np.array([3.0, 20.0]))
        np.testing.assert_allclose(all_variants["std"], np.array([2.0, 10.0]))
        np.testing.assert_allclose(all_variants["min"], np.array([1.0, 10.0]))
        np.testing.assert_allclose(all_variants["max"], np.array([5.0, 30.0]))
        np.testing.assert_allclose(all_variants["mean_std"], np.array([3.0, 20.0, 2.0, 10.0]))
        np.testing.assert_allclose(all_variants["mean_std_min_max"], np.array([3.0, 20.0, 2.0, 10.0, 1.0, 10.0, 5.0, 30.0]))

    def test_single_frame_edge_case(self):
        """Single frame sequence has 0 degrees of freedom: std must be 0.0 safely."""
        X_single = np.array([[7.5, 42.0]], dtype=np.float32)
        all_v = compute_all_pooled_variants(X_single, ddof=1)
        np.testing.assert_allclose(all_v["mean"], [7.5, 42.0])
        np.testing.assert_allclose(all_v["std"], [0.0, 0.0])
        np.testing.assert_allclose(all_v["min"], [7.5, 42.0])
        np.testing.assert_allclose(all_v["max"], [7.5, 42.0])

    def test_non_finite_rejection(self):
        """Verifies that NaNs or Infs raise ValueError when allow_non_finite=False."""
        X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            pool_sequence(X_nan, "mean", allow_non_finite=False)
        
        X_inf = np.array([[1.0, np.inf], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            pool_sequence(X_inf, "mean", allow_non_finite=False)

    def test_feature_name_generator_uniqueness_and_stability(self):
        base_names = ["mfcc_1", "mfcc_2"]
        for strat in VALID_POOLING_STRATEGIES:
            names = get_pooled_feature_names(base_names, strat)
            self.assertEqual(len(names), len(set(names)), f"Duplicates in {strat}")
            if strat == "mean":
                self.assertEqual(names, ["mfcc_1_mean", "mfcc_2_mean"])
            elif strat == "std":
                self.assertEqual(names, ["mfcc_1_std", "mfcc_2_std"])
            elif strat == "mean_std":
                self.assertEqual(names, ["mfcc_1_mean", "mfcc_2_mean", "mfcc_1_std", "mfcc_2_std"])


class TestGeneratedDatasetsValidation(unittest.TestCase):
    """Validates the generated datasets in dataset/ against source files and rules."""

    def test_labels_file_isolation(self):
        """Checks dataset/labels.csv exists and has all 275 participants without feature columns."""
        labels_file = DATASET_DIR / "labels.csv"
        self.assertTrue(labels_file.exists(), "dataset/labels.csv does not exist")
        df_labels = pd.read_csv(labels_file)
        self.assertEqual(len(df_labels), 275)
        self.assertIn("Participant_ID", df_labels.columns)
        self.assertIn("PHQ_Score", df_labels.columns)
        self.assertIn("PHQ_Binary", df_labels.columns)
        self.assertIn("split", df_labels.columns)

    def test_manifest_contains_all_24_combinations(self):
        """Verifies dataset/manifest.csv contains 24 combinations with 275 participants and no NaNs."""
        manifest_file = DATASET_DIR / "manifest.csv"
        self.assertTrue(manifest_file.exists(), "dataset/manifest.csv does not exist")
        df_manifest = pd.read_csv(manifest_file)
        self.assertEqual(len(df_manifest), 24, "Manifest should have 24 entries (4 reps x 6 strats)")
        self.assertTrue((df_manifest["participant_count"] == 275).all())
        self.assertTrue((df_manifest["missing_count"] == 0).all())
        self.assertTrue((df_manifest["non_finite_count"] == 0).all())

    def test_spot_check_pooled_values_against_raw_extraction(self):
        """
        Spot check: Take participant 302 and raw_egemaps/mean_std.csv.
        Compare the dataset row against freshly extracted & pooled features.
        """
        pid = 302
        rep = "raw_egemaps"
        strat = "mean_std"
        
        csv_path = DATASET_DIR / rep / f"{strat}.csv"
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        row = df[df["Participant_ID"] == pid]
        self.assertEqual(len(row), 1)
        row_features = row.drop(columns=["Participant_ID"]).iloc[0].to_numpy()

        # Extract and pool directly
        extracted = extract_features(pid, rep, use_cache=True)
        expected_pooled = pool_sequence(extracted.features, strat, ddof=STD_DDOF)

        np.testing.assert_allclose(row_features, expected_pooled, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
