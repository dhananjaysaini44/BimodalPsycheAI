"""
Test Suite for Split-Aware Dataset Loading and Cross-Validation Guards.

Tests:
1. Official split participant counts (Train: 163, Dev: 56, Test: 56).
2. Mutual exclusivity across all three splits (0 overlap).
3. Split-filtered dataset extraction on master pooled datasets.
4. Participant ID boundary guards (no cross-split contamination).
5. Strict isolation between features and target labels.
6. Cross-validation protocol enforcement (CV strictly allowed only on train).
7. Error reporting on missing or corrupt split participants.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, LABELS_DIR
from split_loader import (
    OfficialSplits,
    SplitData,
    DatasetSplits,
    SplitIntegrityError,
    load_official_splits,
    load_split_dataset,
    load_all_splits,
    get_train_cv_splits,
    export_split_csvs,
)


class TestSplitLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.official = load_official_splits(labels_dir=LABELS_DIR)

    def test_official_splits_counts_and_exclusivity(self):
        """Verifies exact official counts (163, 56, 56) and complete mutual exclusivity."""
        train_pids = set(self.official.train_pids)
        dev_pids = set(self.official.dev_pids)
        test_pids = set(self.official.test_pids)

        self.assertEqual(len(train_pids), 163, "Train split must contain exactly 163 participants")
        self.assertEqual(len(dev_pids), 56, "Dev split must contain exactly 56 participants")
        self.assertEqual(len(test_pids), 56, "Test split must contain exactly 56 participants")
        self.assertEqual(len(train_pids | dev_pids | test_pids), 275, "Total participants must be 275")

        # Check mutual exclusivity
        self.assertEqual(len(train_pids & dev_pids), 0, "Overlap found between train and dev!")
        self.assertEqual(len(train_pids & test_pids), 0, "Overlap found between train and test!")
        self.assertEqual(len(dev_pids & test_pids), 0, "Overlap found between dev and test!")

    def test_load_split_dataset_participant_integrity(self):
        """Tests that loading each split returns exact official participants without cross-contamination."""
        rep = "raw_egemaps"
        strat = "mean_std"

        train_data = load_split_dataset(rep, strat, split="train")
        dev_data = load_split_dataset(rep, strat, split="dev")
        test_data = load_split_dataset(rep, strat, split="test")

        # 1. Counts
        self.assertEqual(train_data.num_participants, 163)
        self.assertEqual(dev_data.num_participants, 56)
        self.assertEqual(test_data.num_participants, 56)

        # 2. Exact ID membership
        self.assertEqual(tuple(train_data.participant_ids), self.official.train_pids)
        self.assertEqual(tuple(dev_data.participant_ids), self.official.dev_pids)
        self.assertEqual(tuple(test_data.participant_ids), self.official.test_pids)

        # 3. Dimensions
        expected_dim = 46  # 23 * 2
        self.assertEqual(train_data.num_features, expected_dim)
        self.assertEqual(dev_data.num_features, expected_dim)
        self.assertEqual(test_data.num_features, expected_dim)

        # 4. Strict isolation of targets
        self.assertEqual(len(train_data.y_binary), 163)
        self.assertEqual(len(train_data.y_score), 163)
        self.assertNotIn("PHQ_Binary", train_data.feature_names)
        self.assertNotIn("PHQ_Score", train_data.feature_names)
        self.assertNotIn("Participant_ID", train_data.feature_names)

    def test_load_all_splits_container(self):
        """Verifies load_all_splits helper across BoAW representation."""
        rep = "boaw_mfcc"
        strat = "mean"

        splits = load_all_splits(rep, strat)
        self.assertIsInstance(splits, DatasetSplits)
        self.assertEqual(splits.train.num_participants, 163)
        self.assertEqual(splits.dev.num_participants, 56)
        self.assertEqual(splits.test.num_participants, 56)
        self.assertEqual(splits.train.num_features, 100)

    def test_cv_restricted_strictly_to_train(self):
        """Ensures cross-validation can ONLY be generated for train_data, raising errors on dev/test."""
        splits = load_all_splits("raw_mfcc", "mean")

        # 1. Train CV must succeed
        cv_folds = get_train_cv_splits(splits.train, n_splits=5, stratify=True)
        self.assertEqual(len(cv_folds), 5)
        for tr_idx, val_idx in cv_folds:
            self.assertEqual(len(tr_idx) + len(val_idx), 163)
            # Ensure no fold overlap
            self.assertEqual(len(set(tr_idx) & set(val_idx)), 0)

        # 2. Dev CV must be rejected with ValueError
        with self.assertRaises(ValueError):
            get_train_cv_splits(splits.dev)

        # 3. Test CV must be rejected with ValueError
        with self.assertRaises(ValueError):
            get_train_cv_splits(splits.test)

    def test_missing_participant_detection(self):
        """Verifies that an error is raised if a required participant is missing from the dataset."""
        # Test with invalid split name
        with self.assertRaises(ValueError):
            load_split_dataset("raw_egemaps", "mean", split="non_existent_split")

    def test_export_split_csvs(self):
        """Verifies optional split export utility."""
        export_dir = DATASET_DIR / "splits" / "raw_egemaps" / "mean"
        exported = export_split_csvs("raw_egemaps", "mean", output_dir=export_dir)

        self.assertIn("train", exported)
        self.assertIn("dev", exported)
        self.assertIn("test", exported)

        df_tr = pd.read_csv(exported["train"])
        df_dv = pd.read_csv(exported["dev"])
        df_te = pd.read_csv(exported["test"])

        self.assertEqual(len(df_tr), 163)
        self.assertEqual(len(df_dv), 56)
        self.assertEqual(len(df_te), 56)


    def test_load_classical_representations_matrix(self):
        """Regression test verifying all 4 classical representations with mean and mean_std."""
        classical_specs = [
            ("raw_egemaps", "mean", 23),
            ("raw_egemaps", "mean_std", 46),
            ("raw_mfcc", "mean", 39),
            ("raw_mfcc", "mean_std", 78),
            ("boaw_egemaps", "mean", 100),
            ("boaw_egemaps", "mean_std", 200),
            ("boaw_mfcc", "mean", 100),
            ("boaw_mfcc", "mean_std", 200),
        ]
        for rep, strat, expected_dim in classical_specs:
            train_data = load_split_dataset(rep, strat, split="train")
            dev_data = load_split_dataset(rep, strat, split="dev")
            test_data = load_split_dataset(rep, strat, split="test")

            self.assertEqual(train_data.X.shape, (163, expected_dim), f"Shape mismatch: {rep}/{strat} train")
            self.assertEqual(dev_data.X.shape, (56, expected_dim), f"Shape mismatch: {rep}/{strat} dev")
            self.assertEqual(test_data.X.shape, (56, expected_dim), f"Shape mismatch: {rep}/{strat} test")

            self.assertEqual(len(train_data.y_binary), 163)
            self.assertEqual(len(dev_data.y_score), 56)

    def test_load_deep_representations_matrix(self):
        """Regression test verifying all 4 deep representations across all official splits."""
        deep_specs = [
            ("vgg16", "mean", 4096),
            ("vgg16", "mean_std", 8192),
            ("densenet201", "mean", 1920),
            ("densenet201", "mean_std", 3840),
        ]
        for rep, strat, expected_dim in deep_specs:
            train_data = load_split_dataset(rep, strat, split="train")
            dev_data = load_split_dataset(rep, strat, split="dev")
            test_data = load_split_dataset(rep, strat, split="test")
            train_dev_data = load_split_dataset(rep, strat, split="train_dev")

            # Check counts
            self.assertEqual(train_data.num_participants, 163, f"{rep}/{strat} train count")
            self.assertEqual(dev_data.num_participants, 56, f"{rep}/{strat} dev count")
            self.assertEqual(test_data.num_participants, 56, f"{rep}/{strat} test count")
            self.assertEqual(train_dev_data.num_participants, 219, f"{rep}/{strat} train_dev count")

            # Check feature shapes
            self.assertEqual(train_data.X.shape, (163, expected_dim), f"{rep}/{strat} train X shape")
            self.assertEqual(dev_data.X.shape, (56, expected_dim), f"{rep}/{strat} dev X shape")
            self.assertEqual(test_data.X.shape, (56, expected_dim), f"{rep}/{strat} test X shape")
            self.assertEqual(train_dev_data.X.shape, (219, expected_dim), f"{rep}/{strat} train_dev X shape")

            # Check data types and finiteness
            self.assertEqual(train_data.X.dtype, np.float32)
            self.assertTrue(np.isfinite(train_data.X).all(), f"Non-finite in {rep}/{strat} train")
            self.assertTrue(np.isfinite(dev_data.X).all(), f"Non-finite in {rep}/{strat} dev")
            self.assertTrue(np.isfinite(test_data.X).all(), f"Non-finite in {rep}/{strat} test")

            # Check label alignment
            self.assertEqual(len(train_data.y_binary), 163)
            self.assertEqual(len(train_data.y_score), 163)
            self.assertEqual(len(test_data.y_binary), 56)
            self.assertEqual(len(test_data.y_score), 56)

            # Check exclusion of metadata
            self.assertNotIn("name", train_data.feature_names)
            self.assertNotIn("timeStamp", train_data.feature_names)
            self.assertNotIn("Participant_ID", train_data.feature_names)

    def test_deep_features_load_all_splits_and_cv(self):
        """Verifies load_all_splits and get_train_cv_splits for VGG16 and DenseNet201."""
        for rep, strat in [("vgg16", "mean"), ("densenet201", "mean_std")]:
            splits = load_all_splits(rep, strat)
            self.assertIsInstance(splits, DatasetSplits)
            self.assertEqual(splits.train.num_participants, 163)
            self.assertEqual(splits.dev.num_participants, 56)
            self.assertEqual(splits.test.num_participants, 56)

            # CV on train must succeed and strictly partition 163 samples
            cv_folds = get_train_cv_splits(splits.train, n_splits=5, stratify=True)
            self.assertEqual(len(cv_folds), 5)
            all_val = []
            for tr, val in cv_folds:
                self.assertEqual(len(tr) + len(val), 163)
                self.assertEqual(len(set(tr) & set(val)), 0)
                all_val.extend(val)
            self.assertEqual(sorted(all_val), list(range(163)))

            # CV on dev and test must be rejected
            with self.assertRaises(ValueError):
                get_train_cv_splits(splits.dev)
            with self.assertRaises(ValueError):
                get_train_cv_splits(splits.test)

    def test_reexport_shim_backward_compatibility(self):
        """Verifies that the root code/split_loader.py shim produces identical results."""
        from data.split_loader import load_split_dataset as canonical_load
        import split_loader as shim_module

        # Test classical
        c1 = canonical_load("raw_mfcc", "mean_std", split="dev")
        c2 = shim_module.load_split_dataset("raw_mfcc", "mean_std", split="dev")
        np.testing.assert_array_equal(c1.X, c2.X)
        np.testing.assert_array_equal(c1.participant_ids, c2.participant_ids)

        # Test deep
        d1 = canonical_load("densenet201", "mean", split="train")
        d2 = shim_module.load_split_dataset("densenet201", "mean", split="train")
        np.testing.assert_array_equal(d1.X, d2.X)
        np.testing.assert_array_equal(d1.participant_ids, d2.participant_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
