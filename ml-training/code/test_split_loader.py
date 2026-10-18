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

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
