"""
Comprehensive Test Suite for E-DAIC Feature Extraction Layer.

Tests:
1. Schema and dimension verification across all 4 representations.
2. Value verification against direct raw file reads (PID 300 and PID 600).
3. Delimiter adaptability (comma in Train/Dev vs semicolon in Test split).
4. Caching mechanism and force_rebuild behavior.
5. Explicit error reporting on non-existent participants (no silent zero substitution).
6. Separation of participant IDs, metadata, and target labels from features.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, CACHE_DIR, FEATURE_REGISTRY
from feature_extraction import (
    ExtractionResult,
    FeatureExtractionError,
    extract_features,
    extract_raw_egemaps,
    extract_raw_mfcc,
    extract_boaw_egemaps,
    extract_boaw_mfcc,
    extract_batch,
    get_feature_path,
)


class TestFeatureExtraction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We test real participants:
        # 300: WoZ participant (Train/Dev, comma-delimited BoAW)
        # 600: Autonomous AI participant (Test, semicolon-delimited BoAW)
        cls.test_pids = [300, 600]

    def test_raw_egemaps_dimensions_and_schema(self):
        """Test Raw eGeMAPS extraction for 300 and 600: expect (T, 23)."""
        for pid in self.test_pids:
            res = extract_raw_egemaps(pid, use_cache=False)
            self.assertIsInstance(res, ExtractionResult)
            self.assertEqual(res.participant_id, pid)
            self.assertEqual(res.feature_name, "raw_egemaps")
            self.assertEqual(res.output_dimensions[1], 23)
            self.assertEqual(len(res.feature_names), 23)
            self.assertEqual(len(res.timestamps), res.output_dimensions[0])
            self.assertFalse(np.isnan(res.features).any())
            self.assertFalse(np.isinf(res.features).any())
            
            # Verify specific feature names
            self.assertIn("Loudness_sma3", res.feature_names)
            self.assertIn("F3amplitudeLogRelF0_sma3nz", res.feature_names)
            self.assertNotIn("name", res.feature_names)
            self.assertNotIn("frameTime", res.feature_names)

    def test_raw_mfcc_dimensions_and_schema(self):
        """Test Raw MFCC extraction for 300 and 600: expect (T, 39)."""
        for pid in self.test_pids:
            res = extract_raw_mfcc(pid, use_cache=False)
            self.assertIsInstance(res, ExtractionResult)
            self.assertEqual(res.participant_id, pid)
            self.assertEqual(res.feature_name, "raw_mfcc")
            self.assertEqual(res.output_dimensions[1], 39)
            self.assertEqual(len(res.feature_names), 39)
            self.assertEqual(len(res.timestamps), res.output_dimensions[0])
            self.assertFalse(np.isnan(res.features).any())
            self.assertFalse(np.isinf(res.features).any())
            
            # Verify specific feature names
            self.assertIn("pcm_fftMag_mfcc[0]", res.feature_names)
            self.assertIn("pcm_fftMag_mfcc_de_de[12]", res.feature_names)
            self.assertNotIn("name", res.feature_names)
            self.assertNotIn("frameTime", res.feature_names)

    def test_boaw_egemaps_dimensions_and_delimiter_handling(self):
        """
        Test BoAW eGeMAPS extraction across splits:
        PID 300 uses comma (',')
        PID 600 uses semicolon (';')
        Both must yield (T, 100) without failure.
        """
        for pid in self.test_pids:
            res = extract_boaw_egemaps(pid, use_cache=False)
            self.assertIsInstance(res, ExtractionResult)
            self.assertEqual(res.participant_id, pid)
            self.assertEqual(res.feature_name, "boaw_egemaps")
            self.assertEqual(res.output_dimensions[1], 100)
            self.assertEqual(len(res.feature_names), 100)
            self.assertEqual(len(res.timestamps), res.output_dimensions[0])
            self.assertFalse(np.isnan(res.features).any())
            self.assertFalse(np.isinf(res.features).any())
            
            # Verify stable naming
            self.assertEqual(res.feature_names[0], "boaw_egemaps_bin_0")
            self.assertEqual(res.feature_names[-1], "boaw_egemaps_bin_99")

    def test_boaw_mfcc_dimensions_and_delimiter_handling(self):
        """
        Test BoAW MFCC extraction across splits:
        PID 300 uses comma (',')
        PID 600 uses semicolon (';')
        Both must yield (T, 100).
        """
        for pid in self.test_pids:
            res = extract_boaw_mfcc(pid, use_cache=False)
            self.assertIsInstance(res, ExtractionResult)
            self.assertEqual(res.participant_id, pid)
            self.assertEqual(res.feature_name, "boaw_mfcc")
            self.assertEqual(res.output_dimensions[1], 100)
            self.assertEqual(len(res.feature_names), 100)
            self.assertEqual(len(res.timestamps), res.output_dimensions[0])
            self.assertFalse(np.isnan(res.features).any())
            self.assertFalse(np.isinf(res.features).any())
            
            # Verify stable naming
            self.assertEqual(res.feature_names[0], "boaw_mfcc_bin_0")
            self.assertEqual(res.feature_names[-1], "boaw_mfcc_bin_99")

    def test_value_parity_against_direct_source_read(self):
        """
        Directly reads source CSV for participant 300 and verifies that
        extracted numeric values and timestamps match with zero drift.
        """
        pid = 300
        # 1. Raw eGeMAPS
        res_ege = extract_raw_egemaps(pid, use_cache=False)
        src_path = get_feature_path(pid, "raw_egemaps")
        df_src = pd.read_csv(src_path, sep=";")
        
        # Verify first 10 frames
        expected_timestamps = df_src["frameTime"].iloc[:10].to_numpy()
        expected_features = df_src.iloc[:10, 2:].to_numpy(dtype=np.float32)
        
        np.testing.assert_allclose(res_ege.timestamps[:10], expected_timestamps, rtol=1e-5)
        np.testing.assert_allclose(res_ege.features[:10], expected_features, rtol=1e-5)

        # 2. BoAW eGeMAPS (PID 600 with semicolon delimiter)
        pid_600 = 600
        res_b_600 = extract_boaw_egemaps(pid_600, use_cache=False)
        src_b_path = get_feature_path(pid_600, "boaw_egemaps")
        df_b_src = pd.read_csv(src_b_path, sep=";", header=None)
        
        expected_b_timestamps = df_b_src.iloc[:10, 1].to_numpy()
        expected_b_features = df_b_src.iloc[:10, 2:].to_numpy(dtype=np.float32)
        
        np.testing.assert_allclose(res_b_600.timestamps[:10], expected_b_timestamps, rtol=1e-5)
        np.testing.assert_allclose(res_b_600.features[:10], expected_b_features, rtol=1e-5)

    def test_caching_and_force_rebuild(self):
        """Verifies that caching writes, subsequent reads are fast and cached, and force_rebuild bypasses cache."""
        pid = 300
        feature_name = "raw_egemaps"
        
        # 1. First extract with force_rebuild -> cached must be False
        res1 = extract_features(pid, feature_name, use_cache=True, force_rebuild=True)
        self.assertFalse(res1.cached)
        
        # 2. Second extract with use_cache=True -> cached must be True
        res2 = extract_features(pid, feature_name, use_cache=True, force_rebuild=False)
        self.assertTrue(res2.cached)
        
        # Verify cached arrays match original exactly
        np.testing.assert_array_equal(res1.features, res2.features)
        np.testing.assert_array_equal(res1.timestamps, res2.timestamps)
        self.assertEqual(res1.feature_names, res2.feature_names)
        self.assertEqual(res1.output_dimensions, res2.output_dimensions)

        # 3. Third extract with force_rebuild=True -> cached must be False
        res3 = extract_features(pid, feature_name, use_cache=True, force_rebuild=True)
        self.assertFalse(res3.cached)
        np.testing.assert_array_equal(res1.features, res3.features)

    def test_non_existent_participant_fails_explicitly(self):
        """Verifies that extracting a non-existent participant raises FeatureExtractionError, not zeros."""
        non_existent_pid = 99999
        with self.assertRaises(FeatureExtractionError) as ctx:
            extract_raw_egemaps(non_existent_pid, use_cache=False)
        
        self.assertIn("Source file not found", str(ctx.exception))
        self.assertEqual(ctx.exception.participant_id, non_existent_pid)

    def test_batch_extraction_and_failure_reporting(self):
        """Tests extract_batch with valid and invalid participants."""
        pids = [300, 99998, 600]
        results, failures = extract_batch(pids, "boaw_mfcc", use_cache=True)
        
        self.assertIn(300, results)
        self.assertIn(600, results)
        self.assertIn(99998, failures)
        self.assertNotIn(99998, results)
        self.assertIn("Source file not found", failures[99998])


if __name__ == "__main__":
    unittest.main(verbosity=2)
