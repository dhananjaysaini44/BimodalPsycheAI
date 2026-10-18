"""
Unit tests for Regression Metrics (CCC, RMSE, MAE, R^2).
"""

import unittest
import numpy as np
from utils.metrics import concordance_correlation_coefficient, compute_regression_metrics

class TestRegressionMetrics(unittest.TestCase):

    def test_perfect_prediction(self):
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        metrics = compute_regression_metrics(y, y)
        self.assertAlmostEqual(metrics.ccc, 1.0, places=6)
        self.assertAlmostEqual(metrics.rmse, 0.0, places=6)
        self.assertAlmostEqual(metrics.mae, 0.0, places=6)
        self.assertAlmostEqual(metrics.r2, 1.0, places=6)

    def test_constant_dummy_prediction(self):
        # Predict the exact mean
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        y_pred = np.full_like(y, fill_value=np.mean(y))
        metrics = compute_regression_metrics(y, y_pred)
        # CCC for constant prediction is 0.0 because covariance is 0.0
        self.assertAlmostEqual(metrics.ccc, 0.0, places=6)
        # R^2 for predicting the mean is 0.0
        self.assertAlmostEqual(metrics.r2, 0.0, places=6)
        # RMSE is std of y
        self.assertAlmostEqual(metrics.rmse, np.std(y), places=6)

    def test_hand_calculated_ccc(self):
        # Hand-calculated example:
        # y_true = [1, 2, 3] -> mean = 2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3 = 0.666667
        # y_pred = [2, 3, 4] -> mean = 3, var = 2/3 = 0.666667
        # cov = ((1-2)*(2-3) + (2-2)*(3-3) + (3-2)*(4-3))/3 = (1 + 0 + 1)/3 = 2/3 = 0.666667
        # (mean_true - mean_pred)^2 = (2 - 3)^2 = 1.0
        # Denominator = 2/3 + 2/3 + 1.0 = 4/3 + 3/3 = 7/3
        # Numerator = 2 * (2/3) = 4/3
        # CCC = (4/3) / (7/3) = 4/7 approx 0.57142857
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        ccc = concordance_correlation_coefficient(y_true, y_pred)
        self.assertAlmostEqual(ccc, 4.0 / 7.0, places=6)

    def test_inverse_correlation(self):
        # Perfectly negative correlation centered at 0
        y_true = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_pred = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        ccc = concordance_correlation_coefficient(y_true, y_pred)
        self.assertAlmostEqual(ccc, -1.0, places=6)

    def test_identical_constants(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics.ccc, 1.0, places=6)
        self.assertAlmostEqual(metrics.rmse, 0.0, places=6)

if __name__ == "__main__":
    unittest.main()
