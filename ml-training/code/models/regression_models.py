"""
Regression Model Registry and Baseline Configurations for E-DAIC Benchmark.

Defines the 10 standard classical baseline regressors targeting continuous PHQ_Score:
1. DummyRegressor (mean of training folds)
2. LinearRegression (with StandardScaler)
3. Ridge (with StandardScaler)
4. Lasso (with StandardScaler)
5. ElasticNet (with StandardScaler)
6. SVR (RBF kernel, with StandardScaler)
7. KNeighborsRegressor (with StandardScaler)
8. RandomForestRegressor (tree ensemble)
9. ExtraTreesRegressor (extremely randomized trees)
10. XGBRegressor (gradient boosting regression)

Enforces strict data-leakage prevention by encapsulating learned preprocessing
(StandardScaler) within a scikit-learn Pipeline.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import RANDOM_SEED


@dataclass(frozen=True)
class RegressionModelSpec:
    """Specification of a baseline regression model."""
    name: str
    display_name: str
    constructor: Callable[[], Pipeline]
    requires_scaling: bool
    default_params: Dict[str, Any]


def _build_dummy() -> Pipeline:
    return Pipeline([("regressor", DummyRegressor(strategy="mean"))])


def _build_linear_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])


def _build_ridge() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ])


def _build_lasso() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Lasso(alpha=1.0, random_state=RANDOM_SEED, max_iter=2000)),
    ])


def _build_elastic_net() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=2000)),
    ])


def _build_svr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])


def _build_knn() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", KNeighborsRegressor(n_neighbors=5)),
    ])


def _build_random_forest() -> Pipeline:
    return Pipeline([
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
    ])


def _build_extra_trees() -> Pipeline:
    return Pipeline([
        ("regressor", ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
    ])


def _build_xgboost() -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed or available.")
    return Pipeline([
        ("regressor", XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ))
    ])


# Regression Model Registry
REGRESSION_REGISTRY: Dict[str, RegressionModelSpec] = {
    "dummy": RegressionModelSpec(
        name="dummy",
        display_name="Dummy (Mean)",
        constructor=_build_dummy,
        requires_scaling=False,
        default_params={"strategy": "mean"},
    ),
    "linear_regression": RegressionModelSpec(
        name="linear_regression",
        display_name="Linear Regression",
        constructor=_build_linear_regression,
        requires_scaling=True,
        default_params={},
    ),
    "ridge": RegressionModelSpec(
        name="ridge",
        display_name="Ridge Regression",
        constructor=_build_ridge,
        requires_scaling=True,
        default_params={"alpha": 1.0},
    ),
    "lasso": RegressionModelSpec(
        name="lasso",
        display_name="Lasso Regression",
        constructor=_build_lasso,
        requires_scaling=True,
        default_params={"alpha": 1.0, "max_iter": 2000},
    ),
    "elastic_net": RegressionModelSpec(
        name="elastic_net",
        display_name="ElasticNet",
        constructor=_build_elastic_net,
        requires_scaling=True,
        default_params={"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 2000},
    ),
    "svr": RegressionModelSpec(
        name="svr",
        display_name="Support Vector Regression (RBF)",
        constructor=_build_svr,
        requires_scaling=True,
        default_params={"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
    ),
    "knn": RegressionModelSpec(
        name="knn",
        display_name="K-Nearest Neighbors",
        constructor=_build_knn,
        requires_scaling=True,
        default_params={"n_neighbors": 5},
    ),
    "random_forest": RegressionModelSpec(
        name="random_forest",
        display_name="Random Forest Regressor",
        constructor=_build_random_forest,
        requires_scaling=False,
        default_params={"n_estimators": 100, "max_depth": None},
    ),
    "extra_trees": RegressionModelSpec(
        name="extra_trees",
        display_name="Extra Trees Regressor",
        constructor=_build_extra_trees,
        requires_scaling=False,
        default_params={"n_estimators": 100, "max_depth": None},
    ),
}

if XGBOOST_AVAILABLE:
    REGRESSION_REGISTRY["xgboost"] = RegressionModelSpec(
        name="xgboost",
        display_name="XGBoost Regressor",
        constructor=_build_xgboost,
        requires_scaling=False,
        default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    )


def get_regressor(model_name: str) -> Pipeline:
    """Instantiates a clean, unfitted Pipeline for the requested regression model."""
    if model_name not in REGRESSION_REGISTRY:
        raise ValueError(
            f"Unknown regression model '{model_name}'. Available: {list(REGRESSION_REGISTRY.keys())}"
        )
    return REGRESSION_REGISTRY[model_name].constructor()
