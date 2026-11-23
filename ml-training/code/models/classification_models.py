"""
Classifier Registry and Baseline Configurations for E-DAIC Benchmark.

Defines the 10 standard classical baseline classifiers:
1. DummyClassifier (majority baseline)
2. LogisticRegression (with StandardScaler)
3. RidgeClassifier (with StandardScaler)
4. LinearSVC (with StandardScaler)
5. RBF SVC (with StandardScaler, probability=True)
6. KNeighborsClassifier (with StandardScaler)
7. GaussianNB (parametric Bayesian baseline)
8. RandomForestClassifier (tree ensemble)
9. ExtraTreesClassifier (extremely randomized trees)
10. XGBoost Classifier (gradient boosting ensemble)

Enforces strict data-leakage prevention by encapsulating learned preprocessing
(StandardScaler) within a scikit-learn Pipeline.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a baseline classification model."""
    name: str
    display_name: str
    constructor: Callable[[], Any]
    requires_scaling: bool
    supports_predict_proba: bool
    default_params: Dict[str, Any]


def _build_dummy() -> Pipeline:
    return Pipeline([("classifier", DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED))])


def _build_logreg() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


def _build_ridge() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RidgeClassifier(random_state=RANDOM_SEED)),
    ])


def _build_linear_svc() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LinearSVC(max_iter=2000, random_state=RANDOM_SEED, dual="auto")),
    ])


def _build_rbf_svc() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
    ])


def _build_knn() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ])


def _build_gaussian_nb() -> Pipeline:
    return Pipeline([("classifier", GaussianNB())])


def _build_random_forest() -> Pipeline:
    return Pipeline([
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
    ])


def _build_extra_trees() -> Pipeline:
    return Pipeline([
        ("classifier", ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
    ])


def _build_xgboost() -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed or available.")
    return Pipeline([
        ("classifier", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1,
        ))
    ])


# Classifier Registry
CLASSIFIER_REGISTRY: Dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        name="dummy",
        display_name="Dummy (Majority)",
        constructor=_build_dummy,
        requires_scaling=False,
        supports_predict_proba=True,
        default_params={"strategy": "most_frequent"},
    ),
    "logreg": ModelSpec(
        name="logreg",
        display_name="Logistic Regression",
        constructor=_build_logreg,
        requires_scaling=True,
        supports_predict_proba=True,
        default_params={"max_iter": 1000, "C": 1.0},
    ),
    "ridge": ModelSpec(
        name="ridge",
        display_name="Ridge Classifier",
        constructor=_build_ridge,
        requires_scaling=True,
        supports_predict_proba=False,
        default_params={"alpha": 1.0},
    ),
    "linear_svc": ModelSpec(
        name="linear_svc",
        display_name="Linear SVC",
        constructor=_build_linear_svc,
        requires_scaling=True,
        supports_predict_proba=False,
        default_params={"max_iter": 2000, "C": 1.0},
    ),
    "rbf_svc": ModelSpec(
        name="rbf_svc",
        display_name="RBF SVC",
        constructor=_build_rbf_svc,
        requires_scaling=True,
        supports_predict_proba=True,
        default_params={"kernel": "rbf", "C": 1.0, "probability": True},
    ),
    "knn": ModelSpec(
        name="knn",
        display_name="K-Nearest Neighbors",
        constructor=_build_knn,
        requires_scaling=True,
        supports_predict_proba=True,
        default_params={"n_neighbors": 5},
    ),
    "gaussian_nb": ModelSpec(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        constructor=_build_gaussian_nb,
        requires_scaling=False,
        supports_predict_proba=True,
        default_params={},
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        constructor=_build_random_forest,
        requires_scaling=False,
        supports_predict_proba=True,
        default_params={"n_estimators": 100, "max_depth": None},
    ),
    "extra_trees": ModelSpec(
        name="extra_trees",
        display_name="Extra Trees",
        constructor=_build_extra_trees,
        requires_scaling=False,
        supports_predict_proba=True,
        default_params={"n_estimators": 100, "max_depth": None},
    ),
}

if XGBOOST_AVAILABLE:
    CLASSIFIER_REGISTRY["xgboost"] = ModelSpec(
        name="xgboost",
        display_name="XGBoost",
        constructor=_build_xgboost,
        requires_scaling=False,
        supports_predict_proba=True,
        default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    )


def get_classifier(model_name: str) -> Pipeline:
    """Instantiates a clean, unfitted Pipeline for the requested model."""
    if model_name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )
    return CLASSIFIER_REGISTRY[model_name].constructor()


def extract_continuous_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Extracts continuous decision scores or probabilities for the positive class (label 1).
    Prefers predict_proba[:, 1]; falls back to decision_function.
    """
    # 1. Try predict_proba
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        # Check if 2D
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] == 1:
            # Single class edge case (e.g., DummyClassifier predicting only class 0)
            return proba[:, 0]
        return proba.ravel()
    
    # 2. Try decision_function
    if hasattr(pipeline, "decision_function"):
        df_scores = pipeline.decision_function(X)
        return df_scores.ravel()

    # 3. Fallback to hard predictions if neither exists
    return pipeline.predict(X).astype(np.float64)
