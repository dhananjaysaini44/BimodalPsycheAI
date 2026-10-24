"""
Text Feature Extraction and Classical ML Pipeline Definitions for Phase 17A.

Defines:
1. Word TF-IDF (1-2 grams, min_df=2, sublinear_tf=True, L2 norm)
2. Character TF-IDF (char_wb 3-5 grams, min_df=2, sublinear_tf=True, L2 norm)
3. Combined TF-IDF (FeatureUnion of Word and Character TF-IDF)

Coupled into scikit-learn Pipelines with:
- LinearSVC(C=1.0, random_state=42) for PHQ_Binary classification
- Ridge(alpha=1.0, random_state=42) for continuous PHQ_Score regression
"""

from typing import Any, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge

RANDOM_SEED = 42

FEATURE_EXTRACTOR_CONFIGS = {
    "word_tfidf": {
        "description": "Word-level TF-IDF unigram and bigram representation",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
        "norm": "l2",
    },
    "char_tfidf": {
        "description": "Character n-gram within word boundaries TF-IDF (char_wb 3-5)",
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "min_df": 2,
        "sublinear_tf": True,
        "norm": "l2",
    },
    "combined_tfidf": {
        "description": "Concatenated Word (1-2) + Character (char_wb 3-5) TF-IDF via FeatureUnion",
        "sublinear_tf": True,
        "norm": "l2",
    },
}

MODEL_CONFIGS = {
    "classification": {
        "model_name": "linear_svc",
        "class_name": "LinearSVC",
        "params": {
            "C": 1.0,
            "random_state": RANDOM_SEED,
            "max_iter": 2000,
        },
    },
    "regression": {
        "model_name": "ridge",
        "class_name": "Ridge",
        "params": {
            "alpha": 1.0,
            "random_state": RANDOM_SEED,
        },
    },
}


def build_feature_extractor(feature_name: str) -> Any:
    """Instantiates an unfitted scikit-learn feature extractor."""
    if feature_name == "word_tfidf":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
    elif feature_name == "char_tfidf":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
    elif feature_name == "combined_tfidf":
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
        return FeatureUnion([
            ("word", word_vec),
            ("char", char_vec),
        ])
    else:
        raise ValueError(
            f"Unknown feature representation '{feature_name}'. Valid: {list(FEATURE_EXTRACTOR_CONFIGS.keys())}"
        )


def build_classification_pipeline(feature_name: str) -> Pipeline:
    """
    Constructs an end-to-end classification pipeline: Feature Extractor -> LinearSVC.
    All transformations are encapsulated inside the Pipeline to guarantee zero leakage.
    """
    feature_extractor = build_feature_extractor(feature_name)
    clf = LinearSVC(
        C=1.0,
        random_state=RANDOM_SEED,
        max_iter=2000,
    )
    return Pipeline([
        ("features", feature_extractor),
        ("classifier", clf),
    ])


def build_regression_pipeline(feature_name: str) -> Pipeline:
    """
    Constructs an end-to-end regression pipeline: Feature Extractor -> Ridge.
    All transformations are encapsulated inside the Pipeline to guarantee zero leakage.
    """
    feature_extractor = build_feature_extractor(feature_name)
    reg = Ridge(
        alpha=1.0,
        random_state=RANDOM_SEED,
    )
    return Pipeline([
        ("features", feature_extractor),
        ("regressor", reg),
    ])
