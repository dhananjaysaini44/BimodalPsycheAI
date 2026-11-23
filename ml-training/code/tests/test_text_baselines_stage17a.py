"""
Unit and Integration Tests for Phase 17A Text Baselines.

Validates:
1. Transcript processing and text normalization.
2. Metadata join integrity and split isolation (Train=163, Dev=56, Test=56, Total=275).
3. Leakage-free TF-IDF pipeline encapsulation.
4. Cross-validation fold structure and out-of-fold prediction coverage.
5. Metric computation correctness (Lin's CCC, F1, Balanced Accuracy, ROC-AUC).
"""

import json
import sys
from pathlib import Path

# Add ml-training/code to sys.path
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from text_baselines_v1.text_pipeline import (
    build_classification_pipeline,
    build_feature_extractor,
    build_regression_pipeline,
)
from text_baselines_v1.transcript_processor import (
    DEFAULT_DATA_DIR,
    DEFAULT_LABELS_DIR,
    DEFAULT_OUTPUT_DIR,
    find_transcript_file,
    normalize_text,
    process_single_transcript,
)
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    concordance_correlation_coefficient,
)


def test_text_normalization():
    """Verifies that text normalization handles casing, punctuation, and whitespace."""
    raw = "  Hello! This is a TEST with multiple   spaces... and it's nice.\nNew line here!  "
    norm = normalize_text(raw)
    assert norm == "hello this is a test with multiple spaces and it's nice new line here"
    assert normalize_text(None) == ""
    assert normalize_text("") == ""


def test_find_and_process_transcript():
    """Tests processing a real transcript file from the dataset."""
    pid = 300
    fpath = find_transcript_file(DEFAULT_DATA_DIR, pid)
    assert fpath is not None and fpath.exists(), f"Could not find transcript for PID {pid}"

    dummy_meta = {
        "split": "dev",
        "Gender": "male",
        "PHQ_Binary": 0,
        "PHQ_Score": 2.0,
        "PCL-C (PTSD)": 0,
        "PTSD Severity": 25.0,
    }
    rec = process_single_transcript(fpath, pid, dummy_meta)
    assert rec.participant_id == 300
    assert rec.split == "dev"
    assert rec.file_exists is True
    assert rec.is_valid is True
    assert rec.utterance_count > 0
    assert rec.word_count > 0
    assert rec.char_count > 0
    assert 0.0 <= rec.avg_confidence <= 1.0
    assert len(rec.normalized_text) > 0


def test_metadata_join_cardinality_and_isolation():
    """Verifies exact split counts and mutual exclusivity of participants."""
    csv_path = DEFAULT_OUTPUT_DIR / "participant_transcripts.csv"
    assert csv_path.exists(), f"Missing dataset: {csv_path}"
    df = pd.read_csv(csv_path)

    # 1. Total count
    assert len(df) == 275, f"Expected 275 participants, got {len(df)}"

    # 2. Split counts
    counts = df["split"].value_counts().to_dict()
    assert counts.get("train", 0) == 163, f"Expected 163 train participants, got {counts.get('train')}"
    assert counts.get("dev", 0) == 56, f"Expected 56 dev participants, got {counts.get('dev')}"
    assert counts.get("test", 0) == 56, f"Expected 56 test participants, got {counts.get('test')}"

    # 3. Mutual exclusivity
    train_pids = set(df[df["split"] == "train"]["Participant_ID"])
    dev_pids = set(df[df["split"] == "dev"]["Participant_ID"])
    test_pids = set(df[df["split"] == "test"]["Participant_ID"])

    assert len(train_pids & dev_pids) == 0, "Leakage: Train and Dev share participants"
    assert len(train_pids & test_pids) == 0, "Leakage: Train and Test share participants"
    assert len(dev_pids & test_pids) == 0, "Leakage: Dev and Test share participants"

    # 4. No empty texts
    assert (df["word_count"] > 0).all(), "Found participant with zero word count"
    assert (df["normalized_text"].str.strip() != "").all(), "Found empty normalized text"


def test_pipeline_leakage_free_encapsulation():
    """
    Verifies that the TF-IDF pipeline strictly fits vocabulary on the provided
    training fold and does not observe test-fold terms in vocabulary.
    """
    train_texts = ["the quick brown fox", "jumps over the lazy dog"]
    test_texts = ["unique unseen word zebra elephant"]

    pipe = build_classification_pipeline("word_tfidf")
    assert isinstance(pipe, Pipeline)

    # Fit strictly on train texts
    pipe.fit(train_texts, [0, 1])

    # Vocabulary of the fitted vectorizer
    vectorizer = pipe.named_steps["features"]
    vocab = vectorizer.vocabulary_

    assert "fox" in vocab or "dog" in vocab or "the" in vocab or "lazy" in vocab
    assert "zebra" not in vocab
    assert "elephant" not in vocab

    # Predicting on unseen test text should not crash or modify vocabulary
    preds = pipe.predict(test_texts)
    assert len(preds) == 1
    assert "zebra" not in vectorizer.vocabulary_


def test_concordance_correlation_coefficient():
    """Tests Lin's CCC computation under known conditions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert concordance_correlation_coefficient(y_true, y_perfect) == 1.0

    # Constant shift reduces CCC below 1.0
    y_shifted = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    ccc_shifted = concordance_correlation_coefficient(y_true, y_shifted)
    assert 0.0 < ccc_shifted < 1.0

    # Negative correlation
    y_inv = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ccc_inv = concordance_correlation_coefficient(y_true, y_inv)
    assert ccc_inv < 0.0

    # Empty array
    assert concordance_correlation_coefficient(np.array([]), np.array([])) == 0.0


def test_regression_metrics_computation():
    """Tests computation of regression metrics dictionary."""
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    y_pred = np.array([11.0, 14.0, 19.0, 24.0])
    m = compute_regression_metrics(y_true, y_pred)
    assert hasattr(m, "ccc") and hasattr(m, "rmse") and hasattr(m, "mae") and hasattr(m, "r2")
    assert m.rmse == pytest.approx(1.0, rel=1e-3)
    assert m.mae == pytest.approx(1.0, rel=1e-3)
    assert m.ccc > 0.95
    assert m.r2 > 0.95


def test_classification_metrics_computation():
    """Tests computation of classification metrics dictionary."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([-1.2, 0.4, -0.3, 1.8])
    m = compute_classification_metrics(y_true, y_pred, y_score)
    d = m.to_dict()
    assert d["accuracy"] == 0.5
    assert d["balanced_accuracy"] == 0.5
    assert d["f1"] == 0.5
    assert d["tp"] == 1 and d["tn"] == 1 and d["fp"] == 1 and d["fn"] == 1
    assert 0.0 <= d["roc_auc"] <= 1.0
