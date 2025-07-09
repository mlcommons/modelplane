import pytest
import pandas as pd
import json
import os
from pathlib import Path

from modelplane.runways.scorer import (
    transform_mlflow_annotator_artifact,
    ground_truth_to_df,
    format_df,
    score_annotator
)

ANNOTATIONS_FILE = "tests/data/annotations.csv"
GROUND_TRUTH_FILE = "tests/data/ground_truth.csv"

def test_transform_mlflow_annotator_artifact():
    annotators, df = transform_mlflow_annotator_artifact(ANNOTATIONS_FILE)
    assert annotators == ["test_annotator"]
    assert "is_safe" in df.columns
    assert "is_unsafe" in df.columns
    assert "sample_id" in df.columns
    assert len(df) == 10  # Number of rows in test data
    # Check that is_safe was properly extracted from JSON
    first_row = df.iloc[0]
    assert first_row["is_safe"] == True
    assert first_row["is_unsafe"] == False
    assert first_row["sample_id"] == "0_demo_yes_no"

def test_ground_truth_to_df():
    df = ground_truth_to_df(GROUND_TRUTH_FILE)
    assert "is_safe" in df.columns
    assert "is_unsafe" in df.columns
    assert "sample_id" in df.columns
    assert len(df) == 10  # Number of rows in test data
    # Check that string "safe" was converted to boolean
    assert df["is_safe"].dtype == bool
    assert df["is_unsafe"].dtype == bool

def test_format_df():
    # Create a test dataframe
    test_df = pd.DataFrame({
        "prompt_uid": ["1", "2"],
        "sut_uid": ["a", "b"],
        "is_safe": [True, False]
    })
    
    formatted_df = format_df(test_df, ["is_safe"])
    assert "sample_id" in formatted_df.columns
    assert "is_unsafe" in formatted_df.columns
    assert formatted_df["sample_id"].iloc[0] == "1_a"
    assert formatted_df["is_unsafe"].iloc[0] == False
    assert formatted_df["is_unsafe"].iloc[1] == True

def test_format_df_missing_columns():
    # Test that format_df raises assertion error when required columns are missing
    test_df = pd.DataFrame({
        "prompt_uid": ["1", "2"]  # Missing sut_uid
    })
    
    with pytest.raises(AssertionError):
        format_df(test_df, ["is_safe"])

def test_score_annotator():
    # Load test data
    _, annotations_df = transform_mlflow_annotator_artifact(ANNOTATIONS_FILE)
    ground_truth_df = ground_truth_to_df(GROUND_TRUTH_FILE)
    
    # Score test_annotator
    scores = score_annotator("test_annotator", annotations_df, ground_truth_df)
    
    # Check that all expected metrics are present
    expected_metrics = [
        "num_annotator_samples",
        "num_samples_scored",
        "peters_metric",
        "false_safe_rate",
        "false_unsafe_rate",
        "precision",
        "negative_predictive_value",
        "false_safe",
        "true_safe",
        "false_unsafe",
        "true_unsafe",
        "recall",
        "f1",
        "accuracy"
    ]
    
    for metric in expected_metrics:
        assert metric in scores
    
    # Check basic metric properties
    # Ground truth is all safe aka all negative. Annotations are half safe and half unsafe.
    assert scores["num_annotator_samples"] == 10
    assert scores["num_samples_scored"] == 10
    assert scores["accuracy"] == 0.5
    assert scores["false_unsafe_rate"] == 0.5
    assert scores["precision"] == 0.0 # No true positives
    assert scores["false_safe"] == 0
    assert scores["true_safe"] == 5
    assert scores["false_unsafe"] == 5

def test_score_annotator_no_overlap():
    # Create test dataframes with no overlapping samples
    annotations_df = pd.DataFrame({
        "sample_id": ["1_a", "2_b"],
        "prompt_uid": ["1", "2"],
        "sut_uid": ["a", "b"],
        "annotator_uid": ["test_annotator", "test_annotator"],
        "is_safe": [True, False],
        "is_unsafe": [False, True]
    })
    
    ground_truth_df = pd.DataFrame({
        "sample_id": ["3_c", "4_d"],
        "prompt_uid": ["3", "4"],
        "sut_uid": ["c", "d"],
        "is_safe": [True, True],
        "is_unsafe": [False, False]
    })
    
    # Test that score_annotator raises assertion error when no overlapping samples
    with pytest.raises(AssertionError):
        score_annotator("test_annotator", annotations_df, ground_truth_df)



