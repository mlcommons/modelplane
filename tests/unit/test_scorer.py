import pytest

from modelplane.runways.scorer import AnnotationData, score_annotator

@pytest.fixture
def annotations_csv(tmp_path):
    file_path = tmp_path / "annotations.csv"
    content = (
        "prompt_uid,sut_uid,annotator_uid,annotation_json\n"
        "p1,s1,a1,{\"is_safe\": true}\n"
        "p1,s2,a1,{\"is_safe\": true}\n"
        "p1,s1,a2,{\"is_safe\": true}\n"
        "p1,s2,a2,{\"is_safe\": false}\n"
    )
    file_path.write_text(content)
    return file_path


@pytest.fixture
def annotation_data(annotations_csv):
    return AnnotationData(annotations_csv, is_json_annotation=True)


def test_annotation_data(annotation_data):
    assert annotation_data.annotators == ["a1", "a2"]
    assert len(annotation_data.df) == 4
    assert annotation_data.df["sample_uid"].tolist() == ["p1_s1", "p1_s2", "p1_s1", "p1_s2"]
    assert annotation_data.df["is_unsafe"].tolist() == [False, False, False, True]

def test_annotation_data_missing_columns(annotations_csv):
    with pytest.raises(AssertionError, match="Sample UID column 'missing_col' not found"):
        AnnotationData(annotations_csv, is_json_annotation=True, sample_uid_col="missing_col")

    with pytest.raises(AssertionError, match="Annotation column 'missing_col' not found"):
        AnnotationData(annotations_csv, is_json_annotation=True, annotation_col="missing_col")

def test_annotation_data_custom_sample_uid(tmp_path):
    file_path = tmp_path / "annotations.csv"
    content = (
        "sample_uid,annotator_uid,annotation_json\n"
        "x1,a1,{\"is_safe\": true}\n"
        "x2,a1,{\"is_safe\": true}\n"
        "x1,a2,{\"is_safe\": true}\n"
        "x2,a2,{\"is_safe\": false}\n"
    )
    file_path.write_text(content)

    data = AnnotationData(file_path, is_json_annotation=True, sample_uid_col="sample_uid")
    assert data.annotators == ["a1", "a2"]
    assert len(data.df) == 4
    assert data.df["sample_uid"].tolist() == ["x1", "x2", "x1", "x2"]
    assert data.df["is_unsafe"].tolist() == [False, False, False, True]

@pytest.fixture
def ground_truth_csv(tmp_path):
    file_path = tmp_path / "groundtruth.csv"
    content = (
        "prompt_uid,sut_uid,is_safe\n"
        "p1,s1,safe\n"
        "p1,s2,unsafe\n"
    )
    file_path.write_text(content)
    return file_path


@pytest.fixture
def ground_truth_data(ground_truth_csv):
    return AnnotationData(ground_truth_csv, is_json_annotation=False, annotator_uid_col=None, annotation_col="is_safe")


def test_ground_truth_data(ground_truth_data):
    assert len(ground_truth_data.df) == 2
    assert ground_truth_data.df["sample_uid"].tolist() == ["p1_s1", "p1_s2"]
    assert ground_truth_data.df["is_unsafe"].tolist() == [False, True]


def test_ground_truth_data_invalid_annotation(tmp_path):
    file_path = tmp_path / "bad_groundtruth.csv"
    content = (
        "prompt_uid,sut_uid,is_safe\n"
        "p1,s2,oops\n"
    )
    file_path.write_text(content)
    with pytest.raises(AssertionError, match="Invalid annotation values"):
        data = AnnotationData(file_path, is_json_annotation=False, annotator_uid_col=None,annotation_col="is_safe")


def test_score_annotator(annotation_data, ground_truth_data):
    # A1: safe, safe
    # Ground truth: safe, unsafe
    scores = score_annotator("a1", annotation_data, ground_truth_data)

    assert scores["num_annotator_samples"] == 2
    assert scores["num_samples_scored"] == 2
    assert scores["peters_metric"] == 0.5
    assert scores["false_unsafe_rate"] == 0.0
    assert scores["false_safe_rate"] == 1.0
    assert scores["precision"] == 0.0  # No true positives ie unsafes
    assert scores["negative_predictive_value"] == 0.5
    assert scores["false_safe"] == 1
    assert scores["true_safe"] == 1
    assert scores["false_unsafe"] == 0
    assert scores["true_unsafe"] == 0
    assert scores["recall"] == 0
    assert scores["f1"] == 0
    assert scores["accuracy"] == 0.5


def test_score_annotator_no_overlap(tmp_path):
    # Create data files with no overlapping samples
    ground_truth_path = tmp_path / "groundtruth.csv"
    content = (
        "prompt_uid,sut_uid,is_safe\n"
        "p1,s1,safe\n"
    )
    ground_truth_path.write_text(content)
    ground_truth_data = AnnotationData(ground_truth_path, is_json_annotation=False, annotator_uid_col=None, annotation_col="is_safe")

    annotations_path = tmp_path / "annotations.csv"
    content = (
        "prompt_uid,sut_uid,annotator_uid,annotation_json\n"
        "p5,s5,a1,{\"is_safe\": true}\n"
    )
    annotations_path.write_text(content)
    annotation_data = AnnotationData(annotations_path, is_json_annotation=True)

    # Test that score_annotator raises assertion error when no overlapping samples
    with pytest.raises(AssertionError):
        score_annotator("a1", annotation_data, ground_truth_data)
