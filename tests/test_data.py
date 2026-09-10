import pandas as pd
import pytest

from src import load_ninapro_csv


def _synthetic_db1_csv(path) -> None:
    emg = {f"emg_{index}": [float(index)] * 4 for index in range(10)}
    pd.DataFrame(
        {
            "Unnamed: 0": [10, 11, 12, 13],
            "subject": [1, 1, 2, 3],
            "exercise": [1, 2, 3, 3],
            "stimulus": [1, 1, 1, 2],
            "restimulus": [1, 1, 1, 2],
            "repetition": [1, 1, 2, 2],
            "rerepetition": [1, 1, 2, 2],
            **emg,
        }
    ).to_csv(path, index=False)


def test_loader_drops_index_filters_exercises_and_reports_dataset(tmp_path) -> None:
    csv_path = tmp_path / "synthetic_db1.csv"
    _synthetic_db1_csv(csv_path)

    frame, report = load_ninapro_csv(csv_path, exercises=(2, 3))

    assert frame["exercise"].tolist() == [2, 3, 3]
    assert "Unnamed: 0" not in frame.columns
    assert report["file_size_bytes"] > 0
    assert report["row_count_before_filtering"] == 4
    assert report["row_count_after_filtering"] == 3
    assert report["unique_subject_ids"] == [1, 2, 3]
    assert report["unique_exercise_ids"] == [1, 2, 3]
    assert report["repetition_counts"] == {1: 2, 2: 2}
    assert report["has_ten_emg_columns"] is True
    assert report["restimulus_label_scope"]["labels_globally_unique_across_exercises"] is False
    assert report["restimulus_label_scope"]["recommended_model_label_key"] == "(exercise, restimulus)"


def test_loader_raises_a_clear_error_for_missing_required_columns(tmp_path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame({"subject": [1], "exercise": [2]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required column\\(s\\):"):
        load_ninapro_csv(csv_path)
