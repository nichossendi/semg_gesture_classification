"""Loading and first-pass validation for Ninapro DB1 csv files."""

from __future__ import annotations

import re
from collections.abc import Iterable
from numbers import Number
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = frozenset(
    {"subject", "exercise", "stimulus", "restimulus", "repetition", "rerepetition"}
)
EMG_COLUMN_PATTERN = re.compile(r"^emg(?:[_-]?\d+)?$", flags=re.IGNORECASE)


def _sort_key(value: object) -> tuple[int, float | str]:
    """Sort numeric values numerically and all other values by text."""
    if isinstance(value, Number) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value))


def _sorted_unique(values: pd.Series) -> list[object]:
    """Return non-null unique values in a stable, numeric-aware order."""
    return sorted(values.dropna().unique().tolist(), key=_sort_key)


def _value_counts(values: pd.Series) -> dict[object, int]:
    """Return non-null value counts with stable, numeric-aware key order."""
    counts = values.dropna().value_counts().to_dict()
    return dict(sorted(counts.items(), key=lambda item: _sort_key(item[0])))


def _find_emg_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if EMG_COLUMN_PATTERN.fullmatch(column)]


def _label_report(frame: pd.DataFrame, label_column: str) -> dict[str, object]:
    """Describe whether non-rest labels are unique across selected exercises."""
    non_rest = frame.loc[frame[label_column] != 0, ["exercise", label_column]].dropna()
    exercises_by_label = non_rest.groupby(label_column)["exercise"].unique()
    collisions = {
        label: sorted(exercises.tolist())
        for label, exercises in exercises_by_label.items()
        if len(exercises) > 1
    }
    globally_unique = not collisions
    return {
        "labels_globally_unique_across_exercises": globally_unique,
        "labels_reused_across_exercises": collisions,
        "recommended_model_label_key": label_column
        if globally_unique
        else f"(exercise, {label_column})",
    }


def load_ninapro_csv(
    path: str | Path,
    *,
    exercises: tuple[int, ...] = (2, 3),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a Ninapro csv file, validate its schema, filter exercises, and report its contents.

    The report is calculated before filtering except for label-collision analysis,
    which applies only to the selected exercises.
    """
    if not exercises:
        raise ValueError("At least one exercise identifier must be supplied.")

    csv_path = Path(path)
    frame = pd.read_csv(csv_path).drop(columns=["Unnamed: 0"], errors="ignore")

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        expected = ", ".join(sorted(REQUIRED_COLUMNS))
        found = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"{csv_path} is not a supported Ninapro DB1 CSV. "
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected: {expected}. Found: {found}."
        )

    emg_columns = _find_emg_columns(frame.columns)
    filtered = frame.loc[frame["exercise"].isin(exercises)].copy()
    report: dict[str, object] = {
        "file_path": str(csv_path),
        "file_size_bytes": csv_path.stat().st_size,
        "row_count_before_filtering": len(frame),
        "row_count_after_filtering": len(filtered),
        "selected_exercises": list(exercises),
        "columns": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_value_counts": {column: int(count) for column, count in frame.isna().sum().items()},
        "unique_subject_ids": _sorted_unique(frame["subject"]),
        "unique_exercise_ids": _sorted_unique(frame["exercise"]),
        "unique_stimulus_values": _sorted_unique(frame["stimulus"]),
        "unique_restimulus_values": _sorted_unique(frame["restimulus"]),
        "repetition_counts": _value_counts(frame["repetition"]),
        "rerepetition_counts": _value_counts(frame["rerepetition"]),
        "emg_columns": emg_columns,
        "has_ten_emg_columns": len(emg_columns) == 10,
        "emg_column_count": len(emg_columns),
        "restimulus_label_scope": _label_report(filtered, "restimulus"),
    }
    return filtered, report

