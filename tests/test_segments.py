import pandas as pd
import pytest

from src.segments import add_run_id


def _frame(rows):
    return pd.DataFrame(rows, columns = ["subject", "exercise", "restimulus", "rerepetition"])


def test_add_run_id_groups_contiguous_matching_rows():
    frame = _frame(
        [
            [1, 2, 0, 0],
            [1, 2, 0, 0],
            [1, 2, 5, 1],
            [1, 2, 5, 1],
            [1, 2, 5, 1],
            [1, 2, 0, 1],  # new run: key changed, even though restimulus repeats an earlier value
        ]
    )
    result = add_run_id(frame)

    # cumsum() over a boolean "is this a new run" series is 1-indexed by
    # construction (row 0 is always a "new run" relative to nothing before
    # it). This matches the same pattern already used in eda.ipynb, so the
    # fix here is the test's expected values, not the source.
    assert result["run_id"].tolist() == [1, 1, 2, 2, 2, 3]


def test_add_run_id_does_not_merge_non_contiguous_matching_rows():
    # Rows 0 and 2 share identical key values but are not adjacent.
    # They must NOT be merged into the same run_id.
    frame = _frame(
        [
            [1, 2, 5, 1],
            [1, 2, 6, 1],
            [1, 2, 5, 1],
        ]
    )
    result = add_run_id(frame)

    assert result["run_id"].tolist() == [1, 2, 3]


def test_add_run_id_raises_on_missing_key_column():
    frame = pd.DataFrame({"subject": [1], "exercise": [2]})
    with pytest.raises(ValueError, match = "Missing required run-key"):
        add_run_id(frame)
