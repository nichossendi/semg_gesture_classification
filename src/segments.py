"""
Contiguous-run identification for Ninapro-style labeled time series.

A "run" is a maximal contiguous block of rows sharing the same
(subject, exercise, restimulus, rerepetition) key. This is the same
construction used in the EDA notebook (notebooks/eda.ipynb),
extracted here as tested, reusable code, since windowing depends
on windows never crossing a run boundary.
"""

from __future__ import annotations

import pandas as pd

RUN_KEY_COLUMNS: tuple[str, ...] = ("subject", "exercise", "restimulus", "rerepetition")


def add_run_id(
    frame: pd.DataFrame,
    *,
    run_key_columns: tuple[str, ...] = RUN_KEY_COLUMNS,
) -> pd.DataFrame:
    """
    Return a copy of `frame` with an added `run_id` column.

    Two rows share a run_id only if they are adjacent in `frame` (by row
    order, after an internal reset_index) AND agree on every column in
    `run_key_columns`. This is a row-order dependent operation. `frame`
    must already be in original recording order, not re-sorted. If rows
    have been reordered, run boundaries will be computed incorrectly
    without any error being raised, since there is no independent way to
    detect that from the data alone.

    Raises
    ------
    ValueError
        If any of `run_key_columns` is missing from `frame`.
    """
    missing = [column for column in run_key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required run-key column(s): {missing}")

    frame = frame.reset_index(drop = True).copy()
    keys = frame[list(run_key_columns)]
    is_new_run = keys.ne(keys.shift()).any(axis = 1)
    frame["run_id"] = is_new_run.cumsum()
    return frame
