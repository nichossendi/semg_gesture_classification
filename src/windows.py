"""
Fixed length windowing of contiguous-run respecting sEMG segments.

Every window is drawn from exactly one run (see src.segments.add_run_id)
and never crosses a run boundary, so a window can never mix samples from
two different subjects, exercises, gestures, or repetitions. See
PREPROCESSING.md for the reasoning behind the window size, step size,
label resolution, trailing window, and rest-transient decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.segments import RUN_KEY_COLUMNS, add_run_id

DEFAULT_REST_PEAK_THRESHOLD = 1.4  # see DATASET_MANIFEST.md rest-peak screening


@dataclass(frozen = True)
class WindowConfig:
    """
    Windowing parameters, in samples (not seconds/ms) to avoid rounding
    ambiguity at the sampling rate boundary.
    """

    window_samples: int
    step_samples: int
    rest_peak_threshold: float = DEFAULT_REST_PEAK_THRESHOLD

    def __post_init__(self) -> None:
        if self.window_samples <= 0 or self.step_samples <= 0:
            raise ValueError("window_samples and step_samples must both be positive.")
        if self.step_samples > self.window_samples:
            raise ValueError(
                f"step_samples ({self.step_samples}) exceeds window_samples "
                f"({self.window_samples}); this would silently skip samples "
                "between consecutive windows."
            )


def _composite_label(exercise: int, restimulus: int) -> str:
    """
    Build a window's class label, unifying rest across exercises.

    Matches notebooks/eda.ipynb's `eda_plot_label` construction and
    DATASET_MANIFEST.md's documented scope (Exercise B + Exercise C + one
    unified rest class = 41 total, not 42). Rest (restimulus == 0)
    collapses to a single "rest" label regardless of which exercise it
    came from. Without this, a rest window from Exercise 2 and a rest
    window from Exercise 3 would get two different labels ("2_0" vs
    "3_0"), silently reintroducing the 42-class space that EDA
    specifically resolved down to 41, which is exactly what an earlier
    version of this function did, caught by a real 2 subject smoke test
    producing 42 distinct labels instead of the documented 41.
    """

    if restimulus == 0:
        return "rest"
    return f"{exercise}_{restimulus}"


def make_windows(
    frame: pd.DataFrame,
    emg_columns: list[str],
    config: WindowConfig,
) -> pd.DataFrame:
    """
    Slide fixed length windows over `frame`, one output row per window.

    `frame` must already be filtered to the desired exercises and be in
    original recording row order (see add_run_id's docstring for why row
    order matters). Row positions reported in the output
    (`frame_row_start` / `frame_row_end`) are positions within `frame` as
    passed to this call, not positions in any earlier, unfiltered version
    of the data. Callers that need to trace back further should keep
    their own record of how `frame` was filtered from the original load.

    Trailing partial windows (fewer than `config.window_samples` rows left
    at the end of a run) are dropped, not padded.

    Every window, active or rest, gets a peak-amplitude value and an
    above-threshold flag. No rows are dropped based on it. That keeps
    preprocessing non-destructive: whether/how to use the flag is a
    modeling time decision, not one made silently here.

    Raises
    ------
    ValueError
        If required columns are missing, or if any window's restimulus
        values are not unanimous (see _resolve_window_label, this signals
        a bug in run construction, not real labeling ambiguity).
    RuntimeError
        If no windows could be produced at all.
    """

    missing = set(RUN_KEY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    missing_emg = [column for column in emg_columns if column not in frame.columns]
    if missing_emg:
        raise ValueError(f"Missing EMG column(s): {missing_emg}")

    frame = add_run_id(frame)  # also resets frame's row order to 0..len(frame)-1
    windows: list[dict[str, object]] = []

    for run_id, run in frame.groupby("run_id", sort = False):
        # Capture positions within `frame` BEFORE resetting the run's own
        # local index. Reversing this order silently produces local
        # (per-run) positions instead of frame-level ones.
        
        frame_positions = run.index.to_numpy()
        run = run.reset_index(drop = True)
        n = len(run)
        if n < config.window_samples:
            continue

        emg_values = run[emg_columns].to_numpy(dtype = np.float64)
        restimulus_values = run["restimulus"].to_numpy()
        subject = int(run["subject"].iloc[0])
        exercise = int(run["exercise"].iloc[0])

        start = 0
        while start + config.window_samples <= n:
            end = start + config.window_samples
            window_label = _resolve_window_label(restimulus_values[start:end])
            window_emg = emg_values[start:end]
            peak_amplitude = float(np.abs(window_emg).max())

            windows.append(
                {
                    "subject": subject,
                    "exercise": exercise,
                    "restimulus": window_label,
                    "composite_label": _composite_label(exercise, window_label),
                    "run_id": int(run_id),
                    "frame_row_start": int(frame_positions[start]),
                    "frame_row_end": int(frame_positions[end - 1]),
                    "peak_abs_amplitude": peak_amplitude,
                    "peak_abs_amplitude_above_threshold": peak_amplitude >= config.rest_peak_threshold,
                    "emg": window_emg,
                }
            )
            start += config.step_samples

    if not windows:
        raise RuntimeError(
            "No windows produced. Check that window_samples is not larger "
            "than every run's length, and that `frame` is not empty."
        )

    return pd.DataFrame(windows)


def _resolve_window_label(restimulus_window: np.ndarray) -> int:
    """
    Return the single restimulus value shared by every row in a window.

    A window is only ever built from rows within one run, and a run is
    defined, in part, by a single constant restimulus value. So, every row
    in a correctly-constructed window must already agree. This function
    does not vote to resolve genuine ambiguity, because none should exist
    here. If it ever finds more than one distinct value, that means a
    window crossed a run boundary somewhere upstream, which is a bug in
    windowing logic, not a labeling edge case to paper over silently.
    """

    unique_values = np.unique(restimulus_window)
    if len(unique_values) != 1:
        raise ValueError(
            f"Window contains {len(unique_values)} distinct restimulus "
            f"values {unique_values.tolist()}. This should be impossible "
            "for a window confined to a single run. This indicates a bug "
            "in run boundary computation (src.segments.add_run_id), not a "
            "real labeling ambiguity. Investigate before trusting any "
            "windowed output produced alongside this error."
        )
    return int(unique_values[0])
