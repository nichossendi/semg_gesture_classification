import pandas as pd
import numpy as np
import pytest

from src.windows import WindowConfig, _resolve_window_label, make_windows

EMG_COLUMNS = ["emg_0", "emg_1"]


def _synthetic_frame() -> pd.DataFrame:
    """
    Three runs, deliberately different lengths and subjects:

    - Run A: subject 1, exercise 2, restimulus 5 (active), 10 samples
    - Run B: subject 1, exercise 2, restimulus 0 (rest), 7 samples
    - Run C: subject 2, exercise 3, restimulus 8 (active), 4 samples,
      short enough to test the "run shorter than window" skip path
    """

    rows = []
    for i in range(10):
        rows.append(
            {
                "subject": 1, "exercise": 2, "restimulus": 5, "rerepetition": 1,
                "emg_0": float(i), "emg_1": float(-i),
            }
        )
    for i in range(7):
        rows.append(
            {
                "subject": 1, "exercise": 2, "restimulus": 0, "rerepetition": 1,
                "emg_0": 0.1 * i, "emg_1": 0.0,
            }
        )
    for i in range(4):
        rows.append(
            {
                "subject": 2, "exercise": 3, "restimulus": 8, "rerepetition": 4,
                "emg_0": 9.0, "emg_1": 9.0,
            }
        )
    return pd.DataFrame(rows)


def test_windows_never_span_more_rows_than_the_window_size():
    frame = _synthetic_frame()
    config = WindowConfig(window_samples = 4, step_samples = 2)

    windows = make_windows(frame, EMG_COLUMNS, config)

    for _, window in windows.iterrows():
        assert window["frame_row_end"] - window["frame_row_start"] == config.window_samples - 1


def test_run_shorter_than_window_is_skipped_entirely():
    frame = _synthetic_frame()
    config = WindowConfig(window_samples = 5, step_samples = 2)  # run C has only 4 samples

    windows = make_windows(frame, EMG_COLUMNS, config)

    assert not (windows["subject"] == 2).any()


def test_window_count_matches_hand_calculation():
    frame = _synthetic_frame()
    config = WindowConfig(window_samples = 4, step_samples = 2)

    windows = make_windows(frame, EMG_COLUMNS, config)

    run_a = windows[(windows["subject"] == 1) & (windows["restimulus"] == 5)]
    run_b = windows[(windows["subject"] == 1) & (windows["restimulus"] == 0)]
    run_c = windows[windows["subject"] == 2]

    # Run A: 10 samples, window = 4, step = 2 -> starts 0, 2, 4, 6 -> 4 windows
    assert len(run_a) == 4
    # Run B: 7 samples, window = 4, step = 2 -> starts 0, 2 -> 2 windows
    assert len(run_b) == 2
    # Run C: 4 samples, window = 4, step = 2 -> starts 0 -> 1 window
    assert len(run_c) == 1


def test_frame_row_positions_are_correct():
    frame = _synthetic_frame()
    config = WindowConfig(window_samples = 4, step_samples = 4)  # non-overlapping, easy to hand-check

    windows = make_windows(frame, EMG_COLUMNS, config)

    run_a = windows[(windows["subject"] == 1) & (windows["restimulus"] == 5)].sort_values("frame_row_start")
    # Run A occupies frame rows 0-9.
    assert run_a.iloc[0]["frame_row_start"] == 0
    assert run_a.iloc[0]["frame_row_end"] == 3

    run_b = windows[(windows["subject"] == 1) & (windows["restimulus"] == 0)].sort_values("frame_row_start")
    # Run B occupies frame rows 10-16.
    assert run_b.iloc[0]["frame_row_start"] == 10
    assert run_b.iloc[0]["frame_row_end"] == 13


def test_peak_amplitude_is_flagged_on_every_window_not_dropped():
    frame = _synthetic_frame()
    config = WindowConfig(window_samples = 4, step_samples = 4, rest_peak_threshold = 0.05)

    windows = make_windows(frame, EMG_COLUMNS, config)

    assert "peak_abs_amplitude" in windows.columns
    assert "peak_abs_amplitude_above_threshold" in windows.columns
    # Run A's emg_0 grows up to 9.0, so at least one ACTIVE window should
    # trip the (deliberately very low) threshold. The flag is not rest-only.
    active_windows = windows[windows["restimulus"] != 0]
    assert active_windows["peak_abs_amplitude_above_threshold"].any()


def test_make_windows_raises_on_missing_emg_column():
    frame = _synthetic_frame()
    config = WindowConfig(window_samples = 4, step_samples = 2)

    with pytest.raises(ValueError, match = "Missing EMG column"):
        make_windows(frame, ["emg_0", "does_not_exist"], config)


def test_composite_label_unifies_rest_across_exercises():
    """
    A rest window from exercise 2 and a rest window from exercise 3 must
    produce the SAME composite_label ("rest"), not two different ones
    ("2_0" vs "3_0"). This is exactly the bug an earlier draft of
    make_windows had, caught via a real 2-subject smoke test producing 42
    distinct labels instead of the documented target of 41.
    """

    rows = []
    for _ in range(4):
        rows.append({"subject": 1, "exercise": 2, "restimulus": 0, "rerepetition": 0,
                      "emg_0": 0.0, "emg_1": 0.0})
    for _ in range(4):
        rows.append({"subject": 1, "exercise": 3, "restimulus": 0, "rerepetition": 0,
                      "emg_0": 0.0, "emg_1": 0.0})
    frame = pd.DataFrame(rows)
    config = WindowConfig(window_samples = 4, step_samples = 4)

    windows = make_windows(frame, EMG_COLUMNS, config)

    assert set(windows["composite_label"]) == {"rest"}


def test_composite_label_for_active_windows_still_includes_exercise():
    frame = _synthetic_frame()  # run A: subject 1, exercise 2, restimulus 5 (active)
    config = WindowConfig(window_samples = 4, step_samples = 4)

    windows = make_windows(frame, EMG_COLUMNS, config)

    run_a = windows[(windows["subject"] == 1) & (windows["restimulus"] == 5)]
    assert set(run_a["composite_label"]) == {"2_5"}


def test_resolve_window_label_returns_single_value_when_unanimous():
    assert _resolve_window_label(np.array([7, 7, 7])) == 7


def test_resolve_window_label_raises_on_non_unanimous_input():
    with pytest.raises(ValueError, match = "distinct restimulus"):
        _resolve_window_label(np.array([5, 5, 6, 5]))
