"""
Classical time-domain sEMG feature extraction.

Computes MAV, RMS, WL, ZC, and SSC per channel, for every window at once
via vectorized numpy operations, not a per-window Python loop, which
would be genuinely slow at ~1.89M windows (unlike src.windows'
per-run loop, which only iterates ~21,000 times and was fine as is).

Before trusting ZC, run check_signal_is_rectified() against the real
data. See its docstring for why this matters.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("mav", "rms", "wl", "zc", "ssc")

def check_signal_is_rectified(emg: np.ndarray) -> bool:
    """
    Return True if `emg` contains no negative values anywhere.

    The raw-trace figures showed both rest and active example segments
    staying entirely non-negative, a real signal this csv mirror's EMG
    channels may already be rectified (an envelope/magnitude signal), not
    raw bipolar EMG. That distinction matters concretely: zero-crossing
    count (ZC) is a real, informative feature on bipolar EMG, but on an
    always non-negative signal, it is structurally degenerate. It will
    return exactly 0 for every window, regardless of threshold, because a
    signal that never goes negative can never cross zero. This function
    exists so that assumption gets checked against real data before ZC is
    trusted, rather than silently shipping a dead feature column.

    SSC is NOT affected by this. It measures direction reversals in the
    signal's slope, which stays meaningful whether or not the signal is
    rectified.
    """
    return not bool(np.any(emg < 0))

def extract_features(
    emg: np.ndarray,
    *,
    zc_threshold: float,
    ssc_threshold: float,
) -> dict[str, np.ndarray]:
    """
    Compute MAV, RMS, WL, ZC, SSC per channel for every window at once.

    Parameters
    ----------
    emg : ndarray, shape (n_windows, window_samples, n_channels)
    zc_threshold, ssc_threshold : minimum magnitude a crossing/reversal
        must clear to be counted, filtering quantization-level noise
        rather than every microscopic wiggle. See FEATURES.md for how
        these defaults were chosen and why they need revisiting once
        run against the real noise floor.

    Returns
    -------
    dict mapping each name in FEATURE_NAMES to an (n_windows, n_channels)
    array. Concatenating them, in FEATURE_NAMES order, is the caller's
    job (see build_feature_matrix), so the column order is explicit and
    documented in one place rather than implicit in this function.

    Raises
    ------
    ValueError
        If `emg` is not 3-dimensional, or window_samples < 3 (too short
        for waveform length/slope based features to be meaningful).
    """

    if emg.ndim != 3:
        raise ValueError(f"Expected EMG with shape (n_windows, window_samples, n_channels), got ndim = {emg.ndim}")
    if emg.shape[1] < 3:
        raise ValueError(
            f"window_samples = {emg.shape[1]} is too short for WL/ZC/SSC "
            "(need at least 3 samples per window)."
        )

    mav = np.mean(np.abs(emg), axis = 1)
    rms = np.sqrt(np.mean(emg**2, axis = 1))

    diffs = np.diff(emg, axis = 1)  # (n_windows, window_samples-1, n_channels)
    wl = np.sum(np.abs(diffs), axis = 1)

    # Zero crossings: consecutive samples with opposite sign, big enough
    # to not be quantization noise. Degenerate (always 0) on a rectified
    # signal, see check_signal_is_rectified.
    sign_change = (emg[:, :-1, :] * emg[:, 1:, :]) < 0
    zc_big_enough = np.abs(diffs) >= zc_threshold
    zc = np.sum(sign_change & zc_big_enough, axis = 1)

    # Slope sign changes: consecutive slopes pointing in opposite
    # directions, again filtered against quantization noise. Stays
    # meaningful on a rectified signal, since it only looks at direction
    # reversals in the envelope, not at whether the signal itself
    # crosses zero.
    slope_before = diffs[:, :-1, :]
    slope_after = diffs[:, 1:, :]
    slope_reversal = (slope_before * slope_after) < 0
    slope_big_enough = (np.abs(slope_before) >= ssc_threshold) | (np.abs(slope_after) >= ssc_threshold)
    ssc = np.sum(slope_reversal & slope_big_enough, axis = 1)

    return {"mav": mav, "rms": rms, "wl": wl, "zc": zc, "ssc": ssc}

def build_feature_matrix(
    emg: np.ndarray,
    emg_columns: list[str],
    *,
    zc_threshold: float,
    ssc_threshold: float,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features and lay them out as one flat (n_windows, n_features) matrix.

    Column order is fixed and explicit: for each name in FEATURE_NAMES,
    all channels in `emg_columns` order. Returns the matrix alongside its
    column names, so the two can never silently drift apart.
    """

    features = extract_features(emg, zc_threshold = zc_threshold, ssc_threshold = ssc_threshold)

    columns: list[str] = []
    blocks: list[np.ndarray] = []
    for name in FEATURE_NAMES:
        blocks.append(features[name])  # (n_windows, n_channels)
        columns.extend(f"{name}_{channel}" for channel in emg_columns)

    matrix = np.concatenate(blocks, axis = 1)  # (n_windows, len(FEATURE_NAMES) * n_channels)
    return matrix, columns