import numpy as np
import pytest

from src.features import (
    FEATURE_NAMES,
    build_feature_matrix,
    check_signal_is_rectified,
    extract_features,
)

def test_feature_formulas_match_hand_computed_values():
    # x = [0, 2, -1, 3, 3, -2], see review for the full hand derivation.
    x = np.array([0, 2, -1, 3, 3, -2], dtype = float)
    emg = x.reshape(1, 6, 1)

    result = extract_features(emg, zc_threshold = 0.5, ssc_threshold = 0.5)

    assert np.isclose(result["mav"][0, 0], 11 / 6)
    assert np.isclose(result["rms"][0, 0], np.sqrt(27 / 6))
    assert np.isclose(result["wl"][0, 0], 14.0)
    assert result["zc"][0, 0] == 3
    assert result["ssc"][0, 0] == 2

def test_zc_is_structurally_zero_on_a_rectified_signal():
    # A signal that never goes negative can never cross zero. ZC must
    # come back all-zero here regardless of threshold, by construction.
    x = np.array([0.1, 0.9, 0.2, 0.8, 0.0, 0.7], dtype = float)
    emg = x.reshape(1, 6, 1)

    result = extract_features(emg, zc_threshold = 0.0, ssc_threshold = 0.0)

    assert result["zc"][0, 0] == 0
    # SSC is NOT degenerate the same way. This same signal has real
    # direction reversals (up, down, up, down, up), so it must be > 0.
    assert result["ssc"][0, 0] > 0

def test_check_signal_is_rectified():
    assert check_signal_is_rectified(np.array([[[0.1, 0.2, 0.0]]])) is True
    assert check_signal_is_rectified(np.array([[[0.1, -0.2, 0.0]]])) is False

def test_extract_features_shapes_for_multiple_windows_and_channels():
    rng = np.random.default_rng(0)
    emg = rng.normal(size = (5, 20, 10))  # 5 windows, 20 samples, 10 channels

    result = extract_features(emg, zc_threshold = 0.01, ssc_threshold = 0.01)

    for name in FEATURE_NAMES:
        assert result[name].shape == (5, 10)

def test_extract_features_rejects_too_short_windows():
    emg = np.zeros((2, 2, 3))  # window_samples = 2, below the minimum of 3
    with pytest.raises(ValueError, match = "too short"):
        extract_features(emg, zc_threshold = 0.0, ssc_threshold = 0.0)

def test_extract_features_rejects_wrong_ndim():
    emg = np.zeros((5, 20))  # missing the channel axis
    with pytest.raises(ValueError, match = "ndim"):
        extract_features(emg, zc_threshold = 0.0, ssc_threshold = 0.0)

def test_build_feature_matrix_column_order_and_shape():
    rng = np.random.default_rng(1)
    emg = rng.normal(size = (4, 20, 3))
    emg_columns = ["emg_0", "emg_1", "emg_2"]

    matrix, columns = build_feature_matrix(emg, emg_columns, zc_threshold = 0.01, ssc_threshold = 0.01)

    assert matrix.shape == (4, len(FEATURE_NAMES) * 3)
    assert columns == [
        "mav_emg_0", "mav_emg_1", "mav_emg_2",
        "rms_emg_0", "rms_emg_1", "rms_emg_2",
        "wl_emg_0", "wl_emg_1", "wl_emg_2",
        "zc_emg_0", "zc_emg_1", "zc_emg_2",
        "ssc_emg_0", "ssc_emg_1", "ssc_emg_2",
    ]
    # Spot-check that a specific column actually holds the feature it
    # claims to, not just that the shape/naming looks right.
    mav_col_index = columns.index("mav_emg_1")
    expected_mav_channel1 = np.mean(np.abs(emg[:, :, 1]), axis = 1)
    assert np.allclose(matrix[:, mav_col_index], expected_mav_channel1)