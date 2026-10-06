"""
Build feature matrices from windowed data, split by subject.

Usage:
    python scripts/build_features.py --windows-dir data/processed/full --out data/processed/features
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import yaml

from src.features import build_feature_matrix, check_signal_is_rectified  # noqa: E402
from src.labels import build_label_encoding, save_label_encoding  # noqa: E402
from src.split import SPLIT_NAMES, validate_subject_split  # noqa: E402

def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--windows-dir", type = Path, required = True)
    parser.add_argument("--out", type = Path, required = True)
    args = parser.parse_args()

    config = load_config()
    subject_split = config["subject_split"]
    feature_cfg = config.get("features", {})
    zc_threshold = feature_cfg.get("zc_threshold", 0.01)
    ssc_threshold = feature_cfg.get("ssc_threshold", 0.01)

    windows_dir = REPO_ROOT / args.windows_dir
    print(f"Loading windows from {windows_dir}")
    t0 = time.time()
    metadata = pd.read_parquet(windows_dir / "windows_metadata.parquet")
    emg = np.load(windows_dir / "windows_emg.npy")
    print(f"Loaded {len(metadata):,} windows in {time.time() - t0:.1f}s")

    if len(metadata) != emg.shape[0]:
        raise ValueError(
            f"Metadata has {len(metadata):,} rows but the EMG array has "
            f"{emg.shape[0]:,} - files are supposed to be row-aligned."
        )

    is_rectified = check_signal_is_rectified(emg)
    print(f"\nSignal is rectified (no negative values anywhere): {is_rectified}")
    if is_rectified:
        print(
            "  -> ZC is structurally degenerate on this data. Still computed and "
            "saved for a complete, auditable record, treat as a non-feature at "
            "modeling time, not a working one."
        )

    print("\nExtracting features...")
    t0 = time.time()
    n_channels = emg.shape[2]
    if n_channels != 10:
        raise ValueError(f"Expected 10 EMG channels, got {n_channels}.")
    channel_labels = [f"emg_{i}" for i in range(n_channels)]
    feature_matrix, feature_columns = build_feature_matrix(
        emg, channel_labels, zc_threshold = zc_threshold, ssc_threshold = ssc_threshold
    )
    print(f"Built {feature_matrix.shape} feature matrix in {time.time() - t0:.1f}s")

    print("\nBuilding label encoding...")
    label_encoding = build_label_encoding(set(metadata["composite_label"].unique()))
    print(f"{len(label_encoding)} classes")
    y_all = metadata["composite_label"].map(label_encoding).to_numpy()

    args.out.mkdir(parents = True, exist_ok = True)
    save_label_encoding(label_encoding, REPO_ROOT / args.out / "label_encoding.json")

    # NEW: save the feature column names alongside the matrices, so any
    # downstream code reads column identity from the data itself instead
    # of silently hardcoding "columns 0-9 are MAV, 10-19 are RMS...".
    # That assumption was previously implicit and unverified anywhere.
    with open(REPO_ROOT / args.out / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent = 2)
    print(f"Saved feature_columns.json ({len(feature_columns)} columns)")

    all_subjects = subject_split["subject_ids"]
    split_groups = {name: subject_split[name] for name in SPLIT_NAMES}
    validate_subject_split(split_groups, expected_subject_ids = all_subjects)

    print("\nSplitting and saving:")
    subject_col = metadata["subject"].to_numpy()
    window_index_col = metadata["window_index"].to_numpy()
    for split_name in SPLIT_NAMES:
        subjects_in_split = set(subject_split[split_name])
        mask = np.isin(subject_col, list(subjects_in_split))
        split_path = REPO_ROOT / args.out / f"{split_name}.npz"
        np.savez(
            split_path,
            X = feature_matrix[mask],
            y = y_all[mask],
            subject = subject_col[mask],
            window_index = window_index_col[mask],
        )
        print(f"  {split_name:<10} {mask.sum():>9,} windows  -> {split_path}")

    print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")

if __name__ == "__main__":
    main()