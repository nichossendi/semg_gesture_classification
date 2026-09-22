"""
Run the windowing pipeline against the real Ninapro DB1 csv.

Usage:
    # Smoke test on a couple of subjects first. Do this before the full run.
    python scripts/run_windowing.py --subjects 1 2 --out data/processed/smoke_test

    # Full run, all subjects listed in config.yaml's subject_split.subject_ids
    python scripts/run_windowing.py --out data/processed/full

Note on the smoke test: load_ninapro_csv() reads the entire csv before any
exercise/subject filtering happens, so a "--subjects 1 2" run still pays
the full ~3.3GB read cost. There is no cheap way around that without
changing the loader itself, which is out of scope here. What the smoke
test actually buys you is testing windowing correctness and per-subject
timing on real data cheaply, before committing to building ~1.5-2M windows
across all 27 subjects.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import yaml

from src.data import load_ninapro_csv
from src.windows import WindowConfig, make_windows


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def save_windows(windows: pd.DataFrame, out_dir: Path) -> None:
    """
    Split windowed output into a metadata table and a stacked EMG array.

    The `emg` column holds a (window_samples, n_channels) array per row,
    which doesn't serialize to parquet as a clean, fixed shape type.
    Forcing it in either explodes to hundreds of flat columns or needs a
    nested list type that plain pandas.read_parquet doesn't hand back as a
    usable array automatically. Splitting into a lightweight, queryable
    metadata parquet table plus one contiguous .npy array, joined by row
    position via `window_index`, is the more standard pattern for this
    shape of data.
    """

    out_dir.mkdir(parents = True, exist_ok = True)

    emg_stack = np.stack(windows["emg"].to_numpy())  # (n_windows, window_samples, n_channels)
    np.save(out_dir / "windows_emg.npy", emg_stack)

    metadata = windows.drop(columns = ["emg"]).reset_index(drop = True)
    metadata["window_index"] = metadata.index  # position into windows_emg.npy, row for row
    metadata.to_parquet(out_dir / "windows_metadata.parquet", index = False)

    print(f"\nSaved {len(metadata):,} windows:")
    print(f"  {out_dir / 'windows_emg.npy'}  shape = {emg_stack.shape}  dtype = {emg_stack.dtype}")
    print(f"  {out_dir / 'windows_metadata.parquet'}  {len(metadata.columns)} columns")


def summarize(windows: pd.DataFrame, org_rest_pct: float = 58.53) -> None:
    print(f"\nTotal windows: {len(windows):,}")

    is_rest = windows["restimulus"] == 0
    rest_pct = is_rest.mean() * 100
    print(f"Rest-labeled windows:   {is_rest.sum():,} ({rest_pct:.2f}%)")
    print(f"Active-labeled windows: {(~is_rest).sum():,} ({100 - rest_pct:.2f}%)")
    print(
        f"(Original row-level rest share was {org_rest_pct:.2f}%. Expect this in "
        "the same neighborhood, not identical. Windowing changes the unit of "
        "count from rows to windows, and trailing partial windows are dropped.)"
    )

    n_labels = windows["composite_label"].nunique()
    print(f"\nDistinct composite labels present: {n_labels} (expect 41 once all subjects are included)")

    # Broken down by segment type, not blended. A combined "% of all
    # windows above threshold" figure conflates two very different things.
    # Rest windows that look surprisingly active (the actual anomaly
    # worth tracking, comparable to the original 7.40%-of-rest-runs figure) and
    # active windows correctly showing high amplitude (expected, not an
    # anomaly at all). A blended number isn't directly comparable to
    # anything already in the manifest.

    rest_above = windows.loc[is_rest, "peak_abs_amplitude_above_threshold"]
    active_above = windows.loc[~is_rest, "peak_abs_amplitude_above_threshold"]
    print(f"\nRest windows above threshold:   {rest_above.mean() * 100:.2f}% (Original rest-run-level figure: 7.40%)")
    print(f"Active windows above threshold: {active_above.mean() * 100:.2f}%")

    print("\nWindows per subject:")
    print(windows.groupby("subject").size().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--subjects", type = int, nargs = "+", default = None,
        help = "Subject IDs to process. Default: all IDs in config.yaml's subject_split.subject_ids.",
    )
    parser.add_argument("--out", type = Path, required = True, help = "Output directory under data/processed/")
    args = parser.parse_args()

    config = load_config()
    subjects = args.subjects or config["subject_split"]["subject_ids"]
    win_cfg = config["windowing"]

    print(f"Processing subjects: {subjects}")
    t0 = time.time()
    frame, _report = load_ninapro_csv(REPO_ROOT / "data" / "raw" / "Ninapro_DB1.csv", exercises = (2, 3))
    frame = frame[frame["subject"].isin(subjects)].reset_index(drop = True)
    print(f"Loaded and filtered to {len(frame):,} rows in {time.time() - t0:.1f}s")

    emg_columns = [column for column in frame.columns if column.startswith("emg_")]
    window_config = WindowConfig(
        window_samples = win_cfg["window_samples"],
        step_samples = win_cfg["step_samples"],
        rest_peak_threshold = win_cfg["rest_peak_threshold"],
    )

    t0 = time.time()
    windows = make_windows(frame, emg_columns, window_config)
    print(f"Built {len(windows):,} windows in {time.time() - t0:.1f}s")

    summarize(windows)
    save_windows(windows, REPO_ROOT / args.out)


if __name__ == "__main__":
    main()
