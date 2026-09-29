"""
Per-class boxplots and summary stats for MAV and RMS.

Sanity checks whether classes look separable before training
anything on the feature matrices. Real data only, nothing here is
meaningful run against synthetic stand-ins.

Usage:
    python scripts/plot_feature_distributions.py --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless-safe. This script may run without a display
import numpy as np
import matplotlib.pyplot as plt

from src.features import FEATURE_NAMES  # noqa: E402, single source of truth for column order,
# not a duplicated local constant that could silently drift from src/features.py's actual layout

N_CHANNELS = 10

def _feature_block(X: np.ndarray, feature_name: str) -> np.ndarray:
    """
    Return the (n_windows, n_channels) block for one feature.

    Relies on build_feature_matrix's documented column layout: FEATURE_NAMES
    order, each block N_CHANNELS wide, channel-ordered emg_0..emg_9.
    """

    index = FEATURE_NAMES.index(feature_name)
    start = index * N_CHANNELS
    return X[:, start : start + N_CHANNELS]

def print_summary(X: np.ndarray, y: np.ndarray, label_names: dict[int, str], feature_name: str) -> None:
    per_window_value = _feature_block(X, feature_name).mean(axis = 1)  # aggregate across channels

    rest_index = next(index for index, name in label_names.items() if name == "rest")
    rest_median = float(np.median(per_window_value[y == rest_index]))
    active_medians = [
        float(np.median(per_window_value[y == index])) for index in label_names if index != rest_index
    ]

    print(f"\n{feature_name.upper()} summary (aggregated across all 10 channels per window):")
    print(f"  rest median:          {rest_median:.4f}")
    print(f"  active median range:  {min(active_medians):.4f} to {max(active_medians):.4f}")
    print(f"  active median mean:   {np.mean(active_medians):.4f}")
    if rest_median < min(active_medians):
        print("  -> rest median sits below every active class's median (expected direction).")
    else:
        print("  -> rest median is NOT below every active class's median, worth a closer look before trusting separability.")

def plot_feature_by_class(
    X: np.ndarray,
    y: np.ndarray,
    label_names: dict[int, str],
    feature_name: str,
    out_path: Path,
) -> None:
    """
    Boxplot of one feature (averaged across all 10 channels per window),
    grouped by class, classes ordered exactly as label_encoding.json orders
    them (rest first, then numerically by (exercise, restimulus)), not
    re-sorted alphabetically.

    Outlier points are hidden (showfliers = False) for readability across 41
    classes on one axis. This is a different question from the peak-based
    rest-transient screening: that measured the single peak sample within
    a window. This measures MAV/RMS, an average over the whole 20-sample
    window. A brief spike has a much smaller effect on an averaged feature
    than on a peak-based one. Hiding boxplot outliers here does not hide or
    contradict that earlier finding. They are answering different questions
    about the same data.
    """

    per_window_value = _feature_block(X, feature_name).mean(axis = 1)

    ordered_indices = sorted(label_names.keys())
    data_by_class = [per_window_value[y == index] for index in ordered_indices]
    labels = [label_names[index] for index in ordered_indices]

    fig, ax = plt.subplots(figsize = (18, 6))
    ax.boxplot(data_by_class, tick_labels = labels, showfliers = False)
    ax.set_title(f"{feature_name.upper()} by class (mean across 10 channels per window, outlier points hidden)")
    ax.set_ylabel(feature_name.upper())
    ax.set_xlabel("Class")
    plt.setp(ax.get_xticklabels(), rotation = 90, fontsize = 7)

    if "rest" in labels:
        rest_position = labels.index("rest") + 1  # matplotlib boxplot positions are 1-indexed
        ax.axvline(rest_position + 0.5, color = "red", linestyle = "--", alpha = 0.4, linewidth = 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print(f"Saved {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--split", default = "train", choices = ["train", "validation", "test", "holdout"])
    parser.add_argument("--features-dir", type = Path, default = Path("data/processed/features"))
    args = parser.parse_args()

    features_dir = REPO_ROOT / args.features_dir
    data = np.load(features_dir / f"{args.split}.npz")
    X, y = data["X"], data["y"]

    with open(features_dir / "label_encoding.json") as f:
        label_to_index = json.load(f)
    label_names = {index: name for name, index in label_to_index.items()}

    print(f"Loaded {args.split}: X = {X.shape}, y = {y.shape}, {len(label_names)} classes")

    figures_dir = REPO_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents = True, exist_ok = True)

    for feature_name in ("mav", "rms"):
        print_summary(X, y, label_names, feature_name)
        plot_feature_by_class(X, y, label_names, feature_name, figures_dir / f"{feature_name}_by_class_{args.split}.png")

if __name__ == "__main__":
    main()