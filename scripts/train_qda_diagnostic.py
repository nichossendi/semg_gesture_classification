"""
QDA diagnostic, isolates whether LDA's shared covariance assumption
is part of why the LDA baseline's macro-F1 stayed low (~0.11) under both
empirical and uniform priors.

LDA assumes every class shares one covariance matrix. QDA drops that
assumption. Each class gets its own covariance estimate, while keeping
everything else about the comparison identical: same features (same 40
non-ZC columns), same scaler-fit-on-train discipline, same train/
validation split, same subject-grouped CV protocol, and the same
empirical-vs-uniform priors comparison already used for LDA. That holds
every other variable fixed and changes exactly one thing, so the result
tells you whether shared covariance specifically was the problem:

  - QDA meaningfully better than LDA  -> shared covariance was a real
    part of the problem (plausible on physical grounds too: rest's
    distribution, tight and near-zero, has little reason to share a
    covariance structure with the wider-spread active classes)
  - QDA similarly poor to LDA         -> the ceiling is coming from
    somewhere else, most likely the features themselves (MAV/RMS/WL/SSC)
    not carrying enough linearly or quadratically usable signal for most
    of these 41 classes, which would make CNN (learning its
    own features from raw signal) load-bearing, not just a comparison
    point.

QDA fits one covariance matrix PER CLASS (40x40 here), which needs
reasonably large per-class sample counts to estimate reliably. This
script catches and reports sklearn's own collinearity warnings
explicitly. If a class's covariance comes out singular, that is a
numerical-degeneracy problem specific to that class, a different finding
from "quadratic boundaries don't help", and the two should not be
conflated when reading the result.

Usage:
    python scripts/train_qda_diagnostic.py --features-dir data/processed/features --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.baselines import per_subject_accuracy, select_feature_columns, subject_group_cv_scores  # noqa: E402

def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def load_split(features_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    data = np.load(features_dir / f"{split_name}.npz")
    return {key: data[key] for key in data.files}

def evaluate_qda_variant(
    *,
    priors,
    priors_label: str,
    reg_param: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subject_val: np.ndarray,
    cv_folds: int,
) -> dict:
    print(f"\n--- QDA, {priors_label} priors (reg_param = {reg_param}) ---")

    def make_model():
        return QuadraticDiscriminantAnalysis(priors = priors, reg_param = reg_param)

    collinearity_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model = make_model()
        try:
            model.fit(X_train, y_train)
        except ValueError as error:
            # sklearn's QDA raises a hard ValueError (not a warning) when a
            # class has too few samples to even define a covariance (needs
            # at least 2 points per class). Caught here rather than left to
            # crash with a raw traceback, found via an explicit test of
            # this exact failure mode (a deliberately 1-sample class),
            # which my original collinearity-warning catch did NOT handle,
            # since this is a harder failure than a warning. Not expected
            # on the real dataset (train's thinnest active class has
            # thousands of samples, see the per-class count check printed
            # above), but this is the difference between "crashes with no
            # explanation" and "fails with a clear, actionable message" if
            # it ever is hit, e.g. on a future smaller or re-scoped run.
            print(f"  FAILED TO FIT: {error}")
            print(
                "  -> At least one class has too few samples for QDA to "
                "define a covariance at all (needs >= 2 per class). Check "
                "the thin-class counts printed above. This result is "
                "unusable, fix the data (more samples for that class, or "
                "exclude it) before trusting any number from this run."
            )
            raise
        fit_seconds = time.time() - t0
        collinearity_warnings.extend(str(w.message) for w in caught if "collinear" in str(w.message).lower())

    print(f"Fit in {fit_seconds:.1f}s")
    if collinearity_warnings:
        print(f"  WARNING: {len(collinearity_warnings)} collinearity warning(s) from sklearn during fit:")
        for message in collinearity_warnings[:5]:
            print(f"    {message}")
        print(
            "  -> At least one class's covariance matrix is singular/near-singular. "
            "A poor result here may reflect this numerical issue for specific "
            "classes, not a general failure of quadratic boundaries, check "
            "per-class accuracy for the affected classes before concluding either way."
        )
    else:
        print("  No collinearity warnings, no class's covariance estimate degenerated during this fit.")

    val_predictions = model.predict(X_val)
    val_accuracy = float(np.mean(y_val == val_predictions))
    val_macro_f1 = float(f1_score(y_val, val_predictions, average = "macro"))
    val_per_subject = per_subject_accuracy(y_val, val_predictions, subject_val)

    print(f"Validation accuracy:  {val_accuracy:.4f}")
    print(f"Validation macro-F1:  {val_macro_f1:.4f}")
    print("Validation accuracy per subject:")
    for subject_id, accuracy in val_per_subject.items():
        print(f"  subject {subject_id}: {accuracy:.4f}")

    print(f"Running {cv_folds}-fold subject-grouped CV within train...")
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught_cv:
        warnings.simplefilter("always")
        cv = subject_group_cv_scores(make_model, X_train, y_train, subject_train, n_splits = cv_folds)
        cv_collinearity_count = sum(1 for w in caught_cv if "collinear" in str(w.message).lower())
    print(f"CV done in {time.time() - t0:.1f}s")
    if cv_collinearity_count:
        print(f"  WARNING: {cv_collinearity_count} collinearity warning(s) across CV folds.")
    print(f"CV fold accuracies: {[round(s, 4) for s in cv['accuracy']]}")
    print(f"CV fold macro-F1:   {[round(s, 4) for s in cv['macro_f1']]}")
    print(f"CV mean accuracy: {np.mean(cv['accuracy']):.4f}  CV mean macro-F1: {np.mean(cv['macro_f1']):.4f}")

    return {
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "validation_accuracy_per_subject": val_per_subject,
        "fit_collinearity_warnings": collinearity_warnings,
        "cv_folds": cv_folds,
        "cv_fold_accuracies": cv["accuracy"],
        "cv_fold_macro_f1": cv["macro_f1"],
        "cv_mean_accuracy": float(np.mean(cv["accuracy"])),
        "cv_mean_macro_f1": float(np.mean(cv["macro_f1"])),
        "cv_collinearity_warning_count": cv_collinearity_count,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--features-dir", type = Path, default = Path("data/processed/features"))
    parser.add_argument("--out", type = Path, default = Path("reports"))
    parser.add_argument("--cv-folds", type = int, default = 6)
    parser.add_argument(
        "--reg-param", type = float, default = 0.0,
        help = "QDA covariance shrinkage toward a pooled estimate (0 = none, 1 = fully pooled i.e. LDA like). "
             "Only raise this from 0.0 if collinearity warnings appear. It trades away QDA's per-class "
             "flexibility, which is the exact thing this diagnostic is trying to test.",
    )
    args = parser.parse_args()

    features_dir = REPO_ROOT / args.features_dir
    with open(features_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(features_dir / "label_encoding.json") as f:
        label_encoding = json.load(f)
    n_classes = len(label_encoding)

    column_mask, kept_columns = select_feature_columns(feature_columns)
    print(f"Using {len(kept_columns)} of {len(feature_columns)} feature columns (excluded: zc_*)")

    print("\nLoading train/validation splits...")
    train = load_split(features_dir, "train")
    validation = load_split(features_dir, "validation")
    print(f"  train:      X = {train['X'].shape}")
    print(f"  validation: X = {validation['X'].shape}")

    # Sample count sanity check: QDA fits one (n_features x n_features)
    # covariance per class, so each class needs comfortably more samples
    # than there are features to estimate that reliably. Printed up front
    # so a thin class is known about before results are read, not
    # discovered after the fact from a collinearity warning alone.
    n_features = int(column_mask.sum())
    train_class_counts = {int(k): int(v) for k, v in zip(*np.unique(train["y"], return_counts = True))}
    thin_classes = {label_encoding_name: count for label_encoding_name, count in train_class_counts.items() if count < 5 * n_features}
    print(f"\n{n_features} features; smallest training class has {min(train_class_counts.values()):,} samples "
          f"(rule of thumb: want >> {n_features} per class for a stable per-class covariance estimate)")
    if thin_classes:
        print(f"  Classes with < {5 * n_features} samples (5x feature count, a conservative margin): {thin_classes}")

    X_train_full = train["X"][:, column_mask]
    X_val_full = validation["X"][:, column_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_full)
    X_val = scaler.transform(X_val_full)

    uniform_priors = np.full(n_classes, 1.0 / n_classes)

    empirical_result = evaluate_qda_variant(
        priors = None, priors_label = "Empirical (default)", reg_param = args.reg_param,
        X_train = X_train, y_train = train["y"], subject_train = train["subject"],
        X_val = X_val, y_val = validation["y"], subject_val = validation["subject"],
        cv_folds = args.cv_folds,
    )
    uniform_result = evaluate_qda_variant(
        priors = uniform_priors, priors_label = "Uniform", reg_param = args.reg_param,
        X_train = X_train, y_train = train["y"], subject_train = train["subject"],
        X_val = X_val, y_val = validation["y"], subject_val = validation["subject"],
        cv_folds = args.cv_folds,
    )

    print("\n--- QDA empirical vs. uniform ---")
    print(f"{'Metric':<28} {'Empirical':>12} {'Uniform':>12}")
    print(f"{'Validation accuracy':<28} {empirical_result['validation_accuracy']:>12.4f} {uniform_result['validation_accuracy']:>12.4f}")
    print(f"{'Validation macro-F1':<28} {empirical_result['validation_macro_f1']:>12.4f} {uniform_result['validation_macro_f1']:>12.4f}")
    print(f"{'CV mean accuracy':<28} {empirical_result['cv_mean_accuracy']:>12.4f} {uniform_result['cv_mean_accuracy']:>12.4f}")
    print(f"{'CV mean macro-F1':<28} {empirical_result['cv_mean_macro_f1']:>12.4f} {uniform_result['cv_mean_macro_f1']:>12.4f}")

    # Pull in the already saved LDA results for a direct side-by-side, if
    # present, never hardcoded, so this always reflects whatever LDA run
    # actually produced, not a remembered number that could go stale.
    lda_results_path = REPO_ROOT / args.out / "lda_baseline_results.json"
    if lda_results_path.exists():
        with open(lda_results_path) as f:
            lda_results = json.load(f)
        print("\n--- LDA vs. QDA (macro-F1, the metric that matters here) ---")
        print(f"{'Setting':<28} {'LDA':>12} {'QDA':>12}")
        print(f"{'Empirical priors':<28} {lda_results['empirical_priors']['validation_macro_f1']:>12.4f} {empirical_result['validation_macro_f1']:>12.4f}")
        print(f"{'Uniform priors':<28} {lda_results['uniform_priors']['validation_macro_f1']:>12.4f} {uniform_result['validation_macro_f1']:>12.4f}")
    else:
        print(f"\n(No {lda_results_path} found, run train_lda_baseline.py to get a direct LDA comparison alongside this.)")

    args.out.mkdir(parents = True, exist_ok = True)
    results = {
        "model": "QDA",
        "reg_param": args.reg_param,
        "feature_columns_used": kept_columns,
        "feature_columns_excluded": [c for c in feature_columns if c not in kept_columns],
        "train_class_counts": train_class_counts,
        "thin_classes_under_5x_feature_count": thin_classes,
        "empirical_priors": empirical_result,
        "uniform_priors": uniform_result,
    }
    results_path = REPO_ROOT / args.out / "qda_diagnostic_results.json"
    results_path.write_text(json.dumps(results, indent = 2))
    print(f"\nSaved results to {results_path}")

if __name__ == "__main__":
    main()