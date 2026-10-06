"""Train and evaluate the LDA classical baseline.

Usage:
    python scripts/train_lda_baseline.py --features-dir data/processed/features --out reports

Fits and compares TWO prior settings directly in one run:
  - empirical priors (scikit-learn's default): class priors estimated
    from training-data frequency, which bakes rest's ~59% share into
    the decision rule before features are even considered
  - uniform priors: every class treated as equally likely a priori,
    letting the features do the discriminating

This comparison exists because the first real run of this script showed
validation accuracy (62.6%) far outpacing macro-F1 (11.0%) - a signature
of majority-class bias, not genuine multi-class separability. Reported
side by side rather than silently switching the default, so the fix is
demonstrated against real numbers, not just asserted.

The uniform-priors model is saved as the baseline going forward
(models/lda_baseline.joblib); the empirical-priors run is kept in the
results JSON for comparison only, not saved as a separate model file.

Evaluation design otherwise unchanged from the original: trained on the
18 train subjects, evaluated on the 3 validation subjects, plus 6-fold
GroupKFold CV within train (3 held-out subjects per fold, matching
validation's size). Test and holdout are never touched here.
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

import joblib
import numpy as np
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.baselines import per_subject_accuracy, select_feature_columns, subject_group_cv_scores  # noqa: E402


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_split(features_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    data = np.load(features_dir / f"{split_name}.npz")
    return {key: data[key] for key in data.files}


def evaluate_priors_setting(
    *,
    priors,
    priors_label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subject_val: np.ndarray,
    cv_folds: int,
) -> dict:
    print(f"\n--- {priors_label} priors ---")
    t0 = time.time()
    model = LinearDiscriminantAnalysis(solver="svd", priors=priors)
    model.fit(X_train, y_train)
    print(f"Fit in {time.time() - t0:.1f}s")

    val_predictions = model.predict(X_val)
    val_accuracy = float(np.mean(y_val == val_predictions))
    val_macro_f1 = float(f1_score(y_val, val_predictions, average="macro"))
    val_per_subject = per_subject_accuracy(y_val, val_predictions, subject_val)

    print(f"Validation accuracy:  {val_accuracy:.4f}")
    print(f"Validation macro-F1:  {val_macro_f1:.4f}")
    print("Validation accuracy per subject:")
    for subject_id, accuracy in val_per_subject.items():
        print(f"  subject {subject_id}: {accuracy:.4f}")

    print(f"Running {cv_folds}-fold subject-grouped CV within train...")
    t0 = time.time()
    cv = subject_group_cv_scores(
        lambda: LinearDiscriminantAnalysis(solver="svd", priors=priors),
        X_train, y_train, subject_train,
        n_splits=cv_folds,
    )
    print(f"CV done in {time.time() - t0:.1f}s")
    print(f"CV fold accuracies: {[round(s, 4) for s in cv['accuracy']]}")
    print(f"CV fold macro-F1:   {[round(s, 4) for s in cv['macro_f1']]}")
    print(f"CV mean accuracy: {np.mean(cv['accuracy']):.4f}  CV mean macro-F1: {np.mean(cv['macro_f1']):.4f}")

    return {
        "model": model,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "validation_accuracy_per_subject": val_per_subject,
        "cv_folds": cv_folds,
        "cv_fold_accuracies": cv["accuracy"],
        "cv_fold_macro_f1": cv["macro_f1"],
        "cv_mean_accuracy": float(np.mean(cv["accuracy"])),
        "cv_mean_macro_f1": float(np.mean(cv["macro_f1"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--cv-folds", type=int, default=6)
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
    print(f"  train:      X={train['X'].shape}")
    print(f"  validation: X={validation['X'].shape}")

    X_train_full = train["X"][:, column_mask]
    X_val_full = validation["X"][:, column_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_full)
    X_val = scaler.transform(X_val_full)

    uniform_priors = np.full(n_classes, 1.0 / n_classes)

    empirical_result = evaluate_priors_setting(
        priors=None, priors_label="Empirical (default)",
        X_train=X_train, y_train=train["y"], subject_train=train["subject"],
        X_val=X_val, y_val=validation["y"], subject_val=validation["subject"],
        cv_folds=args.cv_folds,
    )
    uniform_result = evaluate_priors_setting(
        priors=uniform_priors, priors_label="Uniform",
        X_train=X_train, y_train=train["y"], subject_train=train["subject"],
        X_val=X_val, y_val=validation["y"], subject_val=validation["subject"],
        cv_folds=args.cv_folds,
    )

    print("\n--- Comparison ---")
    print(f"{'Metric':<28} {'Empirical':>12} {'Uniform':>12}")
    print(f"{'Validation accuracy':<28} {empirical_result['validation_accuracy']:>12.4f} {uniform_result['validation_accuracy']:>12.4f}")
    print(f"{'Validation macro-F1':<28} {empirical_result['validation_macro_f1']:>12.4f} {uniform_result['validation_macro_f1']:>12.4f}")
    print(f"{'CV mean accuracy':<28} {empirical_result['cv_mean_accuracy']:>12.4f} {uniform_result['cv_mean_accuracy']:>12.4f}")
    print(f"{'CV mean macro-F1':<28} {empirical_result['cv_mean_macro_f1']:>12.4f} {uniform_result['cv_mean_macro_f1']:>12.4f}")

    # The uniform-priors model is the one that becomes THE baseline going
    # forward - saved as a model file. The empirical-priors run is kept
    # only as a documented comparison point in the results JSON below.
    model_dir = REPO_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": uniform_result["model"], "scaler": scaler, "feature_columns": kept_columns, "priors": "uniform"},
        model_dir / "lda_baseline.joblib",
    )
    print(f"\nSaved uniform-priors model (the baseline going forward) to {model_dir / 'lda_baseline.joblib'}")

    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "LDA (solver=svd)",
        "feature_columns_used": kept_columns,
        "feature_columns_excluded": [c for c in feature_columns if c not in kept_columns],
        "empirical_priors": {k: v for k, v in empirical_result.items() if k != "model"},
        "uniform_priors": {k: v for k, v in uniform_result.items() if k != "model"},
        "saved_model_priors": "uniform",
    }
    results_path = REPO_ROOT / args.out / "lda_baseline_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
