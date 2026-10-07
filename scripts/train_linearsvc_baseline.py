"""
Train and evaluate the LinearSVC classical baselines.

Usage:
    python scripts/train_linearsvc_baseline.py --features-dir data/processed/features --out reports

Why this model, now: LDA and QDA are both generative, Gaussian-likelihood
methods. They model each class's feature distribution and classify by
which distribution a point is more likely to have come from. Both hit the
same macro-F1 ceiling (~0.10-0.12) under every prior/covariance variant
tried previously. LinearSVC is structurally different in the one way
that matters here, it is discriminative, not generative. It never
estimates a per-class distribution or inverts a covariance matrix, it
just finds a max-margin linear boundary directly. Everything else about
the comparison is held identical, same 40 non-ZC feature columns, same
scaler-fit-on-train discipline, same train/validation split, same
6-fold subject-grouped CV protocol.

  - LinearSVC meaningfully beats LDA/QDA  -> the generative/covariance-
    based approach itself was the limitation, not the features, a
    genuinely different and more optimistic finding than previously
    suggested.
  - LinearSVC also caps out around macro-F1 ~0.10-0.12 -> three
    structurally different classical approaches (shared-covariance
    generative, per-class-covariance generative, margin-based
    discriminative) all hit the same wall. That is about as strong a
    case as classical ML can make that the limitation is the hand-crafted
    features (MAV/RMS/WL/SSC) themselves, not the model family, which
    makes the CNN (learning its own features from the raw signal) load-bearing,
    not just a nice comparison.

Class-weight comparison, parallel to the previous priors comparison:
  - unweighted (default): every training window counted equally, so
    rest's ~59% share dominates the loss the same way empirical priors
    let it dominate LDA's decision rule
  - class_weight = "balanced": per-class weights set inversely proportional
    to class frequency. The discriminative-method analogue of uniform
    priors

The previous result is the reason this script does NOT default to saving
whichever variant nudges macro-F1 up: uniform priors paid a ~22-point
accuracy cost for a macro-F1 gain smaller than CV fold-to-fold noise, and
that was reversed once the full picture was in. Same discipline here,
both variants are fit, reported side by side, and the unweighted model is
saved as the baseline unless the balanced variant's macro-F1 gain clearly
survives CV noise AND does not cost accuracy disproportionately. The
script does not decide this silently, it prints both and states which it
saved and why.

Implementation notes:
  - dual = False: scikit-learn's own guidance is to prefer the primal
    formulation when n_samples > n_features, which is drastically true
    here (~1.26M training windows vs 40 features). Requires
    loss = "squared_hinge" (the default), which is compatible.
  - max_iter raised from sklearn's default (1000) and liblinear's
    ConvergenceWarning is caught and reported explicitly (not left to
    print a raw warning to stderr that's easy to miss in a long run),
    a result from a non-converged fit is a different finding from a
    genuinely poor max-margin fit, and the two should not be conflated
    when reading the macro-F1 numbers.
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

import joblib
import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.baselines import per_subject_accuracy, select_feature_columns, subject_group_cv_scores  # noqa: E402

def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def load_split(features_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    data = np.load(features_dir / f"{split_name}.npz")
    return {key: data[key] for key in data.files}

def evaluate_svc_variant(
    *,
    class_weight,
    class_weight_label: str,
    C: float,
    max_iter: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subject_val: np.ndarray,
    cv_folds: int,
) -> dict:
    print(f"\n--- LinearSVC, {class_weight_label} class weights (C = {C}, max_iter = {max_iter}) ---")

    def make_model():
        return LinearSVC(C = C, class_weight = class_weight, dual = False, max_iter = max_iter)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model = make_model()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - t0
        convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]

    print(f"Fit in {fit_seconds:.1f}s")
    if convergence_warnings:
        print(f"  WARNING: did not converge within max_iter = {max_iter}:")
        for message in convergence_warnings[:3]:
            print(f"    {message}")
        print(
            "  -> Results below come from an unconverged fit, a poor macro-F1 here "
            "may just mean 'needs more iterations', not 'max-margin boundaries do not "
            "help'. Re-run with a higher --max-iter before trusting this number."
        )
    else:
        print(f"  Converged within max_iter = {max_iter}.")

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
    with warnings.catch_warnings(record = True) as caught_cv:
        warnings.simplefilter("always")
        cv = subject_group_cv_scores(make_model, X_train, y_train, subject_train, n_splits = cv_folds)
        cv_convergence_count = sum(1 for w in caught_cv if issubclass(w.category, ConvergenceWarning))
    print(f"CV done in {time.time() - t0:.1f}s")
    if cv_convergence_count:
        print(f"  WARNING: {cv_convergence_count} fold(s)/fits did not converge within max_iter = {max_iter}.")
    print(f"CV fold accuracies: {[round(s, 4) for s in cv['accuracy']]}")
    print(f"CV fold macro-F1:   {[round(s, 4) for s in cv['macro_f1']]}")
    print(f"CV mean accuracy: {np.mean(cv['accuracy']):.4f}  CV mean macro-F1: {np.mean(cv['macro_f1']):.4f}")

    return {
        "model": model,
        "fit_seconds": fit_seconds,
        "fit_convergence_warnings": convergence_warnings,
        "cv_convergence_warning_count": cv_convergence_count,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "validation_accuracy_per_subject": val_per_subject,
        "cv_folds": cv_folds,
        "cv_fold_accuracies": cv["accuracy"],
        "cv_fold_macro_f1": cv["macro_f1"],
        "cv_mean_accuracy": float(np.mean(cv["accuracy"])),
        "cv_mean_macro_f1": float(np.mean(cv["macro_f1"])),
    }

def print_classical_comparison(out_dir: Path, linearsvc_unweighted_macro_f1: float, linearsvc_balanced_macro_f1: float) -> None:
    """
    Pulls in LDA and QDA results if present, never hardcoded, always
    reflects whatever those scripts actually produced, not a remembered
    number that could go stale.
    """
    lda_path = out_dir / "lda_baseline_results.json"
    qda_path = out_dir / "qda_diagnostic_results.json"
    if not lda_path.exists() and not qda_path.exists():
        print(f"\n(No {lda_path.name} or {qda_path.name} found in {out_dir}, skipping cross-model comparison.)")
        return

    print("\n--- Classical baselines so far (validation macro-F1, the metric that matters here) ---")
    rows: list[tuple[str, float | None]] = []
    if lda_path.exists():
        with open(lda_path) as f:
            lda = json.load(f)
        rows.append(("LDA, empirical priors", lda["empirical_priors"]["validation_macro_f1"]))
        rows.append(("LDA, uniform priors", lda["uniform_priors"]["validation_macro_f1"]))
    if qda_path.exists():
        with open(qda_path) as f:
            qda = json.load(f)
        rows.append(("QDA, empirical priors", qda["empirical_priors"]["validation_macro_f1"]))
        rows.append(("QDA, uniform priors", qda["uniform_priors"]["validation_macro_f1"]))
    rows.append(("LinearSVC, unweighted", linearsvc_unweighted_macro_f1))
    rows.append(("LinearSVC, balanced", linearsvc_balanced_macro_f1))

    for label, value in rows:
        print(f"  {label:<28} {value:.4f}")

    values = [v for _, v in rows]
    spread = max(values) - min(values)
    print(f"\n  Spread across all generative + discriminative variants tried: {spread:.4f}")
    if spread < 0.05:
        print(
            "  -> All three model families (LDA, QDA, LinearSVC) land within a narrow "
            "band regardless of prior/weight handling. That is a strong case the ceiling "
            "is the hand-crafted features, not the model family, see MODELING.md."
        )

def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--features-dir", type=Path, default = Path("data/processed/features"))
    parser.add_argument("--out", type = Path, default = Path("reports"))
    parser.add_argument("--cv-folds", type = int, default = 6)
    parser.add_argument("--C", type = float, default = 1.0, help = "LinearSVC regularization strength (no tuning done here, a baseline, not a tuned model).")
    parser.add_argument("--max-iter", type = int, default = 5000)
    args = parser.parse_args()

    features_dir = REPO_ROOT / args.features_dir
    with open(features_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(features_dir / "label_encoding.json") as f:
        label_encoding = json.load(f)

    column_mask, kept_columns = select_feature_columns(feature_columns)
    print(f"Using {len(kept_columns)} of {len(feature_columns)} feature columns (excluded: zc_*)")

    print("\nLoading train/validation splits...")
    train = load_split(features_dir, "train")
    validation = load_split(features_dir, "validation")
    print(f"  train:      X = {train['X'].shape}")
    print(f"  validation: X = {validation['X'].shape}")

    X_train_full = train["X"][:, column_mask]
    X_val_full = validation["X"][:, column_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_full)
    X_val = scaler.transform(X_val_full)

    unweighted_result = evaluate_svc_variant(
        class_weight = None, class_weight_label = "Unweighted (default)", C = args.C, max_iter = args.max_iter,
        X_train = X_train, y_train = train["y"], subject_train = train["subject"],
        X_val = X_val, y_val = validation["y"], subject_val = validation["subject"],
        cv_folds = args.cv_folds,
    )
    balanced_result = evaluate_svc_variant(
        class_weight = "balanced", class_weight_label = "Balanced", C = args.C, max_iter = args.max_iter,
        X_train = X_train, y_train = train["y"], subject_train = train["subject"],
        X_val = X_val, y_val = validation["y"], subject_val = validation["subject"],
        cv_folds = args.cv_folds,
    )

    print("\n--- Comparison ---")
    print(f"{'Metric':<28} {'Unweighted':>12} {'Balanced':>12}")
    print(f"{'Validation accuracy':<28} {unweighted_result['validation_accuracy']:>12.4f} {balanced_result['validation_accuracy']:>12.4f}")
    print(f"{'Validation macro-F1':<28} {unweighted_result['validation_macro_f1']:>12.4f} {balanced_result['validation_macro_f1']:>12.4f}")
    print(f"{'CV mean accuracy':<28} {unweighted_result['cv_mean_accuracy']:>12.4f} {balanced_result['cv_mean_accuracy']:>12.4f}")
    print(f"{'CV mean macro-F1':<28} {unweighted_result['cv_mean_macro_f1']:>12.4f} {balanced_result['cv_mean_macro_f1']:>12.4f}")

    print_classical_comparison(
        REPO_ROOT / args.out,
        unweighted_result["validation_macro_f1"],
        balanced_result["validation_macro_f1"],
    )

    # Decision rule, applied explicitly rather than silently picking
    # whichever macro-F1 is numerically higher. This is exactly the
    # mistake previously made and then reversed. "Meaningfully better" here
    # means: beats CV fold-to-fold spread AND does not cost accuracy
    # disproportionately (judged against what was previously found too costly).
    cv_f1_spread_unweighted = max(unweighted_result["cv_fold_macro_f1"]) - min(unweighted_result["cv_fold_macro_f1"])
    cv_f1_spread_balanced = max(balanced_result["cv_fold_macro_f1"]) - min(balanced_result["cv_fold_macro_f1"])
    noise_floor = max(cv_f1_spread_unweighted, cv_f1_spread_balanced)
    macro_f1_gain = balanced_result["validation_macro_f1"] - unweighted_result["validation_macro_f1"]
    accuracy_cost = unweighted_result["validation_accuracy"] - balanced_result["validation_accuracy"]

    print(f"\nCV fold-to-fold macro-F1 spread (noise floor): unweighted = {cv_f1_spread_unweighted:.4f}, balanced = {cv_f1_spread_balanced:.4f}")
    print(f"Balanced vs. unweighted: macro-F1 gain = {macro_f1_gain:+.4f}, accuracy cost = {accuracy_cost:+.4f}")

    if macro_f1_gain > noise_floor and accuracy_cost < 0.10:
        saved_result, saved_label = balanced_result, "balanced"
        print(f"  -> Balanced class weights clear the CV noise floor without a disproportionate accuracy cost. Saving the BALANCED model as the baseline.")
    else:
        saved_result, saved_label = unweighted_result, "unweighted"
        print(f"  -> Balanced class weights do not clearly clear the noise floor, or cost too much accuracy (same pattern as previous priors finding). Saving the UNWEIGHTED model as the baseline.")

    model_dir = REPO_ROOT / "models"
    model_dir.mkdir(parents = True, exist_ok = True)
    joblib.dump(
        {"model": saved_result["model"], "scaler": scaler, "feature_columns": kept_columns, "class_weight": saved_label},
        model_dir / "linearsvc_baseline.joblib",
    )
    print(f"Saved {saved_label} model to {model_dir / 'linearsvc_baseline.joblib'}")

    args.out.mkdir(parents = True, exist_ok = True)
    results = {
        "model": f"LinearSVC (C = {args.C}, dual = False, max_iter = {args.max_iter})",
        "feature_columns_used": kept_columns,
        "feature_columns_excluded": [c for c in feature_columns if c not in kept_columns],
        "unweighted": {k: v for k, v in unweighted_result.items() if k != "model"},
        "balanced": {k: v for k, v in balanced_result.items() if k != "model"},
        "saved_model_class_weight": saved_label,
    }
    results_path = REPO_ROOT / args.out / "linearsvc_baseline_results.json"
    results_path.write_text(json.dumps(results, indent = 2))
    print(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()