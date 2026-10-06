"""
Reusable classical modeling utilities.

Not baseline specific, per_subject_accuracy and subject_group_cv_scores
are written to be reused by every later classical baseline and by the
final evaluation, not rebuilt each time.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

def select_feature_columns(
    feature_columns: list[str],
    *,
    exclude_prefixes: tuple[str, ...] = ("zc_",),
) -> tuple[np.ndarray, list[str]]:
    """
    Return (boolean mask, kept column names) excluding any column whose
    name starts with one of `exclude_prefixes`.

    Default excludes ZC columns specifically because FEATURES.md documents
    them as confirmed structurally degenerate on this (rectified) dataset,
    not as a general purpose feature-selection tool.

    Raises
    ------
    ValueError
        If excluding these prefixes would remove every column (almost
        certainly a typo'd prefix, not an intended empty feature set).
    """

    mask = np.array([not any(column.startswith(prefix) for prefix in exclude_prefixes) for column in feature_columns])
    if not mask.any():
        raise ValueError(
            f"Excluding prefixes {exclude_prefixes} would remove all "
            f"{len(feature_columns)} columns, check for a typo."
        )
    kept = [column for column, keep in zip(feature_columns, mask) if keep]
    return mask, kept

def per_subject_accuracy(y_true: np.ndarray, y_pred: np.ndarray, subject: np.ndarray) -> dict[int, float]:
    """
    Accuracy broken down by subject, not just one aggregate number.

    Worth doing every time, not just as a diagnostic afterthought:
    real per-subject heterogeneity has already been found in this dataset (rest-peak
    anomaly rates spanning 6.40%-12.94% across just four 3-subject groups),
    so a single aggregate accuracy can hide a lot.
    """
    if not (len(y_true) == len(y_pred) == len(subject)):
        raise ValueError(
            f"Length mismatch: y_true = {len(y_true)}, y_pred = {len(y_pred)}, subject = {len(subject)}"
        )
    result = {}
    for subject_id in sorted(set(subject.tolist())):
        subject_mask = subject == subject_id
        result[int(subject_id)] = float(np.mean(y_true[subject_mask] == y_pred[subject_mask]))
    return result

def subject_group_cv_scores(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    *,
    n_splits: int,
) -> dict[str, list[float]]:
    """
    Subject-grouped k-fold CV, reporting both accuracy and macro-F1 per fold.

    Returns both because they can tell very different stories on imbalanced
    multi-class data: a model that mostly predicts the majority class can
    show respectable accuracy while macro-F1 (which weights every class
    equally, regardless of size) reveals it's barely working on most other
    classes. An earlier version of this function returned accuracy only.
    That was a real gap, not a style choice: it did not surface a genuine
    class-imbalance problem the first time this was run against real data,
    where accuracy (62.6%) looked reasonable while macro-F1 (11.0%) told
    the true story.

    Uses GroupKFold on `subject` so no subject ever appears in both the
    training and validation side of the same fold. The same
    no-subject-overlap discipline as the train/val/test/holdout split,
    applied inside cross-validation too.

    `model_factory` is a zero-argument callable returning a fresh,
    unfitted model, a new one is built for every fold, so folds can
    never share fitted state.
    """

    group_kfold = GroupKFold(n_splits = n_splits)
    accuracy_scores: list[float] = []
    macro_f1_scores: list[float] = []
    for train_index, test_index in group_kfold.split(X, y, groups = subject):
        model = model_factory()
        model.fit(X[train_index], y[train_index])
        predictions = model.predict(X[test_index])
        accuracy_scores.append(float(np.mean(y[test_index] == predictions)))
        macro_f1_scores.append(float(f1_score(y[test_index], predictions, average = "macro")))
    return {"accuracy": accuracy_scores, "macro_f1": macro_f1_scores}