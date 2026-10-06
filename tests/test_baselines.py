import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.baselines import per_subject_accuracy, select_feature_columns, subject_group_cv_scores

def test_select_feature_columns_excludes_zc_by_default():
    columns = ["mav_emg_0", "mav_emg_1", "rms_emg_0", "zc_emg_0", "zc_emg_1", "ssc_emg_0"]
    mask, kept = select_feature_columns(columns)

    assert kept == ["mav_emg_0", "mav_emg_1", "rms_emg_0", "ssc_emg_0"]
    assert mask.tolist() == [True, True, True, False, False, True]

def test_select_feature_columns_custom_exclude():
    columns = ["mav_emg_0", "rms_emg_0", "wl_emg_0"]
    mask, kept = select_feature_columns(columns, exclude_prefixes = ("rms_",))
    assert kept == ["mav_emg_0", "wl_emg_0"]

def test_select_feature_columns_raises_if_everything_excluded():
    columns = ["zc_emg_0", "zc_emg_1"]
    with pytest.raises(ValueError, match = "would remove all"):
        select_feature_columns(columns)

def test_per_subject_accuracy_hand_computed():
    y_true = np.array([0, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 0])
    subject = np.array([1, 1, 1, 1, 2, 2])

    result = per_subject_accuracy(y_true, y_pred, subject)

    assert result == {1: 0.75, 2: 1.0}

def test_per_subject_accuracy_rejects_length_mismatch():
    with pytest.raises(ValueError, match = "Length mismatch"):
        per_subject_accuracy(np.array([0, 1]), np.array([0]), np.array([1, 1]))

def test_subject_group_cv_returns_accuracy_and_macro_f1_per_fold():
    rng = np.random.default_rng(0)
    n_subjects = 6
    per_subject = 20
    X = rng.normal(size = (n_subjects * per_subject, 4))
    y = rng.integers(0, 2, size = n_subjects * per_subject)
    subject = np.repeat(np.arange(n_subjects), per_subject)

    results = subject_group_cv_scores(
        lambda: LogisticRegression(max_iter = 200), X, y, subject, n_splits = 3)

    assert set(results.keys()) == {"accuracy", "macro_f1"}
    assert len(results["accuracy"]) == 3
    assert len(results["macro_f1"]) == 3
    assert all(0.0 <= score <= 1.0 for score in results["accuracy"])
    assert all(0.0 <= score <= 1.0 for score in results["macro_f1"])

def test_subject_group_cv_fold_groups_are_actually_disjoint():
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(1)
    n_subjects = 6
    per_subject = 10
    X = rng.normal(size = (n_subjects * per_subject, 3))
    subject = np.repeat(np.arange(n_subjects), per_subject)

    for train_index, test_index in GroupKFold(n_splits = 3).split(X, groups = subject):
        train_subjects = set(subject[train_index].tolist())
        test_subjects = set(subject[test_index].tolist())
        assert train_subjects.isdisjoint(test_subjects)

def test_cv_macro_f1_exposes_majority_class_bias_that_accuracy_hides():
    """
    Reproduces, in miniature and under a controlled synthetic case, the
    exact pattern found on the real LDA run: a model that mostly
    predicts the majority class can show deceptively high accuracy while
    macro-F1 reveals it barely works on the other classes. This is the
    reason macro_f1 was added to subject_group_cv_scores in the first
    place. A fake classifier here, not real LDA, so the test is
    deterministic and fast rather than depending on LDA's actual
    empirical behavior.
    """

    class AlwaysPredictMajorityClass:
        def fit(self, X, y):
            values, counts = np.unique(y, return_counts = True)
            self._majority = values[np.argmax(counts)]
            return self

        def predict(self, X):
            return np.full(len(X), self._majority)

    rng = np.random.default_rng(2)
    n_subjects = 6
    # 90% class 0, remaining 10% spread across classes 1-4, a rough
    # miniature of the real ~59% rest / 41% spread-across-40-classes shape.
    y_majority = np.zeros(90 * n_subjects, dtype = int)
    y_minority = rng.integers(1, 5, size = 10 * n_subjects)
    y = np.concatenate([y_majority, y_minority])
    X = rng.normal(size = (len(y), 3))
    subject = np.concatenate(
        [np.repeat(np.arange(n_subjects), 90), np.repeat(np.arange(n_subjects), 10)]
    )

    results = subject_group_cv_scores(AlwaysPredictMajorityClass, X, y, subject, n_splits = 3)

    assert np.mean(results["accuracy"]) > 0.8   # looks good on accuracy alone
    assert np.mean(results["macro_f1"]) < 0.3   # macro-F1 reveals it is not