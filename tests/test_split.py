import pytest

from src.split import make_subject_split, validate_subject_split

SUBJECT_IDS = list(range(1, 28))
SEED = 20260907


def test_same_seed_produces_identical_assignments() -> None:
    first = make_subject_split(SUBJECT_IDS, seed = SEED)
    second = make_subject_split(SUBJECT_IDS, seed = SEED)

    assert first == second
    assert first == {
        "train": [7, 24, 15, 27, 17, 25, 12, 26, 2, 21, 8, 3, 1, 22, 6, 9, 10, 5],
        "validation": [20, 23, 16],
        "test": [11, 14, 19],
        "holdout": [4, 13, 18],
    }


def test_split_is_mutually_exclusive_and_exhaustive() -> None:
    split = make_subject_split(SUBJECT_IDS, seed = SEED)

    validate_subject_split(split, expected_subject_ids = SUBJECT_IDS)
    assert [len(split[name]) for name in ("train", "validation", "test", "holdout")] == [18, 3, 3, 3]


def test_validation_rejects_overlap() -> None:
    invalid_split = {
        "train": list(range(1, 19)),
        "validation": [19, 20, 21],
        "test": [21, 22, 23],
        "holdout": [24, 25, 26, 27],
    }

    with pytest.raises(ValueError, match = "overlap"):
        validate_subject_split(invalid_split, expected_subject_ids = SUBJECT_IDS)

def test_validation_rejects_non_exhaustive_split() -> None:
    invalid_split = {
        "train": list(range(1, 19)),
        "validation": [19, 20, 21],
        "test": [22, 23, 24],
        "holdout": [25, 26],  # subject 27 is missing; no group overlaps
    }

    with pytest.raises(ValueError, match = "not exhaustive"):
        validate_subject_split(invalid_split, expected_subject_ids = SUBJECT_IDS)