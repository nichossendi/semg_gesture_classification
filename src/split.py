"""Deterministic, participant-disjoint dataset splitting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

SPLIT_NAMES = ("train", "validation", "test", "holdout")


def make_subject_split(
    subject_ids: Iterable[int],
    *,
    seed: int,
    group_sizes: tuple[int, int, int, int] = (18, 3, 3, 3),
) -> dict[str, list[int]]:
    
    """Create a seeded, mutually exclusive split of participant IDs.

    The holdout group is assigned in the same random permutation as the other
    groups. It is not formed from manually selected leftovers.
    """

    subjects = list(subject_ids)
    if len(set(subjects)) != len(subjects):
        raise ValueError("subject_ids must not contain duplicates.")
    if sum(group_sizes) != len(subjects):
        raise ValueError(
            f"Group sizes total {sum(group_sizes)}, but {len(subjects)} subject IDs were supplied."
        )

    permutation = np.random.default_rng(seed).permutation(subjects).tolist()
    boundaries = np.cumsum(group_sizes)[:-1]
    groups = np.split(np.asarray(permutation), boundaries)
    split = {
        name: [int(subject) for subject in group]
        for name, group in zip(SPLIT_NAMES, groups, strict = True)
    }
    validate_subject_split(split, expected_subject_ids = subjects)
    return split


def validate_subject_split(
    split: dict[str, list[int]],
    *,
    expected_subject_ids: Iterable[int],
) -> None:
    
    """Raise a clear error if a split overlaps, omits, or adds participants."""

    expected = set(expected_subject_ids)
    missing_groups = set(SPLIT_NAMES).difference(split)
    unexpected_groups = set(split).difference(SPLIT_NAMES)

    if missing_groups or unexpected_groups:
        raise ValueError(
            f"Split groups must be exactly {SPLIT_NAMES}; "
            f"missing={sorted(missing_groups)}, unexpected={sorted(unexpected_groups)}."
        )

    assigned = [subject for group in split.values() for subject in group]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Participant overlap detected between split groups.")
    if set(assigned) != expected:
        raise ValueError(
            "Split is not exhaustive. "
            f"missing={sorted(expected.difference(assigned))}, "
            f"unexpected={sorted(set(assigned).difference(expected))}."
        )