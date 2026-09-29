"""
Deterministic composite_label <-> integer class index encoding.

rest always maps to 0. Active labels are sorted numerically by
(exercise, restimulus), not lexicographically as strings.
"""

from __future__ import annotations

import json
from pathlib import Path


def _sort_key(composite_label: str) -> tuple[int, int]:
    exercise, restimulus = composite_label.split("_")
    return int(exercise), int(restimulus)


def build_label_encoding(composite_labels: set[str] | list[str]) -> dict[str, int]:
    """
    Build a composite_label -> integer index mapping.

    Raises
    ------
    ValueError
        If "rest" is not present, or if any non-rest label doesn't match
        the expected "{exercise}_{restimulus}" format.
    """
    labels = set(composite_labels)
    if "rest" not in labels:
        raise ValueError("Expected 'rest' to be present among composite_labels.")

    active_labels = labels - {"rest"}
    malformed = [label for label in active_labels if len(label.split("_")) != 2]
    if malformed:
        raise ValueError(f"Label(s) not in '{{exercise}}_{{restimulus}}' format: {malformed}")

    ordered_active = sorted(active_labels, key = _sort_key)
    encoding = {"rest": 0}
    encoding.update({label: index + 1 for index, label in enumerate(ordered_active)})
    return encoding


def save_label_encoding(encoding: dict[str, int], path: str | Path) -> None:
    """Save as JSON, sorted by integer index for a human-readable file."""
    path = Path(path)
    ordered = dict(sorted(encoding.items(), key = lambda item: item[1]))
    path.write_text(json.dumps(ordered, indent = 2), encoding = "utf-8")


def load_label_encoding(path: str | Path) -> dict[str, int]:
    path = Path(path)
    return json.loads(path.read_text(encoding = "utf-8"))