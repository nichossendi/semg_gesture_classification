import json

import pytest

from src.labels import build_label_encoding, load_label_encoding, save_label_encoding

def test_rest_always_maps_to_zero():
    encoding = build_label_encoding({"rest", "2_1", "3_5"})
    assert encoding["rest"] == 0

def test_active_labels_sort_numerically_not_lexicographically():
    # If this sorted as plain strings, "2_10" and "2_11" would land
    # between "2_1" and "2_2".
    labels = {"rest", "2_1", "2_2", "2_9", "2_10", "2_11"}
    encoding = build_label_encoding(labels)

    ordered_by_index = sorted(encoding.items(), key = lambda item: item[1])
    ordered_labels = [label for label, _ in ordered_by_index]

    assert ordered_labels == ["rest", "2_1", "2_2", "2_9", "2_10", "2_11"]

def test_encoding_covers_the_full_41_class_case():
    labels = {"rest"} | {f"2_{i}" for i in range(1, 18)} | {f"3_{i}" for i in range(1, 24)}
    encoding = build_label_encoding(labels)

    assert len(encoding) == 41
    assert set(encoding.values()) == set(range(41))  # every index 0-40 used exactly once

def test_build_label_encoding_requires_rest():
    with pytest.raises(ValueError, match = "'rest'"):
        build_label_encoding({"2_1", "2_2"})

def test_build_label_encoding_rejects_malformed_labels():
    with pytest.raises(ValueError, match = "not in"):
        build_label_encoding({"rest", "not_a_valid_label_at_all"})

def test_save_and_load_round_trip(tmp_path):
    encoding = build_label_encoding({"rest", "2_1", "3_5"})
    path = tmp_path / "label_encoding.json"

    save_label_encoding(encoding, path)
    reloaded = load_label_encoding(path)

    assert reloaded == encoding

def test_saved_file_is_sorted_by_index_for_readability(tmp_path):
    encoding = build_label_encoding({"rest", "3_5", "2_1"})
    path = tmp_path / "label_encoding.json"
    save_label_encoding(encoding, path)

    raw = json.loads(path.read_text())
    assert list(raw.keys())[0] == "rest"  # index 0 should appear first in the file