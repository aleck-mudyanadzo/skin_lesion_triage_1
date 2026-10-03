import json

import pandas as pd
import pytest

from scripts import prepare_data


def make_metadata(groups_per_class=120):
    rows = []
    image_number = 0
    for label, diagnosis in (("benign", "nv"), ("malignant", "mel")):
        for group_number in range(groups_per_class):
            lesion_id = f"{label}-{group_number}"
            for _ in range(1 + group_number % 3):
                rows.append({
                    "image_id": f"image-{image_number}",
                    "lesion_id": lesion_id,
                    "dx": diagnosis,
                })
                image_number += 1
    return pd.DataFrame(rows)


def test_metadata_rejects_missing_or_empty_lesion_ids():
    frame = make_metadata(3)
    frame.loc[0, "lesion_id"] = None
    with pytest.raises(ValueError, match="lesion_id values must be present"):
        prepare_data._validate_metadata(frame)

    frame = make_metadata(3)
    frame.loc[0, "lesion_id"] = "  "
    with pytest.raises(ValueError, match="lesion_id values must be present"):
        prepare_data._validate_metadata(frame)


def test_group_splits_are_reproducible_disjoint_and_class_balanced():
    frame = prepare_data._validate_metadata(make_metadata())
    splits = prepare_data.split_by_lesion(frame, seed=19)
    repeated = prepare_data.split_by_lesion(frame, seed=19)

    assert {
        name: part["image_id"].tolist()
        for name, part in splits.items()
    } == {
        name: part["image_id"].tolist()
        for name, part in repeated.items()
    }
    group_sets = {name: set(part["lesion_id"]) for name, part in splits.items()}
    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])

    overall_prevalence = (frame["binary_label"] == "malignant").mean()
    for part in splits.values():
        assert set(part["binary_label"]) == {"benign", "malignant"}
        assert abs((part["binary_label"] == "malignant").mean() - overall_prevalence) < 0.03

    manifest = prepare_data._build_manifest(splits, seed=19, test_size=0.15, val_size=0.15)
    assert manifest["group_disjoint"] is True
    assert manifest["group_overlaps"] == {"train/val": [], "train/test": [], "val/test": []}
    assert manifest["seed"] == 19
    assert manifest["splits"]["test"]["class_distribution"]["malignant"]["count"] > 0


def test_preparation_replaces_stale_tree_and_writes_manifest(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    images = raw / "HAM10000_images_part_1"
    images.mkdir(parents=True)
    metadata = make_metadata(groups_per_class=30)
    for image_id in metadata["image_id"]:
        (images / f"{image_id}.jpg").write_bytes(b"test-image")
    metadata_path = raw / "HAM10000_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    processed = tmp_path / "processed"
    stale = processed / "train" / "benign" / "stale.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    monkeypatch.setattr(prepare_data, "locate_metadata_csv", lambda: metadata_path)
    monkeypatch.setattr(prepare_data, "locate_image_dirs", lambda: [images])
    monkeypatch.setattr(prepare_data, "PROCESSED_DIR", processed)

    prepare_data.prepare(seed=7)

    assert not stale.exists()
    manifest = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["group_disjoint"] is True
    assert sum(item["rows"] for item in manifest["splits"].values()) == len(metadata)
    assert (processed / "split_manifest.csv").is_file()
