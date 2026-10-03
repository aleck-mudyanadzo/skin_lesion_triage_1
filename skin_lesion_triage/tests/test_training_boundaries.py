import json

import numpy as np
import pytest
import tensorflow as tf
import cv2

from app.model_manager import ModelManager
from scripts import train_model


class FakeDataset:
    class_names = ["benign", "malignant"]

    def map(self, function, **kwargs):
        return self

    def prefetch(self, buffer_size):
        return self


def test_training_datasets_load_only_train_and_validation(monkeypatch):
    loaded = []
    monkeypatch.setattr(train_model, "validate_split_manifest", lambda: {})
    monkeypatch.setattr(
        train_model,
        "_load_directory",
        lambda path, shuffle=False: loaded.append((path.name, shuffle)) or FakeDataset(),
    )
    monkeypatch.setattr(train_model, "build_augmentation", lambda: object())

    train_ds, val_ds, class_names = train_model.build_datasets("mobilenetv2")

    assert loaded == [("train", True), ("val", False)]
    assert isinstance(train_ds, FakeDataset)
    assert isinstance(val_ds, FakeDataset)
    assert class_names == ["benign", "malignant"]


def test_validation_stage_does_not_inspect_test_image_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    split_summary = {
        split: {
            "rows": 2,
            "class_distribution": {
                "benign": {"count": 1},
                "malignant": {"count": 1},
            },
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "grouping_column": "lesion_id",
        "group_disjoint": True,
        "group_overlaps": {"train/val": [], "train/test": [], "val/test": []},
        "splits": split_summary,
    }
    data_dir.mkdir()
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for split in ("train", "val"):
        for label in ("benign", "malignant"):
            directory = data_dir / split / label
            directory.mkdir(parents=True)
            (directory / "sample.jpg").write_bytes(b"sample")
    monkeypatch.setattr(train_model, "DATA_DIR", data_dir)

    train_model.validate_split_manifest()

    with pytest.raises(ValueError, match="test/benign image count"):
        train_model.validate_split_manifest(verify_test_files=True)


def test_final_test_loader_accepts_plain_tf_dataset_and_enforces_class_order(tmp_path, monkeypatch):
    test_dir = tmp_path / "test"
    for class_name, color in (("benign", (20, 80, 160)), ("malignant", (180, 90, 30))):
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True)
        image = np.full((12, 16, 3), color, dtype=np.uint8)
        assert cv2.imwrite(str(class_dir / "sample.jpg"), image)

    manifest_checks = []
    monkeypatch.setattr(train_model, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        train_model,
        "validate_split_manifest",
        lambda verify_test_files=False: manifest_checks.append(verify_test_files),
    )
    monkeypatch.setattr(train_model, "IMG_SIZE", (16, 16))

    test_ds = train_model.load_test_dataset("mobilenetv2")
    _, labels = next(iter(test_ds))

    assert manifest_checks == [True]
    np.testing.assert_array_equal(labels.numpy().ravel(), [0.0, 1.0])


def test_final_evaluation_uses_locked_threshold_in_a_single_pass():
    scores = tf.constant([[0.6], [0.9], [0.8], [0.2]], dtype=tf.float32)
    labels = tf.constant([0.0, 1.0, 1.0, 0.0], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((scores, labels)).batch(2)

    class Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, images, training=False):
            self.calls += 1
            return images

    model = Model()
    result = train_model.evaluate_final_test(model, dataset, locked_threshold=0.7)

    assert model.calls == 2
    assert result["threshold"] == 0.7
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (2, 0, 0, 2)
    assert result["accuracy"] == 1.0
    assert result["recall"] == 1.0


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_final_evaluation_rejects_unlocked_or_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="finite value in \\[0, 1\\]"):
        train_model.evaluate_final_test(None, [], threshold)


def test_explicit_final_command_reads_locked_threshold_and_evaluates_once(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "resnet50_skin_lesion.keras"
    model_path.write_bytes(b"selected-model-placeholder")
    threshold_path = model_dir / "resnet50_skin_lesion_threshold.json"
    threshold_path.write_text(json.dumps({"threshold": 0.37}), encoding="utf-8")
    monkeypatch.setattr(train_model, "MODEL_DIR", model_dir)
    monkeypatch.setattr(train_model.tf.keras.models, "load_model", lambda path: "selected-model")
    monkeypatch.setattr(train_model, "load_test_dataset", lambda arch: "held-out-test")
    calls = []
    monkeypatch.setattr(
        train_model,
        "evaluate_final_test",
        lambda model, test_ds, locked_threshold: calls.append((model, test_ds, locked_threshold)) or {"recall": 0.5},
    )

    train_model.run_final_test_evaluation("resnet50")

    assert calls == [("selected-model", "held-out-test", 0.37)]
    assert json.loads((model_dir / "resnet50_skin_lesion_final_test_metrics.json").read_text()) == {"recall": 0.5}


@pytest.mark.parametrize("threshold", [-0.2, 1.2, float("nan")])
def test_inference_rejects_invalid_saved_threshold(tmp_path, threshold):
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    manager = ModelManager.__new__(ModelManager)
    manager.threshold_path = threshold_path
    manager.threshold = 0.5
    manager.threshold_source = "default"

    manager._load_threshold()

    assert manager.threshold == 0.5
    assert manager.threshold_source == "default"
