"""
train_model.py
Two-phase transfer learning training script for the skin lesion triage model.
Designed to run on Google Colab (GPU runtime) but works locally too if you
have a GPU and the data/processed folder already populated by prepare_data.py.

Phase 1 (feature extraction): base CNN frozen, train only the new
  classification head. Fast, prevents destroying pretrained features early.
Phase 2 (fine-tuning): unfreeze the top N layers of the base CNN and
  train at a much lower learning rate to adapt features to skin lesions.

USAGE (Colab):
    !python train_model.py --arch mobilenetv2 --epochs_head 10 --epochs_finetune 15
    !python train_model.py --arch resnet50 --epochs_head 10 --epochs_finetune 15

Objective 2 target: validation recall >= 0.85 on the malignant class.
Recall is prioritized over raw accuracy because in a triage setting a missed
malignant case (false negative) is far costlier than a false positive.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision, AUC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backbone_preprocessing import preprocess_rgb_for_model
from app.image_preprocessing import preprocess_image_file

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

ARCH_BUILDERS = {
    "resnet50": (ResNet50, "resnet50_skin_lesion"),
    "mobilenetv2": (MobileNetV2, "mobilenetv2_skin_lesion"),
}

FINE_TUNE_UNFREEZE_LAYERS = {
    "resnet50": 30,       # unfreeze last 30 layers of ResNet50 in phase 2
    "mobilenetv2": 40,    # unfreeze last 40 layers of MobileNetV2 in phase 2
}


def _load_directory(directory: Path, shuffle=False):
    class_names = ["benign", "malignant"]
    image_paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = directory / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        files = sorted(class_dir.glob("*.jpg"))
        if not files:
            raise ValueError(f"No .jpg images found in {class_dir}")
        image_paths.extend(str(path) for path in files)
        labels.extend([label] * len(files))

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=True)

    def load_and_preprocess(path, label):
        rgb = tf.numpy_function(
            lambda image_path: preprocess_image_file(image_path, IMG_SIZE),
            [path],
            Tout=tf.uint8,
        )
        rgb.set_shape((*IMG_SIZE, 3))
        return tf.cast(rgb, tf.float32), tf.reshape(tf.cast(label, tf.float32), (1,))

    return dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)


def validate_split_manifest(verify_test_files=False):
    """Refuse training/evaluation unless prepared files match a disjoint manifest."""
    manifest_path = DATA_DIR / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; rerun scripts/prepare_data.py to create auditable lesion-level splits."
        )
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("grouping_column") != "lesion_id" or manifest.get("group_disjoint") is not True:
        raise ValueError("Split manifest does not verify lesion_id-disjoint train/validation/test groups.")
    overlaps = manifest.get("group_overlaps", {})
    required_pairs = {"train/val", "train/test", "val/test"}
    if not isinstance(overlaps, dict) or not required_pairs.issubset(overlaps):
        raise ValueError("Split manifest is missing the required pairwise lesion-overlap checks.")
    if any(overlaps[pair] for pair in required_pairs):
        raise ValueError("Split manifest records overlapping lesion_id groups.")

    split_info = manifest.get("splits", {})
    if any(split not in split_info for split in ("train", "val", "test")):
        raise ValueError("Split manifest must summarize train, validation, and test.")
    splits_to_verify = ("train", "val", "test") if verify_test_files else ("train", "val")
    for split in splits_to_verify:
        expected_rows = split_info[split].get("rows")
        class_counts = split_info[split].get("class_distribution", {})
        if expected_rows != sum(class_counts.get(label, {}).get("count", 0) for label in ("benign", "malignant")):
            raise ValueError(f"{split} split manifest row count does not match its class counts.")
        for label in ("benign", "malignant"):
            expected = class_counts.get(label, {}).get("count")
            actual = len(list((DATA_DIR / split / label).glob("*.jpg")))
            if expected is None or expected != actual or actual == 0:
                raise ValueError(
                    f"{split}/{label} image count ({actual}) does not match its non-empty "
                    f"split manifest count ({expected})."
                )
    return manifest


def build_augmentation():
    """Augment normalized RGB values; brightness range must match [0, 1]."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        layers.GaussianNoise(0.03),
    ])


def _prepare_dataset(dataset, arch: str, augmentation=None):
    def prepare_batch(images, labels):
        pixels_0_1 = images / 255.0
        if augmentation is not None:
            pixels_0_1 = augmentation(pixels_0_1, training=True)
            pixels_0_1 = tf.clip_by_value(pixels_0_1, 0.0, 1.0)
        pixels_0_255 = pixels_0_1 * 255.0
        return preprocess_rgb_for_model(pixels_0_255, arch), labels

    return dataset.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)


def build_datasets(arch: str):
    """Load only training and validation data for fitting/model selection."""
    validate_split_manifest()
    train_raw = _load_directory(DATA_DIR / "train", shuffle=True)
    # Stable validation order keeps labels aligned with model.predict in threshold selection.
    val_raw = _load_directory(DATA_DIR / "val")
    class_names = ["benign", "malignant"]
    print(f"[*] Class order (0/1): {class_names}")

    # Augmentation is performed on [0, 1] values, before architecture-specific
    # ImageNet preprocessing shared with the inference path.
    augmentation = build_augmentation()
    train_ds = _prepare_dataset(train_raw, arch, augmentation)
    val_ds = _prepare_dataset(val_raw, arch)
    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        class_names,
    )


def load_test_dataset(arch: str):
    """Load the held-out test data only for the explicit final evaluation."""
    validate_split_manifest(verify_test_files=True)
    test_raw = _load_directory(DATA_DIR / "test")
    return _prepare_dataset(test_raw, arch).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(train_dir: Path) -> dict:
    """HAM10000 is heavily imbalanced (many more benign nv cases than
    malignant). Class weighting helps the model not just learn to always
    predict 'benign'."""
    benign_count = len(list((train_dir / "benign").glob("*.jpg")))
    malignant_count = len(list((train_dir / "malignant").glob("*.jpg")))
    total = benign_count + malignant_count
    weight_for_benign = total / (2.0 * benign_count)
    weight_for_malignant = total / (2.0 * malignant_count)
    print(f"[*] Class counts -> benign: {benign_count}, malignant: {malignant_count}")
    print(f"[*] Class weights -> benign: {weight_for_benign:.3f}, malignant: {weight_for_malignant:.3f}")
    return {0: weight_for_benign, 1: weight_for_malignant}


def build_model(arch: str):
    builder_fn, _ = ARCH_BUILDERS[arch]
    base_model = builder_fn(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Phase 1: frozen

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", Recall(name="recall"), Precision(name="precision"), AUC(name="auc")],
    )


def tune_threshold(model, val_ds, target_recall: float = 0.85) -> dict:
    """
    Objective 2 asks for validation recall >= 0.85 on the malignant class.
    The default 0.5 cutoff on the sigmoid output rarely lands there directly,
    so this scans the validation set's precision/recall curve and picks the
    lowest threshold that still meets the recall target, preferring higher
    precision among threshold values that clear the bar. If no threshold on
    the curve reaches the target, it falls back to whichever threshold gives
    the highest recall available, and flags that fallback in the result.
    """
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0).ravel()
    y_prob = model.predict(val_ds, verbose=0).ravel()

    if y_true.size == 0 or set(np.unique(y_true)) != {0.0, 1.0}:
        raise ValueError("Threshold selection requires validation examples from both binary classes.")
    if y_prob.shape != y_true.shape or not np.isfinite(y_prob).all():
        raise ValueError("Validation predictions must be finite and align one-to-one with validation labels.")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more point than thresholds; drop it
    # so the three arrays line up index for index.
    precision, recall = precision[:-1], recall[:-1]

    meets_target = recall >= target_recall
    if meets_target.any():
        candidate_idx = np.where(meets_target)[0]
        best_idx = candidate_idx[np.argmax(precision[candidate_idx])]
        met_target = True
    else:
        best_idx = int(np.argmax(recall))
        met_target = False

    chosen_threshold = float(thresholds[best_idx])
    return {
        "threshold": chosen_threshold,
        "val_recall_at_threshold": round(float(recall[best_idx]), 4),
        "val_precision_at_threshold": round(float(precision[best_idx]), 4),
        "target_recall": target_recall,
        "met_target": met_target,
    }


def evaluate_final_test(model, test_ds, locked_threshold: float) -> dict:
    """Evaluate a selected model once using its already-locked validation threshold."""
    if not np.isfinite(locked_threshold) or not 0.0 <= locked_threshold <= 1.0:
        raise ValueError("The locked decision threshold must be a finite value in [0, 1].")

    labels = []
    probabilities = []
    for images, batch_labels in test_ds:
        labels.append(np.asarray(batch_labels).ravel())
        probabilities.append(np.asarray(model(images, training=False)).ravel())
    if not labels:
        raise ValueError("Final test evaluation requires a non-empty test dataset.")
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    if y_true.size == 0 or set(np.unique(y_true)) != {0.0, 1.0}:
        raise ValueError("Final test evaluation requires test examples from both binary classes.")
    if y_prob.shape != y_true.shape or not np.isfinite(y_prob).all():
        raise ValueError("Test predictions must be finite and align one-to-one with test labels.")

    y_pred = y_prob >= locked_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(locked_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_final_test_evaluation(arch: str):
    """Load the chosen model and run its explicit final evaluation once."""
    _, model_filename = ARCH_BUILDERS[arch]
    model_path = MODEL_DIR / f"{model_filename}.keras"
    threshold_path = MODEL_DIR / f"{model_filename}_threshold.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model does not exist: {model_path}")
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"No validation-selected threshold exists at {threshold_path}; train and select before final evaluation."
        )
    with open(threshold_path, encoding="utf-8") as threshold_file:
        threshold_info = json.load(threshold_file)
    if "threshold" not in threshold_info:
        raise ValueError(f"Threshold file does not contain a locked threshold: {threshold_path}")

    locked_threshold = float(threshold_info["threshold"])
    selected_model = tf.keras.models.load_model(model_path)
    test_ds = load_test_dataset(arch)
    results = evaluate_final_test(selected_model, test_ds, locked_threshold)
    results_path = MODEL_DIR / f"{model_filename}_final_test_metrics.json"
    with open(results_path, "w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)
        result_file.write("\n")
    print(f"Final held-out test evaluation at the locked threshold {locked_threshold:.4f}:")
    for key, value in results.items():
        print(f"    {key}: {value}")
    print(f"Saved final test metrics to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train/select ResNet50 or MobileNetV2 using validation data.")
    parser.add_argument("--arch", choices=["resnet50", "mobilenetv2"], default="mobilenetv2")
    parser.add_argument("--epochs_head", type=int, default=10, help="Phase 1: frozen-base epochs")
    parser.add_argument("--epochs_finetune", type=int, default=15, help="Phase 2: fine-tuning epochs")
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-5)
    parser.add_argument(
        "--final-evaluate",
        action="store_true",
        help="Explicitly evaluate the selected model once on held-out test data at its saved validation threshold.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _, model_filename = ARCH_BUILDERS[args.arch]
    checkpoint_path = MODEL_DIR / f"{model_filename}.keras"

    if args.final_evaluate:
        run_final_test_evaluation(args.arch)
        return

    train_ds, val_ds, _ = build_datasets(args.arch)
    class_weights = compute_class_weights(DATA_DIR / "train")

    model, base_model = build_model(args.arch)

    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_recall", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_recall", mode="max", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # ---------- Phase 1: train the head only ----------
    print("\n" + "=" * 60)
    print(f"PHASE 1: Training classification head ({args.arch}, base frozen)")
    print("=" * 60)
    compile_model(model, args.lr_head)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---------- Phase 2: fine-tune top layers of the base ----------
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-tuning top layers of {args.arch}")
    print("=" * 60)
    base_model.trainable = True
    unfreeze_n = FINE_TUNE_UNFREEZE_LAYERS[args.arch]
    for layer in base_model.layers[:-unfreeze_n]:
        layer.trainable = False

    compile_model(model, args.lr_finetune)  # much lower LR to avoid catastrophic forgetting
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---------- Threshold tuning on the validation set ----------
    print("\n" + "=" * 60)
    print("TUNING DECISION THRESHOLD on validation set")
    print("=" * 60)
    threshold_info = tune_threshold(model, val_ds, target_recall=0.85)
    threshold_path = MODEL_DIR / f"{model_filename}_threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)
        f.write("\n")

    if threshold_info["met_target"]:
        print(f"[✓] Threshold {threshold_info['threshold']} reaches the 0.85 recall "
              f"target (val recall={threshold_info['val_recall_at_threshold']}, "
              f"val precision={threshold_info['val_precision_at_threshold']}).")
    else:
        print(f"[!] No threshold on the validation curve reached 0.85 recall. "
              f"Falling back to the highest-recall threshold available: "
              f"{threshold_info['threshold']} "
              f"(val recall={threshold_info['val_recall_at_threshold']}). "
              f"Consider retraining before trusting this model for triage.")
    print(f"    Saved to {threshold_path}")

    model.save(checkpoint_path)
    print(f"\n[✓] Model saved to {checkpoint_path}")
    print("    Validation selection is complete; no held-out test data was accessed.")
    print("    After choosing one architecture, run --final-evaluate once for its locked-threshold test metrics.")


if __name__ == "__main__":
    main()
