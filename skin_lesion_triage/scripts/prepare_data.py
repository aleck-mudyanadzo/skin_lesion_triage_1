"""
prepare_data.py
Downloads (if requested) and prepares HAM10000 for binary classification.

Lesion-level separation is enforced using HAM10000's lesion_id metadata. The
existing binary mapping is retained pending research/supervisor decisions.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BENIGN_CODES = {"nv", "bkl", "df", "vasc"}
# Retain the existing assignment; the interpretation of akiec is a pending
# supervisor/research decision and is not determined by this pipeline.
MALIGNANT_CODES = {"mel", "bcc", "akiec"}
SPLIT_CANDIDATES = 512

# Kaggle dataset slug for HAM10000 (kmader mirror is the most commonly used one)
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"


def download_from_kaggle():
    """Download HAM10000 after applicable license/institutional approvals."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading {KAGGLE_DATASET} into {RAW_DIR} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        sys.exit(
            "ERROR: 'kaggle' CLI not found. Install it with:\n"
            "    pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json and re-run this script."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"ERROR: Kaggle download failed: {e}")
    print("[*] Download complete.")


def locate_metadata_csv() -> Path:
    candidates = sorted(RAW_DIR.rglob("HAM10000_metadata.csv"))
    if not candidates:
        sys.exit(
            f"ERROR: Could not find HAM10000_metadata.csv under {RAW_DIR}. "
            f"Run with --kaggle_download or place the extracted dataset there manually."
        )
    return candidates[0]


def locate_image_dirs() -> list[Path]:
    """HAM10000 on Kaggle typically ships as two folders."""
    dirs = sorted(p for p in RAW_DIR.rglob("*") if p.is_dir() and "images" in p.name.lower())
    if not dirs:
        sys.exit(f"ERROR: Could not find HAM10000 image folders under {RAW_DIR}.")
    return dirs


def build_image_index(image_dirs: list[Path]) -> dict[str, Path]:
    """Map each image ID to exactly one image file."""
    index = {}
    for directory in image_dirs:
        for img_path in directory.glob("*.jpg"):
            if img_path.stem in index and index[img_path.stem] != img_path:
                raise ValueError(f"Duplicate image_id {img_path.stem!r} in image folders.")
            index[img_path.stem] = img_path
    return index


def label_binary(dx_code: str) -> str:
    if dx_code in MALIGNANT_CODES:
        return "malignant"
    if dx_code in BENIGN_CODES:
        return "benign"
    return "unknown"


def _validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    required = {"image_id", "lesion_id", "dx"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"HAM10000 metadata is missing required columns: {sorted(missing_columns)}")

    invalid_ids = df["lesion_id"].isna() | df["lesion_id"].astype("string").str.strip().isin(
        {"", "nan", "none", "null", "<na>"}
    )
    if invalid_ids.any():
        rows = (df.index[invalid_ids][:5] + 2).tolist()
        raise ValueError(
            "HAM10000 lesion_id values must be present and non-empty; "
            f"invalid metadata row(s): {rows}."
        )

    df = df.copy()
    df["image_id"] = df["image_id"].astype("string").str.strip()
    if df["image_id"].isna().any() or df["image_id"].eq("").any():
        raise ValueError("HAM10000 metadata contains missing or empty image_id values.")
    if df["image_id"].duplicated().any():
        duplicates = df.loc[df["image_id"].duplicated(keep=False), "image_id"].head(5).tolist()
        raise ValueError(f"HAM10000 metadata contains duplicate image_id values: {duplicates}.")

    df["lesion_id"] = df["lesion_id"].astype("string").str.strip()
    df["binary_label"] = df["dx"].astype("string").map(label_binary)
    mapped = df.loc[df["binary_label"] != "unknown"].copy()
    labels_per_lesion = mapped.groupby("lesion_id", dropna=False)["binary_label"].nunique()
    conflicting = labels_per_lesion[labels_per_lesion > 1]
    if not conflicting.empty:
        raise ValueError(
            "A lesion_id maps to multiple binary labels; resolve inconsistent metadata before splitting: "
            f"{conflicting.index[:5].tolist()}."
        )
    if mapped.empty or set(mapped["binary_label"].unique()) != {"benign", "malignant"}:
        raise ValueError("Mapped metadata must contain both benign and malignant examples.")
    return mapped.reset_index(drop=True)


def _split_score(candidate: pd.DataFrame, full: pd.DataFrame, target_fraction: float) -> float:
    """Score a group-isolated candidate for row counts and per-class counts."""
    score = ((len(candidate) - len(full) * target_fraction) / max(len(full) * target_fraction, 1)) ** 2
    for label, full_count in full["binary_label"].value_counts().items():
        target = full_count * target_fraction
        actual = int((candidate["binary_label"] == label).sum())
        score += ((actual - target) / max(target, 1)) ** 2
    return score


def _best_group_subset(frame: pd.DataFrame, target_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups_per_label = frame.groupby("binary_label")["lesion_id"].nunique()
    if (groups_per_label < 3).any():
        raise ValueError(
            "At least three distinct lesion_id groups per binary class are required "
            f"for train/validation/test splits; found {groups_per_label.to_dict()}."
        )

    splitter = GroupShuffleSplit(
        n_splits=SPLIT_CANDIDATES,
        test_size=target_fraction,
        random_state=seed,
    )
    best = None
    best_score = float("inf")
    for _, candidate_indices in splitter.split(frame, frame["binary_label"], groups=frame["lesion_id"]):
        candidate = frame.iloc[candidate_indices]
        score = _split_score(candidate, frame, target_fraction)
        if score < best_score:
            best = candidate_indices
            best_score = score

    selected = frame.iloc[best].copy()
    remainder = frame.drop(index=selected.index).copy()
    if set(selected["binary_label"].unique()) != {"benign", "malignant"}:
        raise ValueError("Could not create a split containing both binary classes without lesion leakage.")
    return selected, remainder


def split_by_lesion(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42) -> dict[str, pd.DataFrame]:
    """Reproducibly approximate requested row/class proportions without splitting lesions."""
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")

    test_df, train_val_df = _best_group_subset(df, test_size, seed)
    relative_val_size = val_size / (1.0 - test_size)
    val_df, train_df = _best_group_subset(train_val_df, relative_val_size, seed + 1)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    group_sets = {name: set(part["lesion_id"]) for name, part in splits.items()}
    overlaps = {
        f"{left}/{right}": sorted(group_sets[left] & group_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    overlaps = {pair: ids for pair, ids in overlaps.items() if ids}
    if overlaps:
        raise RuntimeError(f"Internal error: lesion_id groups overlap between splits: {overlaps}")
    for name, part in splits.items():
        if set(part["binary_label"].unique()) != {"benign", "malignant"}:
            raise ValueError(f"{name} split does not contain both binary classes.")
    return splits


def _build_manifest(splits: dict[str, pd.DataFrame], seed: int, test_size: float, val_size: float) -> dict:
    all_rows = pd.concat(splits.values(), ignore_index=True)
    total_classes = all_rows["binary_label"].value_counts().to_dict()
    summary = {
        "seed": seed,
        "split_method": "two-stage GroupShuffleSplit candidate search scored on row and class counts",
        "candidate_splits_per_stage": SPLIT_CANDIDATES,
        "requested_fractions": {"train": 1 - test_size - val_size, "val": val_size, "test": test_size},
        "grouping_column": "lesion_id",
        "group_disjoint": True,
        "group_overlaps": {},
        "splits": {},
    }
    group_sets = {name: set(part["lesion_id"]) for name, part in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        summary["group_overlaps"][f"{left}/{right}"] = sorted(group_sets[left] & group_sets[right])
        if summary["group_overlaps"][f"{left}/{right}"]:
            summary["group_disjoint"] = False

    for name, part in splits.items():
        counts = part["binary_label"].value_counts().to_dict()
        summary["splits"][name] = {
            "rows": len(part),
            "lesion_groups": int(part["lesion_id"].nunique()),
            "class_distribution": {
                label: {
                    "count": int(counts.get(label, 0)),
                    "fraction_of_split": float(counts.get(label, 0) / len(part)),
                    "fraction_of_total_class": float(counts.get(label, 0) / total_classes[label]),
                }
                for label in ("benign", "malignant")
            },
        }
    return summary


def _install_processed_tree(staged_dir: Path, target_dir: Path, workspace: Path):
    """Swap a fully prepared tree into place, restoring the previous tree on failure."""
    backup_dir = workspace / "previous_processed"
    had_previous = target_dir.exists()
    if had_previous:
        target_dir.rename(backup_dir)
    try:
        staged_dir.rename(target_dir)
    except Exception:
        if had_previous and backup_dir.exists():
            backup_dir.rename(target_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def prepare(test_size=0.15, val_size=0.15, seed=42):
    metadata_csv = locate_metadata_csv()
    print(f"[*] Using metadata: {metadata_csv}")
    source_df = pd.read_csv(metadata_csv)
    df = _validate_metadata(source_df)
    print("[*] Mapped class distribution:")
    print(df["binary_label"].value_counts())

    image_dirs = locate_image_dirs()
    image_index = build_image_index(image_dirs)
    print(f"[*] Indexed {len(image_index)} images across {len(image_dirs)} folders.")
    missing_ids = sorted(set(df["image_id"]) - set(image_index))
    if missing_ids:
        raise FileNotFoundError(
            f"{len(missing_ids)} mapped metadata images are missing; examples: {missing_ids[:5]}."
        )

    splits = split_by_lesion(df, test_size=test_size, val_size=val_size, seed=seed)
    manifest = _build_manifest(splits, seed, test_size, val_size)
    manifest["source_metadata_rows"] = int(len(source_df))
    manifest["mapped_rows"] = int(len(df))
    manifest["excluded_unknown_dx_rows"] = int(len(source_df) - len(df))
    parent = PROCESSED_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-data-", dir=parent) as temporary:
        workspace = Path(temporary)
        staged = workspace / "processed"
        for split_name, split_df in splits.items():
            for label in ("benign", "malignant"):
                (staged / split_name / label).mkdir(parents=True, exist_ok=True)
            for _, row in split_df.iterrows():
                source = image_index[row["image_id"]]
                destination = staged / split_name / row["binary_label"] / f"{row['image_id']}.jpg"
                shutil.copyfile(source, destination)

        if (PROCESSED_DIR / ".gitkeep").exists():
            shutil.copyfile(PROCESSED_DIR / ".gitkeep", staged / ".gitkeep")
        manifest_rows = pd.concat(
            [part.assign(split=name) for name, part in splits.items()],
            ignore_index=True,
        )
        manifest_rows.to_csv(staged / "split_manifest.csv", index=False)
        with open(staged / "split_manifest.json", "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")

        _install_processed_tree(staged, PROCESSED_DIR, workspace)

    print("\n[✓] Data preparation complete.")
    print(f"    Output at: {PROCESSED_DIR}")
    print(f"    Auditable manifest: {PROCESSED_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000 for binary triage classification.")
    parser.add_argument("--kaggle_download", action="store_true", help="Download the dataset via Kaggle CLI first.")
    parser.add_argument("--skip_download", action="store_true", help="Skip download; use existing data/raw contents.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed recorded in the split manifest.")
    args = parser.parse_args()

    if args.kaggle_download:
        download_from_kaggle()
    elif not args.skip_download:
        print("[!] Neither --kaggle_download nor --skip_download passed. Assuming data/raw is already populated.")

    prepare(seed=args.seed)
