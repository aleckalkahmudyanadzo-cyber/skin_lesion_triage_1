"""
prepare_data.py
Downloads (if needed) and prepares the HAM10000 dataset for binary
classification: Benign vs. Malignant Suspect.

USAGE (from project root):
    python scripts/prepare_data.py --kaggle_download
    python scripts/prepare_data.py --skip_download   # if you already have the raw folder

HAM10000 diagnosis codes and how we map them:
    Benign:              nv (melanocytic nevi), bkl (benign keratosis),
                          df (dermatofibroma), vasc (vascular lesions)
    Malignant Suspect:   mel (melanoma), bcc (basal cell carcinoma),
                          akiec (actinic keratoses / intraepithelial carcinoma)

This mapping is a simplification for the triage use-case (Objective 2):
the system's job is to flag "needs a dermatologist look" rather than give
a differential diagnosis.

Output structure (used directly by ImageDataGenerator / image_dataset_from_directory):
    data/processed/
        train/benign/*.jpg
        train/malignant/*.jpg
        val/benign/*.jpg
        val/malignant/*.jpg
        test/benign/*.jpg
        test/malignant/*.jpg
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BENIGN_CODES = {"nv", "bkl", "df", "vasc"}
MALIGNANT_CODES = {"mel", "bcc", "akiec"}

# Kaggle dataset slug for HAM10000 (kmader mirror is the most commonly used one)
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"


def download_from_kaggle():
    """
    Requires the `kaggle` CLI to be installed and a valid kaggle.json API
    token placed at ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json).
    Get your token from https://www.kaggle.com/settings -> API -> Create New Token.
    """
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
    candidates = list(RAW_DIR.rglob("HAM10000_metadata.csv"))
    if not candidates:
        sys.exit(
            f"ERROR: Could not find HAM10000_metadata.csv under {RAW_DIR}. "
            f"Run with --kaggle_download or place the extracted dataset there manually."
        )
    return candidates[0]


def locate_image_dirs() -> list[Path]:
    """HAM10000 on Kaggle typically ships as two folders:
    HAM10000_images_part_1 and HAM10000_images_part_2."""
    dirs = [p for p in RAW_DIR.rglob("*") if p.is_dir() and "images" in p.name.lower()]
    if not dirs:
        sys.exit(f"ERROR: Could not find HAM10000 image folders under {RAW_DIR}.")
    return dirs


def build_image_index(image_dirs: list[Path]) -> dict:
    """Maps image_id -> full file path, across both HAM10000 image folders."""
    index = {}
    for d in image_dirs:
        for img_path in d.glob("*.jpg"):
            index[img_path.stem] = img_path
    return index


def label_binary(dx_code: str) -> str:
    if dx_code in MALIGNANT_CODES:
        return "malignant"
    if dx_code in BENIGN_CODES:
        return "benign"
    return "unknown"


def prepare(test_size=0.15, val_size=0.15, seed=42):
    metadata_csv = locate_metadata_csv()
    print(f"[*] Using metadata: {metadata_csv}")
    df = pd.read_csv(metadata_csv)

    df["binary_label"] = df["dx"].apply(label_binary)
    df = df[df["binary_label"] != "unknown"].reset_index(drop=True)

    print("[*] Class distribution:")
    print(df["binary_label"].value_counts())

    image_dirs = locate_image_dirs()
    image_index = build_image_index(image_dirs)
    print(f"[*] Indexed {len(image_index)} images across {len(image_dirs)} folders.")

    # Stratified split: train / val / test
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), stratify=df["binary_label"], random_state=seed
    )
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val_size), stratify=temp_df["binary_label"], random_state=seed
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, split_df in splits.items():
        for label in ("benign", "malignant"):
            (PROCESSED_DIR / split_name / label).mkdir(parents=True, exist_ok=True)

        copied, missing = 0, 0
        for _, row in split_df.iterrows():
            image_id = row["image_id"]
            label = row["binary_label"]
            src = image_index.get(image_id)
            if src is None:
                missing += 1
                continue
            dst = PROCESSED_DIR / split_name / label / f"{image_id}.jpg"
            shutil.copyfile(src, dst)
            copied += 1

        print(f"[*] {split_name}: copied {copied} images ({missing} missing/skipped)")

    print("\n[✓] Data preparation complete.")
    print(f"    Output at: {PROCESSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000 for binary triage classification.")
    parser.add_argument("--kaggle_download", action="store_true", help="Download the dataset via Kaggle CLI first.")
    parser.add_argument("--skip_download", action="store_true", help="Skip download; use existing data/raw contents.")
    args = parser.parse_args()

    if args.kaggle_download:
        download_from_kaggle()
    elif not args.skip_download:
        print("[!] Neither --kaggle_download nor --skip_download passed. Assuming data/raw is already populated.")

    prepare()
