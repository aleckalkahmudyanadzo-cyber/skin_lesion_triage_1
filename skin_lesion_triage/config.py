"""
config.py
Central configuration for the Skin Lesion Triage System.
Reads from environment where available, falls back to sane defaults.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # --- Flask core ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-before-deployment")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

    # --- Paths ---
    UPLOAD_FOLDER = BASE_DIR / "app" / "static" / "uploads"
    MODEL_DIR = BASE_DIR / "models"
    DATABASE_PATH = BASE_DIR / "instance" / "triage_logs.db"

    # --- Upload constraints ---
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload

    # --- Model settings ---
    # Which architecture to load at runtime: "resnet50" or "mobilenetv2"
    ACTIVE_MODEL = os.environ.get("ACTIVE_MODEL", "mobilenetv2")
    IMG_SIZE = (224, 224)
    CLASS_NAMES = ["Benign", "Malignant Suspect"]

    # Model file names (must match what train_model.py saves)
    MODEL_PATHS = {
        "resnet50": MODEL_DIR / "resnet50_skin_lesion.keras",
        "mobilenetv2": MODEL_DIR / "mobilenetv2_skin_lesion.keras",
    }

    # Grad-CAM target conv layer names per architecture
    GRADCAM_LAYER = {
        "resnet50": "conv5_block3_out",
        "mobilenetv2": "Conv_1",
    }

    # Decision threshold on the malignant-suspect probability
    RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.5"))
