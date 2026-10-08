"""
config.py
Central configuration for the Skin Lesion Triage System.
Reads from environment where available, falls back to sane defaults.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-before-deployment")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

    UPLOAD_FOLDER = BASE_DIR / "app" / "static" / "uploads"
    MODEL_DIR = BASE_DIR / "models"
    DATABASE_PATH = BASE_DIR / "instance" / "triage_logs.db"
    RESULT_IMAGE_DIR = BASE_DIR / "instance" / "result_images"

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024

    ACTIVE_MODEL = os.environ.get("ACTIVE_MODEL", "mobilenetv2")
    IMG_SIZE = (224, 224)
    CLASS_NAMES = ["Benign", "Malignant Suspect"]

    MODEL_PATHS = {
        "resnet50": MODEL_DIR / "resnet50_skin_lesion.keras",
        "mobilenetv2": MODEL_DIR / "mobilenetv2_skin_lesion.keras",
    }

    THRESHOLD_PATHS = {
        "resnet50": MODEL_DIR / "resnet50_skin_lesion_threshold.json",
        "mobilenetv2": MODEL_DIR / "mobilenetv2_skin_lesion_threshold.json",
    }

    GRADCAM_LAYER = {
        "resnet50": "conv5_block3_out",
        "mobilenetv2": "Conv_1",
    }

    RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.5"))
