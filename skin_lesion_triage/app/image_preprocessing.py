"""Deterministic OpenCV image transform shared by dataset training and serving."""

import cv2
import numpy as np


def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """Inpaint dark hair-like structures using a DullRazor-inspired mask."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def denoise(img_bgr: np.ndarray) -> np.ndarray:
    """Apply the project's fixed non-local-means denoising settings."""
    return cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=6,
        hColor=6,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def normalize_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to LAB luminance while preserving the chroma channels."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((equalized, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def enhance_bgr_image(img_bgr: np.ndarray) -> np.ndarray:
    """Apply the deterministic hair removal, denoising, and contrast steps."""
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected an HxWx3 BGR image.")
    if img_bgr.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image.")
    if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
        raise ValueError("Expected a non-empty BGR image.")

    return normalize_contrast(denoise(remove_hair(img_bgr)))


def resize_bgr_to_rgb(img_bgr: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Resize with OpenCV INTER_AREA and return contiguous RGB uint8 pixels."""
    height, width = target_size
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size must contain positive dimensions, got {target_size}.")
    resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def preprocess_bgr_image(img_bgr: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Return shared enhanced, resized RGB pixels in [0, 255] as uint8."""
    enhanced = enhance_bgr_image(img_bgr)
    return resize_bgr_to_rgb(enhanced, target_size)


def preprocess_image_file(image_path: bytes, target_size=(224, 224)) -> np.ndarray:
    """Read and preprocess a disk image for the TensorFlow training pipeline."""
    path = image_path.decode("utf-8")
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not decode training image: {path}")
    return preprocess_bgr_image(img_bgr, target_size)
