"""
preprocessing.py
OpenCV-based preprocessing pipeline for smartphone-acquired lesion images.

Handles the real-world problems named in the proposal (Objective 1):
- variable brightness / contrast
- sensor noise
- hair occlusion (common in dermatoscopic + smartphone skin images)
- low-quality / blurred image rejection

Pipeline: decode -> hair removal (DullRazor-style inpaint) -> denoise ->
CLAHE contrast normalization -> blur/quality check -> resize ->
architecture-specific ImageNet preprocessing.
"""

import io

import cv2
import numpy as np
from PIL import Image

from app.backbone_preprocessing import preprocess_rgb_for_model
from app.image_preprocessing import (
    enhance_bgr_image,
    resize_bgr_to_rgb,
    remove_hair,
    denoise,
    normalize_contrast,
)


class ImageQualityError(Exception):
    """Raised when an image fails the quality gate (too blurry / unusable)."""
    pass


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """Focus measure. Low value = blurry image."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def check_quality(img_bgr: np.ndarray, blur_threshold: float = 15.0) -> float:
    """
    Rejects images that are too blurry / featureless to be triaged safely.
    Returns the sharpness score if it passes; raises ImageQualityError if not.
    This directly implements the proposal's stated limitation: images with
    heavy hair coverage or extreme blur are rejected at preprocessing.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = _variance_of_laplacian(gray)
    if score < blur_threshold:
        raise ImageQualityError(
            f"Image too blurry for reliable triage (sharpness={score:.1f}, "
            f"minimum required={blur_threshold}). Please retake the photo in "
            f"better light, holding the camera steady."
        )
    return score


def preprocess_image(
    image_bytes: bytes,
    target_size=(224, 224),
    for_model: bool = True,
    architecture: str = "mobilenetv2",
):
    """
    Full pipeline entry point.

    Args:
        image_bytes: raw bytes read from the uploaded file.
        target_size: (H, W) expected by the CNN input layer.
        for_model: if True, returns the architecture-specific ImageNet input
                   tensor ready for model.predict(). If False, returns a
                   displayable uint8 BGR image.
        architecture: "mobilenetv2" or "resnet50"; must match the trained model.

    Returns:
        (processed_array, sharpness_score)
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        # OpenCV's JPEG decoder rejects some valid encodings phone cameras
        # produce (e.g. certain non-baseline JPEGs). Fall back to Pillow,
        # which uses a more permissive decoder.
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img_bgr = None
    if img_bgr is None:
        raise ValueError("Could not decode image. File may be corrupted or an unsupported format.")

    enhanced_bgr = enhance_bgr_image(img_bgr)
    # This is an inference-time intake safeguard, not a model transform:
    # training keeps all assigned cases rather than applying this unvalidated
    # rejection threshold to the HAM10000 distribution.
    sharpness = check_quality(enhanced_bgr)

    if not for_model:
        resized_bgr = cv2.resize(
            enhanced_bgr,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        return resized_bgr, sharpness

    img_rgb = resize_bgr_to_rgb(enhanced_bgr, target_size).astype("float32")
    img_batch = np.expand_dims(img_rgb, axis=0)
    model_pixels = preprocess_rgb_for_model(img_batch, architecture)
    return model_pixels.numpy(), sharpness