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


def check_quality(img_bgr: np.ndarray, blur_threshold: float = 8.0) -> float:
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


def check_is_skin(img_bgr: np.ndarray, min_skin_fraction: float = 0.30) -> float:
    """
    Pragmatic pre-filter, not a trained classifier: rejects images that are
    unlikely to be a close-up photo of skin. Combines two color-space skin
    models (YCrCb and HSV) so a color only counts as "skin" if it passes
    both, which rules out most tan/brown object surfaces (bottle plastic,
    labels, wood, cardboard). Also requires the skin-toned pixels to form
    one large contiguous region rather than just being scattered across
    the frame, since a real close-up lesion photo is dominated by one
    solid patch of skin.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(
        hsv,
        np.array([0, 20, 70], dtype=np.uint8),
        np.array([25, 170, 255], dtype=np.uint8),
    )

    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = max((cv2.contourArea(c) for c in contours), default=0)
    skin_fraction = float(largest_area) / (mask.shape[0] * mask.shape[1])

    if skin_fraction < min_skin_fraction:
        raise ImageQualityError(
            f"This does not look like a close-up photo of skin (only "
            f"{skin_fraction*100:.0f}% of the frame is a solid skin-toned "
            f"region). Please upload or take a clear, close-up photo of the "
            f"skin area you want checked, filling as much of the frame as "
            f"possible."
        )
    return skin_fraction


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
    # check_is_skin(enhanced_bgr)  # disabled: too unreliable on dark/pigmented lesions, see dissertation limitations

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