"""Shared ImageNet input contract for transfer-learning and Flask inference.

Inputs are batched RGB float/int pixels in the [0, 255] range. The architecture
specific Keras application preprocessing is applied here exactly once:
MobileNetV2 maps to [-1, 1], while ResNet50 swaps RGB to BGR and subtracts
ImageNet channel means.
"""

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess


PREPROCESSORS = {
    "mobilenetv2": mobilenet_v2_preprocess,
    "resnet50": resnet50_preprocess,
}


def preprocess_rgb_for_model(rgb_pixels, architecture: str):
    """Apply the Keras ImageNet preprocessing contract to a batched RGB tensor."""
    if architecture not in PREPROCESSORS:
        raise ValueError(f"Unsupported architecture {architecture!r}; expected one of {sorted(PREPROCESSORS)}.")

    pixels = tf.convert_to_tensor(rgb_pixels)
    if pixels.shape.rank != 4:
        raise ValueError(f"Expected batched RGB pixels with rank 4, got shape {pixels.shape}.")
    if pixels.shape[-1] not in (3, None):
        raise ValueError(f"Expected three RGB channels, got shape {pixels.shape}.")

    pixels = tf.cast(pixels, tf.float32)
    if tf.executing_eagerly():
        tf.debugging.assert_greater_equal(pixels, 0.0, message="RGB pixel values must be at least 0.")
        tf.debugging.assert_less_equal(pixels, 255.0, message="RGB pixel values must be at most 255.")
    return PREPROCESSORS[architecture](pixels)
