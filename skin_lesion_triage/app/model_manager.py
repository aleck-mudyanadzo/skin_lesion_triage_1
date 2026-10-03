"""
model_manager.py
Loads the trained Keras model (ResNet50 or MobileNetV2 transfer-learning head)
and produces:
  1. A risk classification + uncalibrated sigmoid output score.
  2. A Grad-CAM heatmap visualization.

The model itself is trained separately via scripts/train_model.py (on Colab)
and dropped into /models as a .keras file.
"""

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import Config


class ModelManager:
    _instance = None

    def __init__(self, arch: str = None):
        arch = arch or Config.ACTIVE_MODEL
        self.arch = arch
        self.model_path = Path(Config.MODEL_PATHS[arch])
        self.threshold_path = Path(Config.THRESHOLD_PATHS[arch])
        self.gradcam_layer_name = Config.GRADCAM_LAYER[arch]
        self.class_names = Config.CLASS_NAMES
        self.model = None
        self.threshold = Config.RISK_THRESHOLD
        self.threshold_source = "default"
        self._load()
        self._load_threshold()

    def _load(self):
        if not self.model_path.exists():
            # Deferred failure: allows the Flask app to boot even before a
            # model has been trained/dropped in, so /health still works.
            self.model = None
            return
        self.model = tf.keras.models.load_model(self.model_path)

    def _load_threshold(self):
        """Loads the tuned threshold written by train_model.py next to the
        model, if present. Otherwise keeps Config.RISK_THRESHOLD (0.5 by
        default), which will not generally meet the Objective 2 recall
        target and should be treated as a placeholder, not a deployed
        setting."""
        if not self.threshold_path.exists():
            return
        try:
            with open(self.threshold_path) as f:
                data = json.load(f)
            threshold = float(data["threshold"])
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("Saved threshold must be finite and in [0, 1].")
            self.threshold = threshold
            self.threshold_source = "tuned"
            if not data.get("met_target", True):
                self.threshold_source = "tuned_fallback"
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupt or unexpected file: fail safe to the configured default
            # rather than crashing app start-up.
            self.threshold = Config.RISK_THRESHOLD
            self.threshold_source = "default"

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, img_array: np.ndarray) -> dict:
        """
        Args:
            img_array: shape (1, H, W, 3), float32, architecture-specific
                       ImageNet-preprocessed output of preprocess_image().
        Returns:
            dict with class label and uncalibrated sigmoid output scores.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"No model file found at {self.model_path}. "
                f"Train a model with scripts/train_model.py and place it there."
            )
        raw = self.model.predict(img_array, verbose=0)
        # Binary sigmoid output => shape (1, 1). Index 1 = "Malignant Suspect".
        malignant_score = float(raw[0][0])
        label_idx = int(malignant_score >= self.threshold)
        # Keep the legacy JSON field names for existing clients; these are
        # scores, not calibrated probabilities.
        return {
            "label": self.class_names[label_idx],
            "malignant_probability": round(malignant_score, 4),
            "benign_probability": round(1 - malignant_score, 4),
            "risk_flag": "urgent_referral" if label_idx == 1 else "routine",
            "threshold_used": self.threshold,
            "threshold_source": self.threshold_source,
        }

    def grad_cam(self, img_array: np.ndarray, pred_index: int = None) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap (values 0-1, shape H x W) highlighting
        the region of the lesion image that most influenced the prediction.
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded; cannot compute Grad-CAM.")

        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.gradcam_layer_name).output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                # Binary sigmoid: gradient of the single output neuron
                loss = predictions[:, 0]
            else:
                loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def overlay_heatmap(self, original_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Resizes heatmap to the original image size and blends it as a
        jet colormap overlay for display in the results page."""
        heatmap_resized = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlaid = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)
        return overlaid

    @staticmethod
    def encode_image_base64(img_bgr: np.ndarray) -> str:
        """Encodes a BGR numpy image as a base64 PNG string for direct
        embedding in an HTML <img src="data:image/png;base64,..."> tag."""
        success, buffer = cv2.imencode(".png", img_bgr)
        if not success:
            raise ValueError("Failed to encode image.")
        return base64.b64encode(buffer).decode("utf-8")


def get_model_manager() -> ModelManager:
    """Simple singleton accessor so the (potentially large) model is loaded
    once per process, not once per request."""
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager()
    return ModelManager._instance
