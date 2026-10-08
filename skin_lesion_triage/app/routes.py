"""
routes.py
All HTTP endpoints for the triage system.
"""

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request, abort

from app.database import (
    get_summary_stats,
    get_survey_summary,
    get_recent_logs,
    get_prediction_by_reference_code,
    log_prediction,
    log_survey_response,
    compute_sus_score,
)
from app.model_manager import get_model_manager
from app.preprocessing import ImageQualityError, preprocess_image
from config import Config

bp = Blueprint("main", __name__)


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/health")
def health():
    manager = get_model_manager()
    return jsonify({
        "status": "ok",
        "model_loaded": manager.is_ready(),
        "active_model": manager.arch,
        "risk_threshold": manager.threshold,
        "threshold_source": manager.threshold_source,
    })


@bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under field 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG or JPG."}), 400

    image_bytes = file.read()

    manager = get_model_manager()
    if not manager.is_ready():
        return jsonify({
            "error": f"Model not loaded. Expected file at {manager.model_path}. "
                     f"Train and place a model before running predictions."
        }), 503

    try:
        img_array, sharpness = preprocess_image(
            image_bytes,
            target_size=Config.IMG_SIZE,
            architecture=manager.arch,
        )
    except ImageQualityError as e:
        return jsonify({"error": str(e), "error_type": "quality_rejected"}), 422
    except ValueError as e:
        return jsonify({"error": str(e), "error_type": "decode_error"}), 400

    result = manager.predict(img_array)

    file_bytes = np.frombuffer(image_bytes, np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_bgr = cv2.resize(original_bgr, Config.IMG_SIZE)

    heatmap = manager.grad_cam(img_array)
    overlay = manager.overlay_heatmap(original_bgr, heatmap)

    _, original_png = cv2.imencode(".png", original_bgr)
    _, heatmap_png = cv2.imencode(".png", overlay)
    original_b64 = manager.encode_image_base64(original_bgr)
    overlay_b64 = manager.encode_image_base64(overlay)

    log_result = {"id": None, "reference_code": None}
    try:
        log_result = log_prediction(
            image_bytes, manager.arch, result, sharpness,
            original_png_bytes=original_png.tobytes(),
            heatmap_png_bytes=heatmap_png.tobytes(),
        )
    except Exception as e:
        current_app.logger.warning(f"Failed to log prediction: {e}")

    return jsonify({
        **result,
        "reference_id": log_result["id"],
        "reference_code": log_result["reference_code"],
        "sharpness_score": round(sharpness, 1),
        "model_used": manager.arch,
        "original_image_b64": original_b64,
        "heatmap_overlay_b64": overlay_b64,
    })


@bp.route("/result/<reference_code>")
def view_result(reference_code):
    row = get_prediction_by_reference_code(reference_code)
    if row is None or not row.get("original_image_path"):
        abort(404)

    import base64
    try:
        with open(row["original_image_path"], "rb") as f:
            original_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(row["heatmap_image_path"], "rb") as f:
            heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        abort(404)

    return render_template(
        "result.html",
        row=row,
        original_b64=original_b64,
        heatmap_b64=heatmap_b64,
    )


@bp.route("/lookup")
def lookup():
    return render_template("lookup.html")


@bp.route("/stats")
def stats():
    try:
        data = get_summary_stats()
        data["survey"] = get_survey_summary()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/history")
def history():
    try:
        logs = get_recent_logs(limit=50)
        summary = get_summary_stats()
    except Exception as e:
        logs = []
        summary = {"total_predictions": 0, "urgent_referrals": 0}
        current_app.logger.warning(f"Failed to load history: {e}")
    return render_template("history.html", logs=logs, summary=summary)


SUS_STATEMENTS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]


@bp.route("/survey", methods=["GET", "POST"])
def survey():
    if request.method == "GET":
        return render_template("survey.html", statements=SUS_STATEMENTS, submitted=False)

    answers = {}
    errors = []
    for i in range(1, 11):
        raw = request.form.get(f"q{i}")
        if raw is None or raw == "":
            errors.append(f"Question {i} was not answered.")
            continue
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"Question {i} has an invalid answer.")
            continue
        if value < 1 or value > 5:
            errors.append(f"Question {i} must be rated 1 to 5.")
            continue
        answers[i] = value

    if errors:
        return render_template(
            "survey.html", statements=SUS_STATEMENTS, submitted=False, errors=errors
        ), 400

    sus_score = compute_sus_score(answers)

    try:
        log_survey_response(answers, sus_score)
    except Exception as e:
        current_app.logger.warning(f"Failed to log survey response: {e}")

    return render_template(
        "survey.html", statements=SUS_STATEMENTS, submitted=True, sus_score=sus_score
    )


@bp.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 8MB."}), 413
