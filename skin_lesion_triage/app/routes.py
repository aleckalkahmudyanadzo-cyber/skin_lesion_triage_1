"""
routes.py
All HTTP endpoints for the triage system.
"""

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request

from app.database import get_summary_stats, log_prediction
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

    # --- Preprocess (raises ImageQualityError for blurry/unusable images) ---
    try:
        img_array, sharpness = preprocess_image(image_bytes, target_size=Config.IMG_SIZE)
    except ImageQualityError as e:
        return jsonify({"error": str(e), "error_type": "quality_rejected"}), 422
    except ValueError as e:
        return jsonify({"error": str(e), "error_type": "decode_error"}), 400

    # --- Predict ---
    result = manager.predict(img_array)

    # --- Grad-CAM overlay ---
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_bgr = cv2.resize(original_bgr, Config.IMG_SIZE)

    heatmap = manager.grad_cam(img_array)
    overlay = manager.overlay_heatmap(original_bgr, heatmap)
    overlay_b64 = manager.encode_image_base64(overlay)
    original_b64 = manager.encode_image_base64(original_bgr)

    # --- Log (best-effort; don't fail the request if logging breaks) ---
    try:
        log_prediction(image_bytes, manager.arch, result, sharpness)
    except Exception as e:
        current_app.logger.warning(f"Failed to log prediction: {e}")

    return jsonify({
        **result,
        "sharpness_score": round(sharpness, 1),
        "model_used": manager.arch,
        "original_image_b64": original_b64,
        "heatmap_overlay_b64": overlay_b64,
    })


@bp.route("/stats")
def stats():
    try:
        return jsonify(get_summary_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 8MB."}), 413
