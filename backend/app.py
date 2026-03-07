"""
DeepFake Detection API - Flask Backend
Serves analysis endpoints for image, video, and audio deepfake detection.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from analyzers.image_analyzer import analyze_image
from analyzers.video_analyzer import analyze_video
from analyzers.audio_analyzer import analyze_audio
from utils.ensemble import compute_ensemble

# ─────────────────────────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# Max file sizes
MAX_IMAGE_SIZE = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


# ─────────────────────────────────────────────────────────────────
# Frontend Serving
# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ─────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


# ─────────────────────────────────────────────────────────────────
# Image Analysis Endpoint
# ─────────────────────────────────────────────────────────────────
@app.route("/api/analyze/image", methods=["POST"])
def analyze_image_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return jsonify({"error": f"Unsupported image format: {ext}"}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_IMAGE_SIZE:
        return jsonify({"error": "File too large (max 20MB)"}), 413
    if len(file_bytes) < 100:
        return jsonify({"error": "File too small or empty"}), 400

    try:
        raw_results = analyze_image(file_bytes, file.filename)
        methods = raw_results.get("methods", {})
        ela_image = raw_results.get("ela_image")

        ensemble = compute_ensemble(methods, "image")

        return jsonify({
            "status": "success",
            "modality": "image",
            "filename": file.filename,
            "file_size_kb": round(len(file_bytes) / 1024, 1),
            "methods": methods,
            "ensemble": ensemble,
            "ela_image": ela_image,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────
# Video Analysis Endpoint
# ─────────────────────────────────────────────────────────────────
@app.route("/api/analyze/video", methods=["POST"])
def analyze_video_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return jsonify({"error": f"Unsupported video format: {ext}"}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_VIDEO_SIZE:
        return jsonify({"error": "File too large (max 200MB)"}), 413
    if len(file_bytes) < 1000:
        return jsonify({"error": "File too small or empty"}), 400

    try:
        raw_results = analyze_video(file_bytes, file.filename)
        methods = raw_results.get("methods", {})
        frame_previews = raw_results.get("frame_previews", [])

        ensemble = compute_ensemble(methods, "video")

        return jsonify({
            "status": "success",
            "modality": "video",
            "filename": file.filename,
            "file_size_mb": round(len(file_bytes) / (1024 * 1024), 2),
            "frame_count": raw_results.get("frame_count", 0),
            "fps": raw_results.get("fps", 0),
            "duration": raw_results.get("duration", 0),
            "methods": methods,
            "ensemble": ensemble,
            "frame_previews": frame_previews,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────
# Audio Analysis Endpoint
# ─────────────────────────────────────────────────────────────────
@app.route("/api/analyze/audio", methods=["POST"])
def analyze_audio_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        return jsonify({"error": f"Unsupported audio format: {ext}"}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_AUDIO_SIZE:
        return jsonify({"error": "File too large (max 50MB)"}), 413
    if len(file_bytes) < 100:
        return jsonify({"error": "File too small or empty"}), 400

    try:
        raw_results = analyze_audio(file_bytes, file.filename)
        methods = raw_results.get("methods", {})

        ensemble = compute_ensemble(methods, "audio")

        return jsonify({
            "status": "success",
            "modality": "audio",
            "filename": file.filename,
            "file_size_kb": round(len(file_bytes) / 1024, 1),
            "duration": raw_results.get("duration", 0),
            "sample_rate": raw_results.get("sample_rate", 0),
            "methods": methods,
            "ensemble": ensemble,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None
    print("=" * 60)
    print("  [*] DeepFake Detection API -- Starting...")
    print("  [>] Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
