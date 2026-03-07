"""
Video DeepFake Analyzer - 8 Detection Methods
Analyzes videos for temporal, facial, and pixel-level deepfake artifacts.
"""

import io
import os
import math
import base64
import tempfile
import numpy as np
import cv2
from scipy import stats
from PIL import Image


def analyze_video(file_bytes: bytes, filename: str) -> dict:
    """Run all 8 detection methods on the video and return results."""
    results = {}
    frame_previews = []

    try:
        # Save to temp file for OpenCV
        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"error": "Could not open video file.", "methods": {}}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Sample frames evenly (max 30 frames for performance)
        sample_count = min(30, max(5, total_frames))
        sample_interval = max(1, total_frames // sample_count)

        frames = []
        frame_indices = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_interval == 0:
                frames.append(frame)
                frame_indices.append(idx)
            idx += 1
        cap.release()

        if len(frames) < 2:
            return {"error": "Too few frames to analyze.", "methods": {}}

        # Generate preview of first frame
        if frames:
            preview_frame = cv2.resize(frames[0], (320, 240))
            _, buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_previews.append(base64.b64encode(buf).decode("utf-8"))

        # Run all methods
        results["frame_ela"] = _frame_ela_analysis(frames)
        results["temporal_consistency"] = _temporal_consistency(frames)
        results["facial_landmark_stability"] = _facial_landmark_stability(frames)
        results["eye_blink_pattern"] = _eye_blink_pattern(frames, fps)
        results["compression_artifact"] = _compression_artifact_anomaly(frames)
        results["pixel_temporal_coherence"] = _pixel_temporal_coherence(frames)
        results["head_pose_estimation"] = _head_pose_estimation(frames)
        results["color_temporal_stability"] = _color_temporal_stability(frames)

        # Clean up
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    except Exception as e:
        results["error"] = str(e)

    return {
        "methods": results,
        "frame_count": len(frames) if 'frames' in dir() else 0,
        "frame_previews": frame_previews,
        "fps": round(fps, 2) if 'fps' in dir() else 0,
        "duration": round(duration, 2) if 'duration' in dir() else 0
    }


# ─────────────────────────────────────────────────────────────────
# METHOD 1: Frame-Level ELA
# ─────────────────────────────────────────────────────────────────
def _frame_ela_analysis(frames: list) -> dict:
    """Apply ELA to sampled frames and compute average/variance."""
    try:
        ela_scores = []
        for frame in frames[::max(1, len(frames)//10)]:  # Max 10 frames
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_img.save(buffer, "JPEG", quality=90)
            buffer.seek(0)
            recompressed = Image.open(buffer).convert("RGB")
            diff = np.abs(np.array(pil_img).astype(float) - np.array(recompressed).astype(float))
            ela_scores.append(float(np.mean(diff)))

        mean_ela = float(np.mean(ela_scores))
        std_ela = float(np.std(ela_scores))
        score = float(np.clip((mean_ela / 10.0) * 0.6 + (std_ela / 5.0) * 0.4, 0.0, 1.0))

        return {
            "score": score,
            "mean_ela": round(mean_ela, 4),
            "std_ela": round(std_ela, 4),
            "frames_analyzed": len(ela_scores),
            "interpretation": _interpret_score(score),
            "description": "ELA applied across video frames detects compression inconsistencies from face-swap editing."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 2: Temporal Consistency (Optical Flow)
# ─────────────────────────────────────────────────────────────────
def _temporal_consistency(frames: list) -> dict:
    """
    Compute optical flow between consecutive frames.
    Deepfakes often show unnatural motion discontinuities.
    """
    try:
        flow_magnitudes = []
        inconsistencies = 0

        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            g1 = cv2.resize(g1, (64, 64))
            g2 = cv2.resize(g2, (64, 64))

            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
            flow_magnitudes.append(mag)

        if len(flow_magnitudes) < 2:
            return {"score": 0.1, "interpretation": "Insufficient frames", "description": ""}

        mag_arr = np.array(flow_magnitudes)
        mean_flow = float(np.mean(mag_arr))
        std_flow = float(np.std(mag_arr))

        # Detect sudden spikes in flow (frame discontinuities)
        if mean_flow > 0:
            z_scores = np.abs((mag_arr - mean_flow) / (std_flow + 1e-9))
            spike_count = int(np.sum(z_scores > 3.0))
        else:
            spike_count = 0

        # Deepfakes often show periodic spikes at transitions
        spike_ratio = spike_count / max(len(flow_magnitudes), 1)
        score = float(np.clip(spike_ratio * 5 + (std_flow / (mean_flow + 1e-9)) * 0.1, 0.0, 1.0))

        return {
            "score": score,
            "mean_flow": round(mean_flow, 4),
            "std_flow": round(std_flow, 4),
            "motion_discontinuities": spike_count,
            "interpretation": _interpret_score(score),
            "description": "Optical flow temporal consistency analysis detects unnatural motion discontinuities in deepfakes."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 3: Facial Landmark Stability
# ─────────────────────────────────────────────────────────────────
def _facial_landmark_stability(frames: list) -> dict:
    """
    Track face bounding box across frames.
    Face-swap deepfakes show jitter and instability in face region.
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face_positions = []
        faces_found = 0

        for frame in frames:
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cx = x + w / 2
                cy = y + h / 2
                face_positions.append((cx, cy, w, h))
                faces_found += 1

        if len(face_positions) < 3:
            return {
                "score": 0.15,
                "faces_detected": faces_found,
                "interpretation": "Not enough faces for tracking.",
                "description": "Facial landmark stability analysis — too few faces detected."
            }

        positions = np.array(face_positions)
        cx_arr = positions[:, 0]
        cy_arr = positions[:, 1]
        w_arr = positions[:, 2]

        # Jitter: per-frame displacement
        dx = np.diff(cx_arr)
        dy = np.diff(cy_arr)
        jitter = float(np.std(np.sqrt(dx**2 + dy**2)))

        # Size jitter (deepfakes may have wobbling face size)
        size_jitter = float(np.std(w_arr) / (np.mean(w_arr) + 1e-9))

        # High jitter = deepfake
        jitter_score = min(1.0, jitter / 10.0)
        size_score = min(1.0, size_jitter * 5.0)
        score = float(np.clip(jitter_score * 0.6 + size_score * 0.4, 0.0, 1.0))

        return {
            "score": score,
            "faces_detected": faces_found,
            "position_jitter_px": round(jitter, 4),
            "size_variation": round(size_jitter, 4),
            "interpretation": _interpret_score(score),
            "description": "Facial region jitter analysis — deepfakes show instability in face position and size across frames."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 4: Eye Blink Pattern Analysis
# ─────────────────────────────────────────────────────────────────
def _eye_blink_pattern(frames: list, fps: float) -> dict:
    """
    Approximate eye aspect ratio (EAR) using pixel intensity in eye region.
    GANs often miss natural blink cadence (avg 15-20 blinks/min).
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eye_openness_values = []

        for frame in frames:
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_roi, 1.05, 3)

                if len(eyes) > 0:
                    ex, ey, ew, eh = eyes[0]
                    eye_region = face_roi[ey:ey+eh, ex:ex+ew]
                    # Eye openness approximated by variance in eye region
                    openness = float(np.var(eye_region.astype(np.float32)))
                    eye_openness_values.append(openness)

        if len(eye_openness_values) < 4:
            return {
                "score": 0.2,
                "blinks_detected": 0,
                "interpretation": "Not enough eye data.",
                "description": "Eye blink pattern analysis — insufficient eye detections for analysis."
            }

        openness_arr = np.array(eye_openness_values)
        mean_openness = float(np.mean(openness_arr))
        std_openness = float(np.std(openness_arr))

        # Detect blink-like dips (sudden drops in openness)
        threshold = mean_openness - std_openness
        blink_candidates = 0
        below = False
        for v in openness_arr:
            if v < threshold:
                if not below:
                    blink_candidates += 1
                    below = True
            else:
                below = False

        # Expected blink rate: ~0.25/sec (15/min)
        duration_sec = len(frames) / (fps + 1e-9)
        expected_blinks = max(1, duration_sec * 0.25)
        blink_ratio = blink_candidates / expected_blinks

        # Too few or too many blinks are suspicious
        if blink_ratio < 0.1 or blink_ratio > 5:
            score = min(1.0, abs(blink_ratio - 1.0) * 0.3)
        else:
            score = max(0.05, 0.2 - blink_ratio * 0.05)

        score = float(np.clip(score, 0.0, 1.0))
        return {
            "score": score,
            "blinks_detected": blink_candidates,
            "expected_blinks": round(expected_blinks, 1),
            "blink_ratio": round(blink_ratio, 3),
            "interpretation": _interpret_score(score),
            "description": "Eye blink cadence analysis — GANs often fail to generate realistic blinking patterns."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 5: Compression Artifact Anomaly
# ─────────────────────────────────────────────────────────────────
def _compression_artifact_anomaly(frames: list) -> dict:
    """
    Detect codec-inconsistent macroblocking patterns.
    Face-swap regions often have different compression history.
    """
    try:
        dct_variances = []

        for frame in frames[::max(1, len(frames)//10)]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            block_vars = []

            for y in range(0, h - 8, 16):
                for x in range(0, w - 8, 16):
                    block = gray[y:y+8, x:x+8]
                    dct_block = cv2.dct(block)
                    block_vars.append(float(np.var(dct_block[2:, 2:])))

            if block_vars:
                dct_variances.append(float(np.mean(block_vars)))

        if not dct_variances:
            return {"score": 0.1, "interpretation": "Unknown", "description": ""}

        variance_arr = np.array(dct_variances)
        mean_var = float(np.mean(variance_arr))
        std_var = float(np.std(variance_arr))
        cv = std_var / (mean_var + 1e-9)

        # High inter-frame variance in DCT coefficients = inconsistent compression
        score = float(np.clip(cv * 0.5, 0.0, 1.0))

        return {
            "score": score,
            "mean_dct_variance": round(mean_var, 4),
            "variance_cv": round(cv, 4),
            "interpretation": _interpret_score(score),
            "description": "DCT compression artifact analysis detects macroblocking inconsistencies from face-swap editing."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 6: Pixel-Level Temporal Coherence
# ─────────────────────────────────────────────────────────────────
def _pixel_temporal_coherence(frames: list) -> dict:
    """
    Compute per-pixel variance over sampled frames.
    Synthetic face regions in deepfakes show unnaturally low temporal variance.
    """
    try:
        resized = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
                   for f in frames[:20]]

        if len(resized) < 3:
            return {"score": 0.1, "interpretation": "Unknown", "description": ""}

        stack = np.stack(resized, axis=0)  # (frames, H, W)
        pixel_var = np.var(stack, axis=0)  # (H, W)

        mean_var = float(np.mean(pixel_var))
        std_var = float(np.std(pixel_var))
        cv_var = std_var / (mean_var + 1e-9)

        # Extremely low variance in center (face) region vs edges = fake
        h, w = pixel_var.shape
        center_var = float(np.mean(pixel_var[h//4:3*h//4, w//4:3*w//4]))
        edge_var = float(np.mean(pixel_var) - center_var * 0.5)

        if edge_var > 0:
            ratio = center_var / (edge_var + 1e-9)
            score = float(np.clip((1.0 - ratio) * 0.5, 0.0, 1.0)) if ratio < 1 else 0.1
        else:
            score = 0.1

        return {
            "score": score,
            "mean_pixel_variance": round(mean_var, 4),
            "center_to_edge_ratio": round(ratio if 'ratio' in dir() else 1.0, 4),
            "interpretation": _interpret_score(score),
            "description": "Pixel temporal coherence detects unnaturally static face regions common in deepfakes."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 7: Head Pose Estimation
# ─────────────────────────────────────────────────────────────────
def _head_pose_estimation(frames: list) -> dict:
    """
    Track approximate head orientation using face bounding box asymmetry.
    Deepfakes often show sudden, unrealistic head rotation changes.
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        aspect_ratios = []

        for frame in frames:
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                aspect_ratios.append(w / (h + 1e-9))

        if len(aspect_ratios) < 3:
            return {
                "score": 0.15,
                "interpretation": "Not enough face detections.",
                "description": "Head pose estimation — insufficient face detections."
            }

        ar_arr = np.array(aspect_ratios)
        ar_std = float(np.std(ar_arr))
        ar_mean = float(np.mean(ar_arr))

        # Sudden changes in ratio = unnatural head movement
        ar_diffs = np.abs(np.diff(ar_arr))
        spike_threshold = ar_std * 2
        pose_spikes = int(np.sum(ar_diffs > spike_threshold))

        score = float(np.clip(pose_spikes / max(len(ar_diffs), 1) * 3, 0.0, 1.0))

        return {
            "score": score,
            "mean_aspect_ratio": round(ar_mean, 4),
            "aspect_ratio_std": round(ar_std, 4),
            "pose_discontinuities": pose_spikes,
            "interpretation": _interpret_score(score),
            "description": "Head pose estimation detects unnatural face orientation changes from deepfake stitching."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 8: Color Temporal Stability
# ─────────────────────────────────────────────────────────────────
def _color_temporal_stability(frames: list) -> dict:
    """
    Track mean color of face region across frames.
    Deepfakes may show skin tone inconsistencies over time.
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face_means = []

        for frame in frames:
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = small[y:y+h, x:x+w]
                hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                face_means.append((
                    float(np.mean(hsv_face[:, :, 0])),  # Hue
                    float(np.mean(hsv_face[:, :, 1])),  # Saturation
                    float(np.mean(hsv_face[:, :, 2]))   # Value
                ))

        if len(face_means) < 3:
            return {
                "score": 0.15,
                "interpretation": "Not enough face detections.",
                "description": "Color temporal stability — too few faces."
            }

        face_arr = np.array(face_means)
        hue_std = float(np.std(face_arr[:, 0]))
        sat_std = float(np.std(face_arr[:, 1]))

        # High hue/saturation variability = inconsistent skin rendering = deepfake
        color_instability = (hue_std / 10.0) * 0.5 + (sat_std / 30.0) * 0.5
        score = float(np.clip(color_instability, 0.0, 1.0))

        return {
            "score": score,
            "hue_std": round(hue_std, 4),
            "saturation_std": round(sat_std, 4),
            "interpretation": _interpret_score(score),
            "description": "Skin color temporal stability analysis detects inconsistent face rendering in deepfakes."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


def _interpret_score(score: float) -> str:
    if score < 0.25:
        return "Likely Authentic"
    elif score < 0.50:
        return "Possibly Authentic"
    elif score < 0.70:
        return "Suspicious"
    elif score < 0.85:
        return "Likely Fake"
    else:
        return "Almost Certainly Fake"
