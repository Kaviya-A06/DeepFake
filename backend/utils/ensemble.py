"""
Ensemble Scorer — combines all method scores into a final verdict.
"""

import numpy as np


# Weights for each method within their modality
IMAGE_WEIGHTS = {
    "ela": 0.18,
    "noise_residual": 0.12,
    "dct_frequency": 0.12,
    "metadata_forensics": 0.10,
    "color_space": 0.08,
    "copy_move": 0.10,
    "double_jpeg": 0.08,
    "edge_sharpness": 0.10,
    "gan_fingerprint": 0.12,
    "prnu_analysis": 0.10,
}

VIDEO_WEIGHTS = {
    "frame_ela": 0.18,
    "temporal_consistency": 0.18,
    "facial_landmark_stability": 0.15,
    "eye_blink_pattern": 0.12,
    "compression_artifact": 0.10,
    "pixel_temporal_coherence": 0.10,
    "head_pose_estimation": 0.10,
    "color_temporal_stability": 0.07,
}

AUDIO_WEIGHTS = {
    "mfcc_analysis": 0.20,
    "spectrogram_artifacts": 0.18,
    "pitch_continuity": 0.16,
    "background_noise": 0.14,
    "silence_pattern": 0.12,
    "formant_analysis": 0.12,
    "phase_coherence": 0.08,
}

MODALITY_WEIGHTS = IMAGE_WEIGHTS, VIDEO_WEIGHTS, AUDIO_WEIGHTS


def compute_ensemble(methods: dict, modality: str) -> dict:
    """
    Compute weighted ensemble score from individual method scores.

    Args:
        methods: dict of {method_name: {score: float, ...}}
        modality: "image" | "video" | "audio"

    Returns:
        dict with ensemble metrics
    """
    weight_map = {"image": IMAGE_WEIGHTS, "video": VIDEO_WEIGHTS, "audio": AUDIO_WEIGHTS}
    weights = weight_map.get(modality, IMAGE_WEIGHTS)

    total_weight = 0.0
    weighted_sum = 0.0
    method_scores = {}

    for method_name, method_result in methods.items():
        if isinstance(method_result, dict) and "score" in method_result:
            score = float(method_result["score"])
            weight = weights.get(method_name, 0.1)
            weighted_sum += score * weight
            total_weight += weight
            method_scores[method_name] = score

    if total_weight <= 0:
        ensemble_score = 0.0
    else:
        ensemble_score = weighted_sum / total_weight

    ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))

    # Boost: if multiple methods agree strongly
    high_score_methods = sum(1 for s in method_scores.values() if s >= 0.65)
    if high_score_methods >= 3:
        boost = min(0.15, high_score_methods * 0.03)
        ensemble_score = float(np.clip(ensemble_score + boost, 0.0, 1.0))

    # Risk level
    if ensemble_score < 0.25:
        risk_level = "LOW"
        verdict = "AUTHENTIC"
        verdict_desc = "This content appears to be genuine. No significant deepfake indicators were detected."
        color = "#00d4aa"
    elif ensemble_score < 0.50:
        risk_level = "MEDIUM"
        verdict = "POSSIBLY AUTHENTIC"
        verdict_desc = "Some minor anomalies detected. Content is likely authentic but warrants review."
        color = "#f5c518"
    elif ensemble_score < 0.70:
        risk_level = "HIGH"
        verdict = "SUSPICIOUS"
        verdict_desc = "Multiple deepfake indicators detected. Treat this content with significant caution."
        color = "#ff8c00"
    elif ensemble_score < 0.85:
        risk_level = "CRITICAL"
        verdict = "LIKELY FAKE"
        verdict_desc = "Strong evidence of AI generation or manipulation. This content is very likely a deepfake."
        color = "#ff3860"
    else:
        risk_level = "CRITICAL"
        verdict = "DEEPFAKE DETECTED"
        verdict_desc = "Overwhelming evidence of AI manipulation. This content is almost certainly a deepfake."
        color = "#ff0055"

    # Fake probability percentage
    fake_probability = round(ensemble_score * 100, 1)
    authentic_probability = round((1 - ensemble_score) * 100, 1)

    return {
        "ensemble_score": round(ensemble_score, 4),
        "fake_probability": fake_probability,
        "authentic_probability": authentic_probability,
        "risk_level": risk_level,
        "verdict": verdict,
        "verdict_description": verdict_desc,
        "verdict_color": color,
        "methods_flagged": high_score_methods,
        "total_methods": len(method_scores),
        "method_scores": {k: round(v * 100, 1) for k, v in method_scores.items()},
    }
