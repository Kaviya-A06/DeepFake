"""
Image DeepFake Analyzer - 10 Detection Methods
Analyzes images for signs of AI generation or manipulation.
"""

import io
import os
import math
import struct
import hashlib
import base64
import tempfile
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ExifTags
from PIL.ExifTags import TAGS
import cv2
from scipy import ndimage, stats
from scipy.fft import fft2, fftshift


def analyze_image(file_bytes: bytes, filename: str) -> dict:
    """Run all 10 detection methods on the image and return results."""
    results = {}
    ela_image_b64 = None

    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        cv_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Run all methods
        results["ela"] = _ela_analysis(file_bytes, pil_img)
        ela_image_b64 = results["ela"].pop("ela_image", None)

        results["noise_residual"] = _noise_residual_analysis(cv_img)
        results["dct_frequency"] = _dct_frequency_analysis(cv_img)
        results["metadata_forensics"] = _metadata_forensics(file_bytes, pil_img, filename)
        results["color_space"] = _color_space_analysis(cv_img)
        results["copy_move"] = _copy_move_detection(cv_img)
        results["double_jpeg"] = _double_jpeg_detection(file_bytes, pil_img)
        results["edge_sharpness"] = _edge_sharpness_analysis(cv_img)
        results["gan_fingerprint"] = _gan_fingerprint_detection(cv_img)
        results["prnu_analysis"] = _prnu_analysis(cv_img)

    except Exception as e:
        results["error"] = str(e)

    return {"methods": results, "ela_image": ela_image_b64}


# ─────────────────────────────────────────────────────────────────
# METHOD 1: Error Level Analysis (ELA)
# ─────────────────────────────────────────────────────────────────
def _ela_analysis(file_bytes: bytes, pil_img: Image.Image) -> dict:
    """
    ELA: Save image at known JPEG quality, compute difference.
    Manipulated regions typically show higher error levels.
    """
    try:
        original = pil_img.convert("RGB")
        buffer = io.BytesIO()
        original.save(buffer, "JPEG", quality=90)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        diff = ImageChops.difference(original, recompressed)
        diff_arr = np.array(diff).astype(np.float32)

        # Amplify for visualization
        scale = 10
        ela_amplified = np.clip(diff_arr * scale, 0, 255).astype(np.uint8)
        ela_pil = Image.fromarray(ela_amplified)

        # Encode ELA image as base64
        ela_buf = io.BytesIO()
        ela_pil.save(ela_buf, "PNG")
        ela_b64 = base64.b64encode(ela_buf.getvalue()).decode("utf-8")

        # Score: high mean ELA across image = likely manipulated
        mean_ela = float(np.mean(diff_arr))
        std_ela = float(np.std(diff_arr))
        max_ela = float(np.max(diff_arr))

        # Normalize: 0 = authentic, 1 = fake
        score = min(1.0, (mean_ela / 12.0) * 0.5 + (std_ela / 15.0) * 0.3 + (max_ela / 100.0) * 0.2)
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "score": score,
            "mean_ela": round(mean_ela, 3),
            "std_ela": round(std_ela, 3),
            "interpretation": _interpret_score(score),
            "ela_image": ela_b64,
            "description": "Error Level Analysis reveals regions with inconsistent compression artifacts suggesting tampering."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 2: Noise Residual Analysis
# ─────────────────────────────────────────────────────────────────
def _noise_residual_analysis(cv_img: np.ndarray) -> dict:
    """
    Subtract Gaussian-blurred image from original to extract sensor noise.
    GAN-generated images lack consistent camera sensor noise patterns.
    """
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred

        # Compute local variance in blocks
        h, w = noise.shape
        block_size = 32
        variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise[y:y+block_size, x:x+block_size]
                variances.append(float(np.var(block)))

        if not variances:
            return {"score": 0.0, "interpretation": "Unknown"}

        var_arr = np.array(variances)
        # Real cameras: consistent noise variance. GANs: too-uniform or too-varied noise.
        noise_std = float(np.std(var_arr))
        noise_mean = float(np.mean(var_arr))
        coefficient_of_variation = noise_std / (noise_mean + 1e-9)

        # Very uniform noise (low CV) or extreme variation both suggest fake
        if coefficient_of_variation < 0.3:
            score = 0.55 + (0.3 - coefficient_of_variation) * 1.5
        elif coefficient_of_variation > 2.5:
            score = 0.5 + min(0.5, (coefficient_of_variation - 2.5) * 0.1)
        else:
            score = max(0.05, 0.4 - coefficient_of_variation * 0.1)

        score = float(np.clip(score, 0.0, 1.0))
        return {
            "score": score,
            "noise_mean": round(noise_mean, 4),
            "noise_std": round(noise_std, 4),
            "coefficient_of_variation": round(coefficient_of_variation, 4),
            "interpretation": _interpret_score(score),
            "description": "Noise residual analysis detects unnatural or missing camera sensor noise patterns."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 3: DCT Frequency Domain Analysis
# ─────────────────────────────────────────────────────────────────
def _dct_frequency_analysis(cv_img: np.ndarray) -> dict:
    """
    2D FFT of image. GANs produce characteristic spectral peaks
    and unnatural frequency distributions.
    """
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude = np.abs(fshift)
        log_magnitude = np.log1p(magnitude)

        h, w = log_magnitude.shape
        cy, cx = h // 2, w // 2

        # Compute energy in different frequency bands
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dist = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
        max_dist = min(cy, cx)

        low_mask = dist < max_dist * 0.1
        mid_mask = (dist >= max_dist * 0.1) & (dist < max_dist * 0.5)
        high_mask = dist >= max_dist * 0.5

        low_energy = float(np.mean(log_magnitude[low_mask]))
        mid_energy = float(np.mean(log_magnitude[mid_mask]))
        high_energy = float(np.mean(log_magnitude[high_mask]))

        # GAN images: abnormal high-freq energy / spectral checkerboard
        ratio = high_energy / (low_energy + 1e-9)

        # Detect spectral peaks (GAN checkerboard artifacts)
        normalized = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-9)
        threshold = 0.92
        peak_count = int(np.sum(normalized > threshold))
        image_size = h * w
        peak_density = peak_count / image_size

        gfake_score = min(1.0, peak_density * 5000 + max(0, ratio - 0.15) * 2)
        score = float(np.clip(gfake_score, 0.0, 1.0))

        return {
            "score": score,
            "low_energy": round(low_energy, 4),
            "mid_energy": round(mid_energy, 4),
            "high_energy": round(high_energy, 4),
            "spectral_peak_density": round(peak_density * 10000, 4),
            "interpretation": _interpret_score(score),
            "description": "DCT frequency analysis detects GAN-specific spectral artifacts and abnormal frequency distributions."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 4: Metadata Forensics
# ─────────────────────────────────────────────────────────────────
def _metadata_forensics(file_bytes: bytes, pil_img: Image.Image, filename: str) -> dict:
    """
    Analyze EXIF metadata for inconsistencies suggesting AI generation.
    """
    try:
        suspicion_score = 0.0
        flags = []
        exif_data = {}

        try:
            raw_exif = pil_img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    exif_data[tag] = str(value)[:100]
        except Exception:
            pass

        # No EXIF at all: suspicious for photos
        if not exif_data:
            suspicion_score += 0.35
            flags.append("No EXIF metadata found — AI-generated images often lack EXIF.")

        # Check for AI software markers
        ai_software_keywords = ["midjourney", "stable diffusion", "dall-e", "dall·e",
                                "adobe firefly", "diffusion", "gan", "generated", "ai", "artificial"]
        software = exif_data.get("Software", "").lower()
        if any(kw in software for kw in ai_software_keywords):
            suspicion_score += 0.55
            flags.append(f"AI-related software tag detected: {software}")

        # Missing camera-specific tags
        camera_tags = ["Make", "Model", "FocalLength", "ExposureTime", "ISOSpeedRatings", "FNumber"]
        missing_camera = [t for t in camera_tags if t not in exif_data]
        if len(missing_camera) >= 4:
            suspicion_score += 0.25
            flags.append(f"Missing camera metadata: {', '.join(missing_camera)}")

        # GPS present but no camera model (unusual)
        if "GPSInfo" in exif_data and "Make" not in exif_data:
            suspicion_score += 0.15
            flags.append("GPS data present but no camera model.")

        # PNG files rarely have EXIF (less suspicious)
        if filename.lower().endswith(".png") and not exif_data:
            suspicion_score -= 0.2

        score = float(np.clip(suspicion_score, 0.0, 1.0))
        return {
            "score": score,
            "flags": flags,
            "exif_fields_found": len(exif_data),
            "interpretation": _interpret_score(score),
            "description": "Metadata forensics inspects EXIF data for AI software traces, missing camera info, and inconsistencies."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 5: Color Space Analysis
# ─────────────────────────────────────────────────────────────────
def _color_space_analysis(cv_img: np.ndarray) -> dict:
    """
    Analyze YCbCr color distribution. GAN outputs show abnormal chroma
    sub-sampling patterns and color channel correlations.
    """
    try:
        ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # Compute histogram for each channel
        score_components = []

        for channel, name in [(Y, 'Y'), (Cr, 'Cr'), (Cb, 'Cb')]:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-9)

            # Entropy of the channel
            entropy = stats.entropy(hist + 1e-9)
            # Real photos: Y entropy ~5-7, Cb/Cr ~3-5
            # GAN: may have very high or very low entropy

        # Chroma correlation (Cr vs Cb)
        cr_flat = Cr.flatten().astype(np.float32)
        cb_flat = Cb.flatten().astype(np.float32)

        correlation = float(np.corrcoef(cr_flat, cb_flat)[0, 1])
        abs_corr = abs(correlation)

        # Very high chroma correlation is suspicious (GAN artifacts)
        if abs_corr > 0.85:
            chroma_score = (abs_corr - 0.85) * 6.67
        else:
            chroma_score = 0.05

        # Check for banding (very few unique values in Cb/Cr after quantization)
        cb_unique = len(np.unique((Cb // 8).flatten()))
        cr_unique = len(np.unique((Cr // 8).flatten()))
        banding_ratio = 1.0 - min(1.0, (cb_unique + cr_unique) / 64.0)
        banding_score = banding_ratio * 0.3

        score = float(np.clip(chroma_score * 0.6 + banding_score * 0.4, 0.0, 1.0))
        return {
            "score": score,
            "chroma_correlation": round(correlation, 4),
            "cb_unique_values": cb_unique,
            "cr_unique_values": cr_unique,
            "interpretation": _interpret_score(score),
            "description": "Color space analysis identifies abnormal chroma correlations and banding patterns common in GAN outputs."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 6: Copy-Move Forgery Detection
# ─────────────────────────────────────────────────────────────────
def _copy_move_detection(cv_img: np.ndarray) -> dict:
    """
    Detect copy-move forgery using ORB feature matching.
    Duplicated regions indicate content manipulation.
    """
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # Resize if too large
        h, w = gray.shape
        if h > 1080 or w > 1080:
            scale = min(1080/h, 1080/w)
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)))

        # ORB detector
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < 20:
            return {
                "score": 0.1,
                "duplicate_regions": 0,
                "interpretation": _interpret_score(0.1),
                "description": "Copy-move forgery detection using ORB feature matching to find duplicated image regions."
            }

        # Brute force matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors, descriptors)

        # Find self-matches that are not trivially close
        suspicious = 0
        min_dist_px = max(gray.shape[0], gray.shape[1]) * 0.05  # 5% of image size
        for m in matches:
            if m.queryIdx == m.trainIdx:
                continue
            pt1 = keypoints[m.queryIdx].pt
            pt2 = keypoints[m.trainIdx].pt
            dist = math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            if dist > min_dist_px and m.distance < 40:
                suspicious += 1

        # Score based on suspicious matches
        match_ratio = suspicious / max(len(keypoints), 1)
        score = float(np.clip(match_ratio * 10, 0.0, 1.0))

        return {
            "score": score,
            "duplicate_regions": suspicious,
            "keypoints_found": len(keypoints),
            "interpretation": _interpret_score(score),
            "description": "Copy-move forgery detection using ORB feature matching to find duplicated image regions."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 7: Double JPEG Compression Detection
# ─────────────────────────────────────────────────────────────────
def _double_jpeg_detection(file_bytes: bytes, pil_img: Image.Image) -> dict:
    """
    Double compression leaves artifacts in DCT coefficient histograms.
    GAN images often show no double-compression artifacts (too clean).
    """
    try:
        img = pil_img.convert("RGB")
        gray = np.array(img.convert("L"), dtype=np.float32)

        # Apply DCT to 8x8 blocks like JPEG
        h, w = gray.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        gray = gray[:h8, :w8]

        dct_coeffs = []
        for y in range(0, h8, 8):
            for x in range(0, w8, 8):
                block = gray[y:y+8, x:x+8]
                dct_block = cv2.dct(block)
                dct_coeffs.extend(dct_block[0:3, 0:3].flatten().tolist())

        dct_arr = np.array(dct_coeffs)
        hist, _ = np.histogram(dct_arr, bins=50, range=(-50, 50))
        hist = hist.astype(np.float32)

        # Detect periodicity in DCT histogram (sign of double compression)
        # Double-compressed JPEGs show dips at multiples of quantization step
        # Simple variance measure: very smooth = single compression; periodic dips = double
        hist_norm = hist / (hist.sum() + 1e-9)
        entropy = float(stats.entropy(hist_norm + 1e-9))

        # Very high entropy DCT hist = GAN (unnaturally smooth distribution)
        max_entropy = math.log(50)
        normalized_entropy = entropy / max_entropy
        score = float(np.clip(normalized_entropy - 0.5, 0.0, 0.5) * 2.0)

        return {
            "score": score,
            "dct_histogram_entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "interpretation": _interpret_score(score),
            "description": "Double JPEG compression detection via DCT coefficient histogram analysis."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 8: Edge Sharpness Inconsistency
# ─────────────────────────────────────────────────────────────────
def _edge_sharpness_analysis(cv_img: np.ndarray) -> dict:
    """
    Analyze spatial distribution of sharpness across image regions.
    GAN-generated faces are often uniformly sharp without the depth-of-field
    blur gradient seen in real photos.
    """
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Compute Laplacian variance in grid cells
        grid_rows, grid_cols = 4, 4
        cell_h, cell_w = h // grid_rows, w // grid_cols
        sharpness_map = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                y1, y2 = r * cell_h, (r+1) * cell_h
                x1, x2 = c * cell_w, (c+1) * cell_w
                cell = gray[y1:y2, x1:x2]
                lap_var = float(cv2.Laplacian(cell, cv2.CV_64F).var())
                sharpness_map.append(lap_var)

        sharpness_arr = np.array(sharpness_map)
        global_sharpness = float(np.mean(sharpness_arr))
        sharpness_std = float(np.std(sharpness_arr))
        cv_sharpness = sharpness_std / (global_sharpness + 1e-9)

        # Real photos have high variance (focus falloff); GANs have uniform sharpness
        if cv_sharpness < 0.4:
            score = (0.4 - cv_sharpness) * 2.0  # Abnormally uniform
        elif cv_sharpness > 4.0:
            score = min(0.5, (cv_sharpness - 4.0) * 0.05)  # Abnormally spiky
        else:
            score = max(0.05, 0.3 - cv_sharpness * 0.05)

        score = float(np.clip(score, 0.0, 1.0))
        return {
            "score": score,
            "mean_sharpness": round(global_sharpness, 4),
            "sharpness_std": round(sharpness_std, 4),
            "uniformity_cv": round(cv_sharpness, 4),
            "interpretation": _interpret_score(score),
            "description": "Edge sharpness analysis detects unnaturally uniform focus distribution across the image."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 9: GAN Fingerprint / Checkerboard Artifact Detection
# ─────────────────────────────────────────────────────────────────
def _gan_fingerprint_detection(cv_img: np.ndarray) -> dict:
    """
    Detect pixel-level periodic artifacts created by transposed convolutions
    in GAN upsampling. These appear as checkerboard patterns in frequency domain.
    """
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # Compute 2D FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        spectrum = np.log1p(np.abs(fshift))

        # Analyze periodicity at N/2, N/4, N/8 frequencies (GAN upsampling artifacts)
        cy, cx = h // 2, w // 2
        artifact_scores = []

        for stride in [2, 4, 8]:
            # Check energy at aliased frequencies
            fy_positions = [cy + h // stride, cy - h // stride] if h > stride * 2 else [cy]
            fx_positions = [cx + w // stride, cx - w // stride] if w > stride * 2 else [cx]

            region_energy = []
            for fy in fy_positions:
                for fx in fx_positions:
                    fy_c = np.clip(fy, 5, h-5)
                    fx_c = np.clip(fx, 5, w-5)
                    neighborhood = spectrum[fy_c-3:fy_c+3, fx_c-3:fx_c+3]
                    region_energy.append(float(np.mean(neighborhood)))

            center_energy = float(np.mean(spectrum[cy-10:cy+10, cx-10:cx+10]))
            if center_energy > 0:
                ratio = np.mean(region_energy) / center_energy
                artifact_scores.append(ratio)

        if artifact_scores:
            mean_ratio = float(np.mean(artifact_scores))
            # High ratio at aliased frequencies = GAN checkerboard
            score = float(np.clip((mean_ratio - 0.1) * 5, 0.0, 1.0))
        else:
            score = 0.0

        # Also check for exact periodic patterns using autocorrelation
        small = cv2.resize(gray, (128, 128))
        autocorr = np.real(np.fft.ifft2(np.abs(np.fft.fft2(small))**2))
        autocorr_norm = autocorr / (autocorr[0, 0] + 1e-9)
        # Secondary peaks in autocorrelation indicate periodicity
        secondary = autocorr_norm.flatten()
        secondary_sorted = np.sort(secondary)[::-1]
        periodic_peak = float(secondary_sorted[10]) if len(secondary_sorted) > 10 else 0.0

        final_score = float(np.clip(score * 0.6 + min(0.4, periodic_peak * 0.4), 0.0, 1.0))
        return {
            "score": final_score,
            "spectral_ratio": round(mean_ratio if artifact_scores else 0.0, 4),
            "periodic_peak": round(periodic_peak, 4),
            "interpretation": _interpret_score(final_score),
            "description": "GAN fingerprint detection identifies checkerboard artifacts from transposed convolution upsampling."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 10: PRNU Analysis (Photo Response Non-Uniformity)
# ─────────────────────────────────────────────────────────────────
def _prnu_analysis(cv_img: np.ndarray) -> dict:
    """
    PRNU is a unique camera sensor fingerprint. AI-generated images
    either completely lack PRNU or have artificially injected patterns.
    """
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Extract noise residual (simplified PRNU estimation)
        denoised = cv2.fastNlMeansDenoising(cv_img, h=10)
        denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = gray - denoised_gray

        # Normalize residual
        residual_norm = residual / (gray + 1e-9)

        # Compute spatial correlation of residual
        h, w = residual_norm.shape
        center_region = residual_norm[h//4:3*h//4, w//4:3*w//4]

        # Measure uniformity of PRNU-like residual
        prnu_std = float(np.std(residual_norm))
        prnu_mean = float(np.mean(np.abs(residual_norm)))
        spatial_autocorr = float(np.corrcoef(
            center_region[:h//4, :].flatten(),
            center_region[h//4:h//2, :].flatten()
        )[0, 1]) if center_region.shape[0] >= h // 2 else 0.0

        # Real cameras: strong positive PRNU autocorrelation across subregions
        # Fake: near-zero or negative autocorrelation
        prnu_absent_score = max(0.0, 1.0 - abs(spatial_autocorr) * 2)
        noise_homogeneity = 1.0 - min(1.0, prnu_std * 10)
        score = float(np.clip(prnu_absent_score * 0.6 + noise_homogeneity * 0.4, 0.0, 1.0))

        return {
            "score": score,
            "prnu_std": round(prnu_std, 6),
            "prnu_mean": round(prnu_mean, 6),
            "spatial_autocorrelation": round(spatial_autocorr, 4),
            "interpretation": _interpret_score(score),
            "description": "PRNU analysis checks for camera sensor fingerprint — absent in AI-generated imagery."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
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
