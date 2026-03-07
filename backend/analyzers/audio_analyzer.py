"""
Audio DeepFake Analyzer - 7 Detection Methods
Analyzes audio for AI synthesis artifacts using signal processing.
"""

import io
import os
import math
import tempfile
import numpy as np
from scipy import stats, signal
import soundfile as sf


def analyze_audio(file_bytes: bytes, filename: str) -> dict:
    """Run all 7 detection methods on the audio and return results."""
    results = {}

    try:
        # Save to temp file
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Load audio
        audio_data, sample_rate = sf.read(tmp_path, always_2d=False)

        # Convert stereo to mono for most analyses
        if audio_data.ndim > 1:
            mono = audio_data.mean(axis=1)
            stereo = audio_data
        else:
            mono = audio_data
            stereo = None

        # Normalize
        if np.max(np.abs(mono)) > 0:
            mono_norm = mono / np.max(np.abs(mono))
        else:
            mono_norm = mono

        duration = len(mono) / sample_rate

        results["mfcc_analysis"] = _mfcc_analysis(mono_norm, sample_rate)
        results["spectrogram_artifacts"] = _spectrogram_artifact_detection(mono_norm, sample_rate)
        results["pitch_continuity"] = _pitch_continuity_analysis(mono_norm, sample_rate)
        results["background_noise"] = _background_noise_consistency(mono_norm, sample_rate)
        results["silence_pattern"] = _silence_pattern_analysis(mono_norm, sample_rate)
        results["formant_analysis"] = _formant_analysis(mono_norm, sample_rate)
        results["phase_coherence"] = _phase_coherence_analysis(audio_data, sample_rate, is_stereo=(stereo is not None))

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    except Exception as e:
        results["error"] = str(e)

    return {
        "methods": results,
        "duration": round(duration, 2) if 'duration' in dir() else 0,
        "sample_rate": sample_rate if 'sample_rate' in dir() else 0
    }


# ─────────────────────────────────────────────────────────────────
# METHOD 1: MFCC Statistical Analysis
# ─────────────────────────────────────────────────────────────────
def _mfcc_analysis(mono: np.ndarray, sr: int) -> dict:
    """
    Compute MFCC manually using filterbanks.
    Synthetic speech has abnormally low variance in MFCC coefficients.
    """
    try:
        # Pre-emphasis
        pre_emph = np.append(mono[0], mono[1:] - 0.97 * mono[:-1])

        # Frame the signal
        frame_length = int(0.025 * sr)  # 25ms
        frame_step = int(0.010 * sr)    # 10ms
        frames = []
        for start in range(0, len(pre_emph) - frame_length, frame_step):
            frame = pre_emph[start:start + frame_length]
            frames.append(frame * np.hamming(frame_length))

        if len(frames) < 10:
            return {"score": 0.2, "interpretation": "Too short", "description": ""}

        # FFT of each frame
        NFFT = 512
        n_mels = 26
        n_mfcc = 13

        # Mel filterbank
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((n_mels, NFFT // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-9)
            for k in range(f_m, f_m_plus):
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-9)

        # Compute features
        mfcc_features = []
        for frame in frames[:200]:
            power_spectrum = (1/NFFT) * np.square(np.abs(np.fft.rfft(frame, NFFT)))
            filter_banks = np.dot(fbank, power_spectrum)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
            log_filter_banks = np.log(filter_banks)
            mfcc = np.fft.rfft(log_filter_banks)[:n_mfcc].real
            mfcc_features.append(mfcc)

        mfcc_arr = np.array(mfcc_features)

        # Statistical analysis of MFCC
        mfcc_var = float(np.mean(np.var(mfcc_arr, axis=0)))
        mfcc_delta_std = float(np.mean(np.std(np.diff(mfcc_arr, axis=0), axis=0)))

        # TTS/GAN audio: unnaturally low MFCC variance and delta
        low_var_score = max(0.0, 1.0 - mfcc_var / 15.0)
        low_delta_score = max(0.0, 1.0 - mfcc_delta_std / 5.0)
        score = float(np.clip(low_var_score * 0.6 + low_delta_score * 0.4, 0.0, 1.0))

        return {
            "score": score,
            "mfcc_variance": round(mfcc_var, 4),
            "mfcc_delta_std": round(mfcc_delta_std, 4),
            "frames_analyzed": len(mfcc_features),
            "interpretation": _interpret_score(score),
            "description": "MFCC statistical analysis — synthetic speech typically shows unnaturally smooth MFCC distributions."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 2: Spectrogram Artifact Detection
# ─────────────────────────────────────────────────────────────────
def _spectrogram_artifact_detection(mono: np.ndarray, sr: int) -> dict:
    """
    Detect harmonic smearing and unnatural frequency band energies in STFT.
    Neural vocoders leave characteristic spectral artifacts.
    """
    try:
        n_fft = 2048
        hop = 512
        window = np.hanning(n_fft)
        frames = []
        for i in range(0, len(mono) - n_fft, hop):
            frame = mono[i:i+n_fft] * window
            frames.append(np.abs(np.fft.rfft(frame)))

        if not frames:
            return {"score": 0.2, "interpretation": "Unknown", "description": ""}

        spectrogram = np.array(frames).T  # (freq_bins, time_frames)
        freq_bins = spectrogram.shape[0]

        # Harmonic smearing: high spectral flatness = noise-like vs tonal
        def spectral_flatness(spectrum):
            geo_mean = np.exp(np.mean(np.log(spectrum + 1e-9)))
            arith_mean = np.mean(spectrum) + 1e-9
            return geo_mean / arith_mean

        flatness_over_time = [spectral_flatness(spectrogram[:, t]) for t in range(spectrogram.shape[1])]
        mean_flatness = float(np.mean(flatness_over_time))
        std_flatness = float(np.std(flatness_over_time))

        # High temporal uniformity of spectral flatness = TTS artifact
        cv_flatness = std_flatness / (mean_flatness + 1e-9)
        uniformity_score = max(0.0, 1.0 - cv_flatness * 3)

        # Check for unnatural silence in high frequencies (vocoder cutoff)
        high_freq_start = freq_bins * 3 // 4
        high_energy = float(np.mean(spectrogram[high_freq_start:, :]))
        low_energy = float(np.mean(spectrogram[:freq_bins//4, :]))
        energy_ratio = high_energy / (low_energy + 1e-9)

        high_freq_cutoff_score = max(0.0, 1.0 - energy_ratio * 20) if energy_ratio < 0.05 else 0.0

        score = float(np.clip(uniformity_score * 0.6 + high_freq_cutoff_score * 0.4, 0.0, 1.0))

        return {
            "score": score,
            "spectral_flatness_mean": round(mean_flatness, 6),
            "spectral_flatness_cv": round(cv_flatness, 4),
            "high_freq_energy_ratio": round(energy_ratio, 6),
            "interpretation": _interpret_score(score),
            "description": "Spectrogram analysis detects harmonic smearing and neural vocoder cutoff artifacts."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 3: Pitch Continuity Analysis
# ─────────────────────────────────────────────────────────────────
def _pitch_continuity_analysis(mono: np.ndarray, sr: int) -> dict:
    """
    Track fundamental frequency (F0) using zero-crossing rate as proxy.
    TTS systems often produce artificially smooth pitch contours.
    """
    try:
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        zcr_values = []

        for i in range(0, len(mono) - frame_length, hop_length):
            frame = mono[i:i + frame_length]
            # Zero-crossing rate as pitch approximation
            zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length))
            zcr_values.append(zcr)

        if len(zcr_values) < 10:
            return {"score": 0.2, "interpretation": "Too short", "description": ""}

        zcr_arr = np.array(zcr_values)
        zcr_mean = float(np.mean(zcr_arr))
        zcr_std = float(np.std(zcr_arr))

        # Compute pitch (ZCR) continuity — sudden resets indicate TTS
        zcr_diffs = np.abs(np.diff(zcr_arr))
        spike_count = int(np.sum(zcr_diffs > zcr_std * 2.5))
        spike_ratio = spike_count / max(len(zcr_diffs), 1)

        # Also check for unnaturally low variance (too smooth = TTS)
        smoothness_score = max(0.0, 1.0 - zcr_std / 0.005) if zcr_std < 0.01 else 0.0
        discontinuity_score = min(1.0, spike_ratio * 10)

        score = float(np.clip(smoothness_score * 0.5 + discontinuity_score * 0.5, 0.0, 1.0))

        return {
            "score": score,
            "zcr_mean": round(zcr_mean, 6),
            "zcr_std": round(zcr_std, 6),
            "pitch_discontinuities": spike_count,
            "interpretation": _interpret_score(score),
            "description": "Pitch continuity analysis detects artificial pitch resets and unnatural smoothness in TTS audio."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 4: Background Noise Consistency
# ─────────────────────────────────────────────────────────────────
def _background_noise_consistency(mono: np.ndarray, sr: int) -> dict:
    """
    Analyze background noise floor uniformity.
    AI-generated speech often has unrealistically consistent or absent noise.
    """
    try:
        # Split into short-term energy segments
        seg_length = int(0.1 * sr)
        energies = []

        for i in range(0, len(mono) - seg_length, seg_length):
            segment = mono[i:i + seg_length]
            energy = float(np.mean(segment ** 2))
            energies.append(energy)

        if len(energies) < 5:
            return {"score": 0.2, "interpretation": "Too short", "description": ""}

        energy_arr = np.array(energies)

        # Find silence segments (lowest 20% energy = noise floor)
        threshold = np.percentile(energy_arr, 20)
        noise_segments = energy_arr[energy_arr <= threshold]

        if len(noise_segments) < 2:
            return {"score": 0.3, "interpretation": "Suspicious", "description": "Possibly no natural noise floor."}

        noise_mean = float(np.mean(noise_segments))
        noise_std = float(np.std(noise_segments))
        noise_cv = noise_std / (noise_mean + 1e-9)

        # Very uniform noise (CV < 0.1) = TTS (artificially consistent)
        # Complete silence (near-zero mean) = TTS
        silence_score = max(0.0, 1.0 - noise_mean * 1000) if noise_mean < 0.001 else 0.0
        uniformity_score = max(0.0, 1.0 - noise_cv * 5) if noise_cv < 0.2 else 0.0

        score = float(np.clip(silence_score * 0.5 + uniformity_score * 0.5, 0.0, 1.0))

        return {
            "score": score,
            "noise_floor_mean": round(float(noise_mean), 8),
            "noise_floor_cv": round(float(noise_cv), 6),
            "interpretation": _interpret_score(score),
            "description": "Background noise consistency checks for unnaturally clean or perfectly uniform noise floors."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 5: Silence Pattern Analysis
# ─────────────────────────────────────────────────────────────────
def _silence_pattern_analysis(mono: np.ndarray, sr: int) -> dict:
    """
    Analyze inter-word silence durations.
    TTS tends to produce overly regular silence between phonemes.
    """
    try:
        # Energy envelope
        frame_length = int(0.02 * sr)
        hop_length = int(0.01 * sr)
        energies = []

        for i in range(0, len(mono) - frame_length, hop_length):
            frame = mono[i:i + frame_length]
            energies.append(float(np.mean(frame ** 2)))

        if len(energies) < 20:
            return {"score": 0.2, "interpretation": "Too short", "description": ""}

        energy_arr = np.array(energies)
        # Adaptive threshold
        threshold = np.percentile(energy_arr, 15)
        is_silence = energy_arr < threshold

        # Find silence durations
        silence_durations = []
        count = 0
        for s in is_silence:
            if s:
                count += 1
            else:
                if count > 2:
                    silence_durations.append(count * (hop_length / sr))
                count = 0

        if len(silence_durations) < 3:
            return {
                "score": 0.15,
                "silence_segments": len(silence_durations),
                "interpretation": "Insufficient silence segments.",
                "description": "Silence pattern analysis — very few pause segments found."
            }

        sd_arr = np.array(silence_durations)
        sd_mean = float(np.mean(sd_arr))
        sd_std = float(np.std(sd_arr))
        sd_cv = sd_std / (sd_mean + 1e-9)

        # Human speech: high variability in pause duration
        # TTS: very regular, low CV
        regularity_score = max(0.0, 1.0 - sd_cv * 3)
        score = float(np.clip(regularity_score, 0.0, 1.0))

        return {
            "score": score,
            "mean_silence_duration_ms": round(sd_mean * 1000, 1),
            "silence_cv": round(sd_cv, 4),
            "silence_segments": len(silence_durations),
            "interpretation": _interpret_score(score),
            "description": "Pause duration regularity analysis — TTS audio shows unnaturally uniform inter-word silences."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 6: Formant Analysis
# ─────────────────────────────────────────────────────────────────
def _formant_analysis(mono: np.ndarray, sr: int) -> dict:
    """
    Track LPC-based formant using spectral peaks.
    Synthetic voices show unnaturally smooth formant trajectories.
    """
    try:
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        lpc_order = 12
        formant_trajectories = []

        for i in range(0, min(len(mono) - frame_length, frame_length * 200), hop_length):
            frame = mono[i:i + frame_length] * np.hamming(frame_length)
            if np.max(np.abs(frame)) < 0.01:
                continue

            # LPC using autocorrelation
            r = np.correlate(frame, frame, mode='full')[frame_length - 1:]
            r = r[:lpc_order + 1]

            if r[0] < 1e-10:
                continue

            # Levinson-Durbin recursion (simplified)
            a = np.zeros(lpc_order)
            e = r[0]
            for m in range(1, lpc_order + 1):
                k = -np.dot(a[:m-1], r[m-1:0:-1]) - r[m]
                k /= (e + 1e-10)
                a_new = a[:m-1] + k * a[m-2::-1] if m > 1 else np.array([])
                a = np.append(a_new, k)
                e *= (1 - k**2)

            # Find peaks in LPC spectrum
            freq_response = np.fft.rfft(np.append([1], a), n=512)
            spectrum = 1.0 / (np.abs(freq_response) + 1e-9)
            freqs = np.linspace(0, sr / 2, len(spectrum))

            # Find first 3 formants
            peaks = []
            for j in range(1, len(spectrum) - 1):
                if spectrum[j] > spectrum[j-1] and spectrum[j] > spectrum[j+1]:
                    if freqs[j] > 90 and freqs[j] < sr // 2:
                        peaks.append(freqs[j])

            # Sort and take F1, F2
            peaks.sort()
            if len(peaks) >= 2:
                formant_trajectories.append(peaks[:2])

        if len(formant_trajectories) < 5:
            return {
                "score": 0.2,
                "interpretation": "Insufficient voiced frames",
                "description": "Formant analysis — insufficient voiced speech segments found."
            }

        ft_arr = np.array(formant_trajectories)
        f1_std = float(np.std(ft_arr[:, 0]))
        f2_std = float(np.std(ft_arr[:, 1]))
        f1_mean = float(np.mean(ft_arr[:, 0]))
        f2_mean = float(np.mean(ft_arr[:, 1]))

        # TTS: unnaturally smooth formant tracks (low std)
        f1_score = max(0.0, 1.0 - f1_std / 150)
        f2_score = max(0.0, 1.0 - f2_std / 300)
        score = float(np.clip(f1_score * 0.5 + f2_score * 0.5, 0.0, 1.0))

        return {
            "score": score,
            "f1_mean_hz": round(f1_mean, 1),
            "f2_mean_hz": round(f2_mean, 1),
            "f1_std_hz": round(f1_std, 1),
            "f2_std_hz": round(f2_std, 1),
            "frames_analyzed": len(formant_trajectories),
            "interpretation": _interpret_score(score),
            "description": "Formant trajectory analysis via LPC — TTS shows unnaturally smooth vowel formant patterns."
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e), "interpretation": "Unknown"}


# ─────────────────────────────────────────────────────────────────
# METHOD 7: Phase Coherence Analysis
# ─────────────────────────────────────────────────────────────────
def _phase_coherence_analysis(audio_data: np.ndarray, sr: int, is_stereo: bool) -> dict:
    """
    Analyze left-right phase coherence in stereo audio.
    In mono/TTS audio: perfect coherence. In real recordings: natural decorrelation.
    """
    try:
        if not is_stereo or audio_data.ndim < 2:
            # Mono: analyze phase consistency in signal itself
            # TTS mono has very uniform phase progression
            n_fft = 1024
            phases = []
            for i in range(0, min(len(audio_data), n_fft * 50), n_fft):
                frame = audio_data[i:i + n_fft]
                if len(frame) < n_fft:
                    break
                spectrum = np.fft.rfft(frame)
                phase = np.angle(spectrum)
                phases.append(phase)

            if len(phases) < 3:
                return {"score": 0.25, "is_stereo": False,
                        "interpretation": "Possibly Authentic",
                        "description": "Phase analysis on mono signal."}

            phase_arr = np.array(phases)
            phase_diff = np.diff(phase_arr, axis=0)
            phase_consistency = float(1.0 - np.mean(np.abs(np.sin(phase_diff))))
            score = float(np.clip(phase_consistency * 0.5, 0.0, 1.0))

            return {
                "score": score,
                "is_stereo": False,
                "phase_consistency": round(phase_consistency, 4),
                "interpretation": _interpret_score(score),
                "description": "Phase consistency analysis on mono audio — TTS shows unnaturally uniform phase progression."
            }
        else:
            left = audio_data[:, 0]
            right = audio_data[:, 1]

            # Compute inter-channel coherence
            n_fft = 2048
            coherences = []
            hop = n_fft // 2
            for i in range(0, len(left) - n_fft, hop):
                l_frame = left[i:i+n_fft] * np.hanning(n_fft)
                r_frame = right[i:i+n_fft] * np.hanning(n_fft)
                L = np.fft.rfft(l_frame)
                R = np.fft.rfft(r_frame)
                coherence = np.abs(np.mean(L * np.conj(R))) ** 2 / (
                    np.mean(np.abs(L)**2) * np.mean(np.abs(R)**2) + 1e-9)
                coherences.append(float(coherence))

            if not coherences:
                return {"score": 0.2, "interpretation": "Unknown", "description": ""}

            mean_coherence = float(np.mean(coherences))
            std_coherence = float(np.std(coherences))

            # Perfect coherence (>0.999) = artificially duplicated mono channel = TTS
            if mean_coherence > 0.999:
                score = 0.7
            elif mean_coherence > 0.99:
                score = 0.4
            else:
                score = max(0.0, 0.2 - std_coherence)

            score = float(np.clip(score, 0.0, 1.0))
            return {
                "score": score,
                "is_stereo": True,
                "inter_channel_coherence": round(mean_coherence, 6),
                "interpretation": _interpret_score(score),
                "description": "Stereo inter-channel phase coherence — perfect coherence indicates mono-duplicated TTS output."
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
