# 🔍 DeepFake Detector — AI Media Forensics Platform

A professional, full-stack deepfake detection web application that analyzes images, videos, and audio files using **25 forensic detection methods** and delivers a comprehensive authenticity report.

---

## 🚀 Quick Start (Windows)

### Option 1 — One-click launcher (Recommended)
```
Double-click: start.bat
```
This will automatically:
1. Create a Python virtual environment
2. Install all required packages
3. Start the Flask server at `http://localhost:5000`
4. Open your browser

### Option 2 — Manual startup
```powershell
# From the DeepFake folder
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
cd backend
python app.py
```
Then open `http://localhost:5000` in your browser.

---

## 🧠 Detection Methods

### 🖼️ Image Analysis (10 Methods)
| Method | Description |
|--------|-------------|
| **Error Level Analysis (ELA)** | Detects inconsistent JPEG compression artifacts from tampering |
| **Noise Residual Analysis** | Identifies missing or unnatural camera sensor noise |
| **DCT Frequency Domain** | Detects GAN spectral peaks and checkerboard artifacts |
| **Metadata Forensics** | Checks EXIF for AI software markers and missing camera data |
| **Color Space Analysis** | Finds unnatural chroma correlations and YCbCr banding |
| **Copy-Move Detection** | ORB feature matching to find duplicated regions |
| **Double JPEG Detection** | DCT histogram analysis for re-compression artifacts |
| **Edge Sharpness Inconsistency** | Detects unnaturally uniform focus distribution |
| **GAN Fingerprint Detection** | Spectral checkerboard from transposed convolution upsampling |
| **PRNU Analysis** | Checks for missing camera sensor fingerprint |

### 🎬 Video Analysis (8 Methods)
| Method | Description |
|--------|-------------|
| **Frame-Level ELA** | ELA applied across sampled video frames |
| **Temporal Consistency** | Optical flow analysis for motion discontinuities |
| **Facial Landmark Stability** | Face bounding box jitter tracking |
| **Eye Blink Pattern** | Natural blink cadence detection |
| **Compression Artifact Anomaly** | Inter-frame DCT coefficient inconsistencies |
| **Pixel Temporal Coherence** | Center vs edge pixel variance analysis |
| **Head Pose Estimation** | Detects unnatural face orientation changes |
| **Color Temporal Stability** | Skin tone consistency across frames |

### 🎵 Audio Analysis (7 Methods)
| Method | Description |
|--------|-------------|
| **MFCC Statistical Analysis** | Mel-frequency cepstral coefficient distribution |
| **Spectrogram Artifact Detection** | Harmonic smearing and vocoder cutoff detection |
| **Pitch Continuity Analysis** | Zero-crossing rate pitch discontinuity tracking |
| **Background Noise Consistency** | SNR uniformity and noise floor analysis |
| **Silence Pattern Analysis** | Inter-word pause duration regularity |
| **Formant Analysis** | LPC-based vowel formant trajectory smoothness |
| **Phase Coherence Analysis** | Stereo inter-channel phase correlation |

---

## 🏗️ Project Structure

```
DeepFake/
├── start.bat                    ← One-click launcher
├── frontend/
│   ├── index.html               ← Single-page UI
│   ├── style.css                ← Dark theme design system
│   └── app.js                   ← UI logic & Chart.js charts
└── backend/
    ├── app.py                   ← Flask API server
    ├── requirements.txt         ← Python dependencies
    ├── analyzers/
    │   ├── image_analyzer.py    ← 10 image detection methods
    │   ├── video_analyzer.py    ← 8 video detection methods
    │   └── audio_analyzer.py    ← 7 audio detection methods
    └── utils/
        └── ensemble.py          ← Weighted ensemble scoring
```

---

## 📦 Requirements

- **Python 3.10+**
- All other dependencies are auto-installed via `start.bat` or `pip install -r backend/requirements.txt`

Key packages: `flask`, `flask-cors`, `Pillow`, `opencv-python`, `numpy`, `scipy`, `soundfile`, `librosa`, `scikit-learn`

---

## 📊 How the Scoring Works

1. Each method returns a **fake probability score** from 0.0 (authentic) to 1.0 (fake)
2. Method scores are combined using **weighted ensemble scoring** tailored per modality
3. If 3+ methods agree on a high score, a confidence boost is applied
4. The final verdict is mapped to: **AUTHENTIC / POSSIBLY AUTHENTIC / SUSPICIOUS / LIKELY FAKE / DEEPFAKE DETECTED**

---

## 🎨 UI Features

- Premium dark theme with glassmorphism card design
- Animated SVG confidence ring with percentage indicator
- Radar chart (all method scores at a glance)
- Horizontal bar chart (per-method breakdown)
- ELA heatmap overlay for images
- Expandable method cards with raw metrics
- Drag-and-drop file upload with live animations
- Fully responsive layout

---

*Built for educational and research purposes in AI media forensics.*
