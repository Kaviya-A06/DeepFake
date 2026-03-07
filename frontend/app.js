/**
 * DeepFake Detector — Frontend Application Logic
 * Handles uploads, API calls, chart rendering, and result display.
 */

const API_BASE = "http://localhost:5000/api";

// ── Method display names and icons ─────────────────────────────
const METHOD_META = {
  // Image
  ela:               { name: "Error Level Analysis",        icon: "🔬" },
  noise_residual:    { name: "Noise Residual Analysis",     icon: "📡" },
  dct_frequency:     { name: "DCT Frequency Domain",        icon: "🌊" },
  metadata_forensics:{ name: "Metadata Forensics",          icon: "🏷️" },
  color_space:       { name: "Color Space Analysis",        icon: "🎨" },
  copy_move:         { name: "Copy-Move Detection",         icon: "📋" },
  double_jpeg:       { name: "Double JPEG Detection",       icon: "📦" },
  edge_sharpness:    { name: "Edge Sharpness Analysis",     icon: "🔭" },
  gan_fingerprint:   { name: "GAN Fingerprint Detection",   icon: "🤖" },
  prnu_analysis:     { name: "PRNU Analysis",               icon: "🔑" },
  // Video
  frame_ela:                  { name: "Frame-Level ELA",              icon: "🎞️" },
  temporal_consistency:       { name: "Temporal Consistency",         icon: "⏱️" },
  facial_landmark_stability:  { name: "Facial Landmark Stability",    icon: "👤" },
  eye_blink_pattern:          { name: "Eye Blink Pattern",            icon: "👁️" },
  compression_artifact:       { name: "Compression Artifact Anomaly", icon: "💾" },
  pixel_temporal_coherence:   { name: "Pixel Temporal Coherence",     icon: "🔲" },
  head_pose_estimation:       { name: "Head Pose Estimation",         icon: "🗿" },
  color_temporal_stability:   { name: "Color Temporal Stability",     icon: "🌈" },
  // Audio
  mfcc_analysis:          { name: "MFCC Analysis",             icon: "📈" },
  spectrogram_artifacts:  { name: "Spectrogram Artifacts",      icon: "🎼" },
  pitch_continuity:       { name: "Pitch Continuity",           icon: "🎵" },
  background_noise:       { name: "Background Noise",           icon: "🌫️" },
  silence_pattern:        { name: "Silence Pattern",            icon: "🤫" },
  formant_analysis:       { name: "Formant Analysis",           icon: "🗣️" },
  phase_coherence:        { name: "Phase Coherence",            icon: "↔️" },
};

// ── State ──────────────────────────────────────────────────────
const state = {
  image: { file: null, chartRadar: null, chartBar: null },
  video: { file: null, chartRadar: null, chartBar: null },
  audio: { file: null, chartRadar: null, chartBar: null },
};

// ── Chart.js Defaults ──────────────────────────────────────────
Chart.defaults.color = "#8b9ec7";
Chart.defaults.font.family = "'Inter', sans-serif";

// ══════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ══════════════════════════════════════════════════════════════════
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.tab === tab);
      b.setAttribute("aria-selected", b.dataset.tab === tab);
    });
    document.querySelectorAll(".tab-panel").forEach(p => {
      p.classList.toggle("active", p.id === `panel-${tab}`);
    });
  });
});

// ══════════════════════════════════════════════════════════════════
// UPLOAD ZONE SETUP (runs for each modality)
// ══════════════════════════════════════════════════════════════════
function setupUpload(modality) {
  const dropZone   = document.getElementById(`${modality}-drop-zone`);
  const fileInput  = document.getElementById(`${modality}-file-input`);
  const browseBtn  = document.getElementById(`${modality}-browse-btn`);
  const fileInfo   = document.getElementById(`${modality}-file-info`);
  const fileName   = document.getElementById(`${modality}-file-name`);
  const fileMeta   = document.getElementById(`${modality}-file-meta`);
  const analyzeBtn = document.getElementById(`${modality}-analyze-btn`);

  // Browse button click
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // Drop zone click
  dropZone.addEventListener("click", () => fileInput.click());

  // File selected
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) selectFile(modality, fileInput.files[0]);
  });

  // Drag events
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) selectFile(modality, e.dataTransfer.files[0]);
  });

  // Analyze button
  analyzeBtn.addEventListener("click", () => runAnalysis(modality));
}

["image", "video", "audio"].forEach(setupUpload);

// ── File selected handler ──────────────────────────────────────
function selectFile(modality, file) {
  state[modality].file = file;

  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
  document.getElementById(`${modality}-file-name`).textContent = file.name;
  document.getElementById(`${modality}-file-meta`).textContent =
    `${sizeMB} MB · ${file.type || "unknown type"}`;
  document.getElementById(`${modality}-file-info`).classList.add("visible");

  // Hide previous results
  hideResults(modality);
  hideError(modality);

  // Show original image preview for ELA
  if (modality === "image") {
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById("image-original-preview").src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
}

// ══════════════════════════════════════════════════════════════════
// ANALYSIS
// ══════════════════════════════════════════════════════════════════
async function runAnalysis(modality) {
  const file = state[modality].file;
  if (!file) return;

  hideError(modality);
  hideResults(modality);
  showProgress(modality);
  animateMethodProgress(modality);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE}/analyze/${modality}`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || `Server error: ${response.status}`);
    }

    hideProgress(modality);
    renderResults(modality, data, file);

  } catch (err) {
    hideProgress(modality);
    showError(modality, err.message || "Connection failed. Is the server running?");
  }
}

// ── Progress animation ─────────────────────────────────────────
const IMAGE_METHODS = ["ela","noise_residual","dct_frequency","metadata_forensics","color_space","copy_move","double_jpeg","edge_sharpness","gan_fingerprint","prnu_analysis"];
const VIDEO_METHODS = ["frame_ela","temporal_consistency","facial_landmark_stability","eye_blink_pattern","compression_artifact","pixel_temporal_coherence","head_pose_estimation","color_temporal_stability"];
const AUDIO_METHODS = ["mfcc_analysis","spectrogram_artifacts","pitch_continuity","background_noise","silence_pattern","formant_analysis","phase_coherence"];
const MODALITY_METHODS = { image: IMAGE_METHODS, video: VIDEO_METHODS, audio: AUDIO_METHODS };

function animateMethodProgress(modality) {
  const methods = MODALITY_METHODS[modality];
  const container = document.getElementById(`${modality}-method-list`);
  container.innerHTML = "";

  // Build items
  const items = methods.map(m => {
    const meta = METHOD_META[m] || { name: m, icon: "🔍" };
    const div = document.createElement("div");
    div.className = "method-progress-item";
    div.id = `prog-${modality}-${m}`;
    div.innerHTML = `<span class="method-status-dot"></span><span>${meta.icon} ${meta.name}</span>`;
    container.appendChild(div);
    return div;
  });

  // Animate sequentially
  let i = 0;
  const interval = setInterval(() => {
    if (i < items.length) {
      if (i > 0) items[i-1].classList.add("done");
      items[i].classList.add("active");
      i++;
    } else {
      items[items.length - 1].classList.add("done");
      clearInterval(interval);
    }
  }, 250);
}

// ══════════════════════════════════════════════════════════════════
// RESULT RENDERING
// ══════════════════════════════════════════════════════════════════
function renderResults(modality, data, file) {
  const ensemble = data.ensemble;
  const methods  = data.methods;

  // ── Verdict Banner ─────────────────────────────────────────
  const pct = ensemble.fake_probability;
  const score = ensemble.ensemble_score;
  const color = ensemble.verdict_color;

  // Ring animation
  const ringFill = document.getElementById(`${modality}-ring-fill`);
  const circumference = 477.52;
  const offset = circumference - (score * circumference);
  ringFill.style.stroke = color;
  setTimeout(() => {
    ringFill.style.strokeDashoffset = offset.toFixed(2);
  }, 100);

  document.getElementById(`${modality}-pct`).textContent = `${pct}%`;
  document.getElementById(`${modality}-pct`).style.color = color;
  document.getElementById(`${modality}-verdict-title`).textContent = ensemble.verdict;
  document.getElementById(`${modality}-verdict-title`).style.color = color;
  document.getElementById(`${modality}-verdict-desc`).textContent = ensemble.verdict_description;

  // Set banner gradient based on verdict
  const banner = document.getElementById(`${modality}-verdict-banner`);
  banner.style.setProperty("--verdict-bg", `linear-gradient(135deg, ${color}18, ${color}08)`);
  banner.style.borderColor = `${color}30`;

  // Pills
  document.getElementById(`${modality}-risk-pill`).textContent = `⚡ Risk: ${ensemble.risk_level}`;
  document.getElementById(`${modality}-risk-pill`).style.color = color;
  document.getElementById(`${modality}-risk-pill`).style.borderColor = `${color}50`;

  if (modality === "image") {
    const sizeMB = data.file_size_kb ? `${(data.file_size_kb/1024).toFixed(2)} MB` : "—";
    document.getElementById("image-methods-pill").textContent = `🔬 ${ensemble.methods_flagged}/${ensemble.total_methods} methods flagged`;
    document.getElementById("image-size-pill").textContent = `📁 ${data.file_size_kb} KB`;
  } else if (modality === "video") {
    document.getElementById("video-duration-pill").textContent = `⏱️ ${data.duration}s`;
    document.getElementById("video-frames-pill").textContent = `🎞️ ${data.frame_count} frames`;
    // Video meta card
    if (data.frame_previews && data.frame_previews.length > 0) {
      document.getElementById("video-thumb").src = `data:image/jpeg;base64,${data.frame_previews[0]}`;
      document.getElementById("video-meta-card").style.display = "flex";
    }
    document.getElementById("vm-duration").textContent = `${data.duration}s`;
    document.getElementById("vm-fps").textContent = data.fps;
    document.getElementById("vm-frames").textContent = data.frame_count;
    document.getElementById("vm-size").textContent = `${data.file_size_mb} MB`;
  } else if (modality === "audio") {
    document.getElementById("audio-duration-pill").textContent = `⏱️ ${data.duration}s`;
    document.getElementById("audio-sr-pill").textContent = `🎚️ ${data.sample_rate} Hz`;
    document.getElementById("am-duration").textContent = `${data.duration}s`;
    document.getElementById("am-sr").textContent = `${data.sample_rate} Hz`;
    document.getElementById("am-size").textContent = `${data.file_size_kb} KB`;
    buildWaveform();
  }

  // ── ELA Image (image only) ─────────────────────────────────
  if (modality === "image" && data.ela_image) {
    document.getElementById("image-ela-preview").src = `data:image/png;base64,${data.ela_image}`;
    document.getElementById("image-ela-section").style.display = "block";
  }

  // ── Charts ─────────────────────────────────────────────────
  const methodScores = ensemble.method_scores; // { method_name: 0-100 }
  const labels = Object.keys(methodScores).map(k => (METHOD_META[k] || { name: k }).name);
  const values = Object.values(methodScores);
  const colors = values.map(v => scoreColor(v / 100));

  renderRadarChart(modality, labels, values);
  renderBarChart(modality, labels, values, colors);

  // ── Method Cards ───────────────────────────────────────────
  renderMethodCards(modality, methods, ensemble.method_scores);

  // ── Show results ───────────────────────────────────────────
  document.getElementById(`${modality}-results`).classList.add("visible");
  document.getElementById(`${modality}-results`).scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Radar Chart ────────────────────────────────────────────────
function renderRadarChart(modality, labels, values) {
  const canvas = document.getElementById(`${modality}-radar-chart`);
  if (state[modality].chartRadar) state[modality].chartRadar.destroy();

  state[modality].chartRadar = new Chart(canvas, {
    type: "radar",
    data: {
      labels: labels.map(l => l.length > 16 ? l.substring(0,14)+"…" : l),
      datasets: [{
        label: "Fake Score %",
        data: values,
        backgroundColor: "rgba(255,0,110,0.15)",
        borderColor: "#ff006e",
        borderWidth: 2,
        pointBackgroundColor: values.map(v => scoreColor(v/100)),
        pointBorderColor: "#fff",
        pointRadius: 4,
        pointHoverRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1000, easing: "easeInOutQuart" },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, color: "#4a5578", font: { size: 10 }, backdropColor: "transparent" },
          grid: { color: "rgba(255,255,255,0.06)" },
          angleLines: { color: "rgba(255,255,255,0.06)" },
          pointLabels: { color: "#8b9ec7", font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.raw.toFixed(1)}%`
          }
        }
      }
    }
  });
}

// ── Bar Chart ──────────────────────────────────────────────────
function renderBarChart(modality, labels, values, colors) {
  const canvas = document.getElementById(`${modality}-bar-chart`);
  if (state[modality].chartBar) state[modality].chartBar.destroy();

  state[modality].chartBar = new Chart(canvas, {
    type: "bar",
    data: {
      labels: labels.map(l => l.length > 18 ? l.substring(0,16)+"…" : l),
      datasets: [{
        label: "Fake %",
        data: values,
        backgroundColor: colors.map(c => c + "99"),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1200, easing: "easeInOutQuart" },
      scales: {
        x: {
          min: 0, max: 100,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#4a5578", callback: v => v + "%" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#8b9ec7", font: { size: 11 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` Fake probability: ${ctx.raw.toFixed(1)}%`
          }
        }
      }
    }
  });
}

// ── Method Cards ───────────────────────────────────────────────
function renderMethodCards(modality, methods, methodScores) {
  const grid = document.getElementById(`${modality}-methods-grid`);
  grid.innerHTML = "";

  Object.entries(methods).forEach(([key, result]) => {
    if (key === "error") return;
    const meta = METHOD_META[key] || { name: key, icon: "🔍" };
    const score = methodScores[key] || 0; // 0-100
    const score01 = score / 100;
    const { cls, barCls, textCls, verdict } = scoreClasses(score01);

    const card = document.createElement("div");
    card.className = "method-card";
    card.innerHTML = `
      <div class="method-card-header">
        <div class="method-score-badge ${cls}">${score.toFixed(0)}%</div>
        <div class="method-card-info">
          <div class="method-card-name">${meta.icon} ${meta.name}</div>
          <div class="method-card-verdict ${textCls}">${verdict}</div>
        </div>
        <span class="expand-chevron">▼</span>
      </div>
      <div class="method-score-bar-wrapper">
        <div class="method-score-bar-bg">
          <div class="method-score-bar-fill ${barCls}" style="width:0%"></div>
        </div>
      </div>
      <div class="method-card-description">${result.description || ""}</div>
      <div class="method-card-details">
        <div class="detail-grid">
          ${buildDetailGrid(result)}
        </div>
      </div>
    `;

    // Animate bar
    setTimeout(() => {
      const bar = card.querySelector(".method-score-bar-fill");
      if (bar) bar.style.width = `${score}%`;
    }, 200);

    // Expand/collapse
    card.querySelector(".method-card-header").addEventListener("click", () => {
      card.classList.toggle("expanded");
    });

    grid.appendChild(card);
  });
}

function buildDetailGrid(result) {
  const skip = new Set(["score", "interpretation", "description", "ela_image", "error", "flags"]);
  const entries = Object.entries(result).filter(([k]) => !skip.has(k));

  let html = entries.map(([k, v]) => {
    const label = k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    const value = typeof v === "number" ? v.toFixed(4) : String(v).substring(0, 40);
    return `<div class="detail-item">
      <div class="detail-label">${label}</div>
      <div class="detail-value">${value}</div>
    </div>`;
  }).join("");

  // Add flags if present
  if (result.flags && result.flags.length > 0) {
    html += `<div class="detail-item" style="grid-column:1/-1">
      <div class="detail-label">⚠️ Flags Detected</div>
      <div class="detail-value" style="font-size:11px; white-space:normal; font-family:var(--font-sans); line-height:1.5">
        ${result.flags.map(f => `• ${f}`).join("<br>")}
      </div>
    </div>`;
  }

  return html || `<div class="detail-item" style="grid-column:1/-1"><div class="detail-value text-muted">No detailed metrics available.</div></div>`;
}

// ── Audio Waveform Build ───────────────────────────────────────
function buildWaveform() {
  const container = document.getElementById("audio-waveform");
  container.innerHTML = "";
  const heights = [20,35,55,70,85,65,45,75,90,60,40,80,50,70,38,62,78,44,88,55,30,65,45,75,60];
  heights.forEach((h, i) => {
    const bar = document.createElement("div");
    bar.className = "wave-bar";
    bar.style.height = `${h}%`;
    bar.style.animationDelay = `${i * 0.05}s`;
    container.appendChild(bar);
  });
}

// ── Score utilities ────────────────────────────────────────────
function scoreColor(score01) {
  if (score01 < 0.25) return "#00d4aa";
  if (score01 < 0.50) return "#f5c518";
  if (score01 < 0.70) return "#ff8c00";
  if (score01 < 0.85) return "#ff3860";
  return "#ff0055";
}

function scoreClasses(score01) {
  if (score01 < 0.25) return { cls:"score-authentic", barCls:"score-authentic-bar", textCls:"score-authentic-text", verdict:"✅ Likely Authentic" };
  if (score01 < 0.50) return { cls:"score-possible",  barCls:"score-possible-bar",  textCls:"score-possible-text",  verdict:"🟡 Possibly Authentic" };
  if (score01 < 0.70) return { cls:"score-suspicious", barCls:"score-suspicious-bar", textCls:"score-suspicious-text", verdict:"🟠 Suspicious" };
  if (score01 < 0.85) return { cls:"score-fake",      barCls:"score-fake-bar",      textCls:"score-fake-text",      verdict:"🔴 Likely Fake" };
  return                      { cls:"score-fake",      barCls:"score-fake-bar",      textCls:"score-fake-text",      verdict:"🚨 Almost Certainly Fake" };
}

// ── UI Helpers ─────────────────────────────────────────────────
function showProgress(modality) {
  document.getElementById(`${modality}-progress`).classList.add("visible");
}
function hideProgress(modality) {
  document.getElementById(`${modality}-progress`).classList.remove("visible");
}
function hideResults(modality) {
  document.getElementById(`${modality}-results`).classList.remove("visible");
  if (modality === "image") {
    document.getElementById("image-ela-section").style.display = "none";
  }
}
function showError(modality, msg) {
  document.getElementById(`${modality}-error-msg`).textContent = msg;
  document.getElementById(`${modality}-error`).classList.add("visible");
}
function hideError(modality) {
  document.getElementById(`${modality}-error`).classList.remove("visible");
}

// ── API Health Check on Load ───────────────────────────────────
window.addEventListener("load", async () => {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    console.log("✅ DeepFake Detection API is online.");
  } catch {
    console.warn("⚠️ Could not connect to API at", API_BASE, "— make sure the server is running.");
  }
});
