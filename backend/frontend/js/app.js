/* app.js - FINAL: No More ReferenceError + All Working */

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const waveCanvas = document.getElementById("waveCanvas");
const waveCtx = waveCanvas.getContext("2d");
const videoCard = document.getElementById("videoCard");

const faceStateStep1 = document.getElementById("faceStateStep1");
const liveEmotionResult = document.getElementById("liveEmotionResult");
const capturedPhoto = document.getElementById("capturedPhoto");
const photoPreviewContainer = document.getElementById("photoPreviewContainer");
const emotionPreview = document.getElementById("emotionPreview");

const captureBtn = document.getElementById("captureBtn");
const retakeBtn = document.getElementById("retakeBtn");
const usePhotoBtn = document.getElementById("usePhotoBtn");
const continueFaceBtn = document.getElementById("continue-face");
const continuePpgBtn = document.getElementById("continue-ppg");
const restartBtn = document.getElementById("restartBtn");

const fingerState = document.getElementById("fingerState");
const hrEstimateStep2 = document.getElementById("hrEstimateStep2");
const hrEstimateFinal = document.getElementById("hrEstimateFinal");
const stressScoreFinal = document.getElementById("stressScoreFinal");
const emotionResultEl = document.getElementById("emotionResult");
const ppgLoadingOverlay = document.getElementById("ppgLoadingOverlay");
const stabilizationText = document.getElementById("stabilizationText");
const progressFill = document.getElementById("progressFill");

// AI Elements
const aiAnalysisBtn = document.getElementById("ai-analysis-btn");
const aiLoading = document.getElementById("ai-loading");
const aiContent = document.getElementById("ai-content");
const chronicPredictionEl = document.getElementById("chronic-prediction");
const dietPlanEl = document.getElementById("diet-plan");
const exerciseSuggestionEl = document.getElementById("exercise-suggestion");
const restartFromAiBtn = document.getElementById("restart-from-ai");

const FRAME_INTERVAL_MS = 100;
const MAX_BUFFER_SECONDS = 20;
const PPG_STABILIZE_TIME = 15000;

let stream = null;
let videoTrack = null;
let torchOn = false;
let captureTimer = null;
let buffer = [];
let currentStep = 1;
let lastHR = null;
let lastEmotions = null;
let lastStress = null;
let cameraReady = false;
let capturedDataUrl = null;
let ppgStartTime = 0;
let hrReadings = [];

/* ====================== HELPER FUNCTIONS ====================== */

function setFaceState(text) {
  if (faceStateStep1) faceStateStep1.textContent = text;
}

function setFingerState(text) {
  if (fingerState) {
    fingerState.textContent = text;
    fingerState.style.color = text.includes("Good") ? "#10b981" : "#ef4444";
  }
}

function setHR(value) {
  const bpm = value == null ? "—" : Math.round(value);
  if (hrEstimateStep2) hrEstimateStep2.textContent = bpm;

  if (currentStep === 2 && value !== null && value >= 40 && value <= 180) {
    hrReadings.push(value);
    if (hrReadings.length > 10) hrReadings.shift();

    const avgHR = hrReadings.reduce((a, b) => a + b, 0) / hrReadings.length;
    lastHR = avgHR;

    const elapsedTime = Date.now() - ppgStartTime;
    const remaining = Math.max(0, PPG_STABILIZE_TIME - elapsedTime);
    const progress = Math.min(elapsedTime / PPG_STABILIZE_TIME, 1);
    if (progressFill) progressFill.style.width = progress * 100 + "%";

    if (elapsedTime > PPG_STABILIZE_TIME && avgHR >= 40 && avgHR <= 180) {
      if (stabilizationText)
        stabilizationText.textContent = "Signal stable! You can continue.";
      if (continuePpgBtn) continuePpgBtn.disabled = false;
    } else {
      const secs = Math.ceil(remaining / 1000);
      if (stabilizationText)
        stabilizationText.textContent = `Measuring... ${secs}s remaining`;
    }
  }
}

function updateFinalResults() {
  if (hrEstimateFinal && lastHR !== null)
    hrEstimateFinal.textContent = Math.round(lastHR);
  if (stressScoreFinal && lastStress !== null)
    stressScoreFinal.textContent = (lastStress * 100).toFixed(1) + "%";

  if (emotionResultEl && lastEmotions) {
    const sorted = Object.entries(lastEmotions)
      .sort((a, b) => b[1] - a[1])
      .map(
        ([e, p]) =>
          `<span class="emotion-chip">${e}: ${(p * 100).toFixed(1)}%</span>`
      )
      .join("");
    emotionResultEl.innerHTML = sorted;
  }
}

/* ====================== CAMERA & PPG FUNCTIONS ====================== */

function toCanvasSize() {
  if (!video.videoWidth || !video.videoHeight) return;
  const rect = video.getBoundingClientRect();
  overlay.style.width = rect.width + "px";
  overlay.style.height = rect.height + "px";
  const dpr = window.devicePixelRatio || 1;
  overlay.width = video.videoWidth * dpr;
  overlay.height = video.videoHeight * dpr;
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function setupCameraReady() {
  const makeReady = () => {
    if (cameraReady) return;
    cameraReady = true;
    toCanvasSize();
    if (currentStep === 1) drawGuideBox();
    startCaptureLoop();
    if (currentStep === 1) {
      if (captureBtn) captureBtn.disabled = false;
      setFaceState("Ready — center your face and tap Capture");
    }
  };

  video.onloadedmetadata = makeReady;
  video.oncanplay = makeReady;
  video.onplaying = makeReady;
  setTimeout(makeReady, 2000);
}

function drawGuideBox() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const box = {
    x: video.videoWidth * 0.15,
    y: video.videoHeight * 0.15,
    w: video.videoWidth * 0.7,
    h: video.videoHeight * 0.7,
  };

  overlayCtx.strokeStyle = "#00ff88";
  overlayCtx.lineWidth = 12;
  overlayCtx.shadowColor = "#00ff88";
  overlayCtx.shadowBlur = 30;
  overlayCtx.strokeRect(box.x, box.y, box.w, box.h);

  overlayCtx.fillStyle = "#ffffff";
  overlayCtx.font = "bold 28px Inter";
  overlayCtx.textAlign = "center";
  overlayCtx.fillText(
    "Center your face here",
    video.videoWidth / 2,
    box.y - 20
  );

  overlayCtx.shadowBlur = 0;
}

function updateOverlay() {
  if (cameraReady && currentStep === 1) drawGuideBox();
}

function appendToBuffer(t, value) {
  buffer.push({ t, value });
  const cutoff = t - MAX_BUFFER_SECONDS;
  while (buffer.length && buffer[0].t < cutoff) buffer.shift();
}

function detrendSignal(arr) {
  if (arr.length < 10) return arr.map((x) => x.value);
  const win = 20;
  return arr.map((p, i) => {
    const start = Math.max(0, i - win);
    const slice = arr.slice(start, i + 1);
    const avg = slice.reduce((s, v) => s + v.value, 0) / slice.length;
    return p.value - avg;
  });
}

function detectPeaks(sig) {
  if (sig.length < 10) return [];
  const mean = sig.reduce((a, b) => a + b, 0) / sig.length;
  const stdDev = Math.sqrt(
    sig.reduce((a, v) => a + Math.pow(v - mean, 2), 0) / sig.length
  );
  const thresh = mean + 0.5 * stdDev;
  const peaks = [];
  for (let i = 1; i < sig.length - 1; i++) {
    if (sig[i] > sig[i - 1] && sig[i] > sig[i + 1] && sig[i] > thresh) {
      peaks.push(i);
    }
  }
  return peaks;
}

function estimateHR(peaks, times) {
  if (peaks.length < 3) return null;
  const intervals = peaks.slice(1).map((p, i) => times[p] - times[peaks[i]]);
  const validIntervals = intervals.filter((i) => i > 0.3 && i < 2);
  if (validIntervals.length < 2) return null;
  const avg = validIntervals.reduce((a, b) => a + b, 0) / validIntervals.length;
  return avg > 0 ? Math.round(60 / avg) : null;
}

function renderWaveform(sig) {
  waveCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
  if (!sig.length) return;
  const scale = waveCanvas.height / 2 / (Math.max(...sig.map(Math.abs)) || 1);
  waveCtx.strokeStyle = "#10b981";
  waveCtx.lineWidth = 4;
  waveCtx.beginPath();
  for (let i = 0; i < sig.length; i++) {
    const x = (i / (sig.length - 1)) * waveCanvas.width;
    const y = waveCanvas.height / 2 - sig[i] * scale;
    i === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y);
  }
  waveCtx.stroke();
}

async function processFrame() {
  if (!cameraReady) return;

  const canvas = document.createElement("canvas");
  canvas.width = 160;
  canvas.height = video.videoHeight * (160 / video.videoWidth);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const roiSize = Math.min(canvas.width, canvas.height) * 0.7;
  const roi = {
    x: (canvas.width - roiSize) / 2,
    y: (canvas.height - roiSize) / 2,
    w: roiSize,
    h: roiSize,
  };

  const imgData = ctx.getImageData(roi.x, roi.y, roi.w, roi.h);
  let sumR = 0,
    sumG = 0;
  for (let i = 0; i < imgData.data.length; i += 4) {
    sumR += imgData.data[i];
    sumG += imgData.data[i + 1];
  }
  const pixelCount = roi.w * roi.h;
  const meanR = sumR / pixelCount;
  const meanG = sumG / pixelCount;

  const ppgSignal = meanR + meanG;

  if (currentStep === 2) {
    const contactQuality =
      meanG > 50 ? "Good contact ✅" : "Cover camera fully";
    setFingerState(contactQuality);
  }

  const now = performance.now() / 1000;
  appendToBuffer(now, ppgSignal);

  const sig = detrendSignal(buffer.slice());
  renderWaveform(sig);

  const peaks = detectPeaks(sig);
  const times = buffer.map((b) => b.t);
  const hr = estimateHR(peaks, times);
  setHR(hr);
}

function startCaptureLoop() {
  if (captureTimer) return;
  captureTimer = setInterval(() => {
    updateOverlay();
    processFrame();
  }, FRAME_INTERVAL_MS);
}

function stopCaptureLoop() {
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = null;
}

/* ====================== PHOTO CAPTURE ====================== */

function capturePhoto() {
  if (!cameraReady || !video.videoWidth) {
    alert("Camera not ready yet");
    return;
  }

  const box = {
    x: video.videoWidth * 0.15,
    y: video.videoHeight * 0.15,
    w: video.videoWidth * 0.7,
    h: video.videoHeight * 0.7,
  };

  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, 224, 224);

  capturedDataUrl = canvas.toDataURL("image/jpeg", 0.95);
  capturedPhoto.src = capturedDataUrl;

  photoPreviewContainer.style.display = "block";
  captureBtn.style.display = "none";
  emotionPreview.style.display = "block";
  liveEmotionResult.innerHTML =
    "<span style='color:#64748b'>Analyzing emotions...</span>";

  sendEmotionPrediction(capturedDataUrl);
}

async function sendEmotionPrediction(dataUrl) {
  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64: dataUrl }),
    });
    if (resp.ok) {
      const data = await resp.json();
      lastEmotions = data.emotion;
      lastStress = data.stress_score ?? 0;

      const sorted = Object.entries(lastEmotions)
        .sort((a, b) => b[1] - a[1])
        .map(
          ([e, p]) =>
            `<span class="emotion-chip">${e}: ${(p * 100).toFixed(1)}%</span>`
        )
        .join("");
      liveEmotionResult.innerHTML = sorted;

      usePhotoBtn.disabled = false;
    }
  } catch (e) {
    liveEmotionResult.innerHTML =
      "<span style='color:#ef4444'>Analysis failed</span>";
  }
}

/* ====================== CAMERA CONTROL ====================== */

async function startCamera(useFront = true) {
  if (stream) await stopCamera();

  if (useFront) {
    setFaceState("Starting front camera...");
    if (ppgLoadingOverlay) ppgLoadingOverlay.style.display = "none";
  } else {
    if (ppgLoadingOverlay) ppgLoadingOverlay.style.display = "flex";
  }

  const constraints = {
    video: {
      facingMode: useFront ? "user" : "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    videoTrack = stream.getVideoTracks()[0];

    setupCameraReady();
    await video.play();

    if (!useFront && videoTrack.getCapabilities()?.torch) {
      await setTorch(true);
    }

    if (!useFront) {
      buffer = [];
      hrReadings = [];
      ppgStartTime = Date.now();
      if (continuePpgBtn) continuePpgBtn.disabled = true;
      if (stabilizationText)
        stabilizationText.textContent = "Measuring... 15s remaining";
      if (progressFill) progressFill.style.width = "0%";
    }
  } catch (err) {
    console.error("Camera error:", err);
    setFaceState("Camera access denied or not available ❌");
  }
}

async function stopCamera() {
  stopCaptureLoop();
  if (torchOn) await setTorch(false);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  videoTrack = null;
  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (ppgLoadingOverlay) ppgLoadingOverlay.style.display = "none";
  cameraReady = false;
}

async function setTorch(enable) {
  if (!videoTrack?.getCapabilities()?.torch) return;
  try {
    await videoTrack.applyConstraints({ advanced: [{ torch: enable }] });
    torchOn = enable;
  } catch (e) {
    console.warn("Torch failed:", e);
  }
}

/* ====================== NAVIGATION ====================== */

async function goToStep(step) {
  currentStep = step;

  document
    .querySelectorAll(".step")
    .forEach((s) => s.classList.remove("active"));
  document.querySelector(`.step[data-step="${step}"]`)?.classList.add("active");

  document
    .querySelectorAll(".step-panel")
    .forEach((p) => p.classList.remove("active"));
  const panelId =
    step === 1
      ? "step-face"
      : step === 2
      ? "step-ppg"
      : step === 3
      ? "step-result"
      : "step-ai";
  const panel = document.getElementById(panelId);
  if (panel) panel.classList.add("active");

  if (step >= 3 && videoCard) videoCard.classList.add("hidden");
  else if (videoCard) videoCard.classList.remove("hidden");

  window.scrollTo({ top: 0, behavior: "smooth" });

  if (step === 1) {
    photoPreviewContainer.style.display = "none";
    emotionPreview.style.display = "none";
    captureBtn.style.display = "block";
    captureBtn.disabled = true;
    usePhotoBtn.disabled = true;
    continueFaceBtn.disabled = true;
    setFaceState("Starting camera...");
    await startCamera(true);
  } else if (step === 2) {
    buffer = [];
    hrReadings = [];
    ppgStartTime = Date.now();
    if (continuePpgBtn) continuePpgBtn.disabled = true;
    if (stabilizationText)
      stabilizationText.textContent = "Measuring... 15s remaining";
    if (progressFill) progressFill.style.width = "0%";
    await startCamera(false);
  } else if (step === 3) {
    await stopCamera();
    updateFinalResults();
  } else if (step === 4 && aiLoading && aiContent) {
    await stopCamera();
    aiLoading.style.display = "block";
    aiContent.style.display = "none";

    // calculating ppg

    let ppg_mean = 0;
    let ppg_variance = 0;
    let ppg_length = buffer.length;
    if (buffer.length > 0) {
      const values = buffer.map((b) => b.value);
      ppg_mean = values.reduce((a, b) => a + b, 0) / values.length;
      const mean_sq =
        values.reduce((a, b) => a + Math.pow(b - ppg_mean, 2), 0) /
        values.length;
      ppg_variance = Math.sqrt(mean_sq);
    }

    try {
      const resp = await fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          heart_rate: lastHR,
          stress: lastStress,
          emotions: lastEmotions,
          ppg_mean_signal: ppg_mean,
          ppg_variance: ppg_variance,
          ppg_buffer_length: ppg_length,
        }),
      });

      // In goToStep(4) after fetch
      if (resp.ok) {
        const data = await resp.json();

        let chronicText = "No chronic risks available";
        let dietText = "No diet plan available";
        let exerciseText = "No exercise suggestions available";

        // Chronic - Display probability first
        try {
          const chronic = JSON.parse(data.chronic_prediction);
          let probText = chronic.chronic_risk_probability
            ? `Overall Risk Probability: ${chronic.chronic_risk_probability}\n\n`
            : "";
          chronicText =
            chronic.error ||
            (chronic.diseases
              ? probText + chronic.diseases.join("\n")
              : "No data");
        } catch (e) {
          chronicText = data.chronic_prediction || chronicText;
        }

        // Diet
        try {
          const diet = JSON.parse(data.diet_plan);
          dietText = diet.error || diet.diet || dietText;
        } catch (e) {
          dietText = data.diet_plan || dietText;
        }

        // Exercise
        try {
          const exercise = JSON.parse(data.exercise_suggestion);
          exerciseText = exercise.error || exercise.exercises || exerciseText;
        } catch (e) {
          exerciseText = data.exercise_suggestion || exerciseText;
        }

        if (chronicPredictionEl) chronicPredictionEl.textContent = chronicText;
        if (dietPlanEl) dietPlanEl.textContent = dietText;
        if (exerciseSuggestionEl)
          exerciseSuggestionEl.textContent = exerciseText;

        aiLoading.style.display = "none";
        aiContent.style.display = "block";
      } else {
        const msg = "Server error – check backend";
        if (chronicPredictionEl) chronicPredictionEl.textContent = msg;
        if (dietPlanEl) dietPlanEl.textContent = msg;
        if (exerciseSuggestionEl) exerciseSuggestionEl.textContent = msg;
        aiLoading.style.display = "none";
        aiContent.style.display = "block";
      }
    } catch (err) {
      console.error("AI Step error:", err);
      const msg = "AI analysis failed – check console/backend logs";
      if (chronicPredictionEl) chronicPredictionEl.textContent = msg;
      if (dietPlanEl) dietPlanEl.textContent = msg;
      if (exerciseSuggestionEl) exerciseSuggestionEl.textContent = msg;
      aiLoading.style.display = "none";
      aiContent.style.display = "block";
    }
  }
}

/* ====================== EVENT LISTENERS - AT THE VERY END ====================== */

if (captureBtn) captureBtn.addEventListener("click", capturePhoto);
if (retakeBtn) retakeBtn.addEventListener("click", () => goToStep(1));
if (usePhotoBtn)
  usePhotoBtn.addEventListener(
    "click",
    () => (continueFaceBtn.disabled = false)
  );
if (continueFaceBtn)
  continueFaceBtn.addEventListener("click", () => goToStep(2));
if (continuePpgBtn) continuePpgBtn.addEventListener("click", () => goToStep(3));
if (restartBtn) restartBtn.addEventListener("click", () => goToStep(1));

if (aiAnalysisBtn) aiAnalysisBtn.addEventListener("click", () => goToStep(4));
if (restartFromAiBtn)
  restartFromAiBtn.addEventListener("click", () => goToStep(1));

/* ====================== START APP ====================== */
(async () => {
  await goToStep(1);
})();
