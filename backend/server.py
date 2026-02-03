# backend/server.py

import os
import io
import base64
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rppg.rppg_inference import predict_heart_rate
from dotenv import load_dotenv

load_dotenv()

# ========== Logging ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heartify")

# ========== Config ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODEL_ENABLED = os.environ.get("ENABLE_MODEL", "1") in ("1", "true", "True")

_model = None
MODEL_LOADED = False
_model_load_attempted = False

# ========== App ==========
app = FastAPI(title="Heartify Backend (Robust)")

# ========== CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Frontend ==========
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    logger.info("Frontend static mounted at /static")
else:
    logger.info("No frontend directory found — API-only mode")

# =====================================================================
# ======================= REQUEST MODELS ===============================
# =====================================================================

class PredictRequest(BaseModel):
    image_b64: str
    ppg_features: Optional[Dict[str, float]] = None
    meta: Optional[Dict[str, Any]] = None


class HRRequest(BaseModel):
    frames: List[str]
    fps: int = 30

class AnalyzeRequest(BaseModel):
    heart_rate: float
    stress: float
    emotions: Dict[str, float]
    ppg_mean_signal: Optional[float] = 0
    ppg_variance: Optional[float] = 0
    ppg_buffer_length: Optional[int] = 0

# =====================================================================
# ======================= UTILITY FUNCTIONS ============================
# =====================================================================

def decode_image_b64(image_b64: str) -> Image.Image:
    if ";base64," in image_b64:
        image_b64 = image_b64.split(";base64,")[-1]

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("Image too large (>10MB)")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")


def decode_frame(b64_img: str):
    if ";base64," in b64_img:
        b64_img = b64_img.split(";base64,")[-1]

    img_bytes = base64.b64decode(b64_img)
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Invalid frame data")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def calculate_stress_score(probs: Dict[str, float]) -> float:
    stress_emotions = {"angry", "fear", "disgust", "sad", "surprise"}
    total, stress = 0.0, 0.0

    for k, v in probs.items():
        v = float(v)
        total += v
        if any(s in k.lower() for s in stress_emotions):
            stress += v

    return round(stress / total, 4) if total > 0 else 0.0


# =====================================================================
# ======================= MODEL LOADING ================================
# =====================================================================

def try_load_model_once():
    global _model, MODEL_LOADED, _model_load_attempted
    if _model_load_attempted:
        return

    _model_load_attempted = True

    if not MODEL_ENABLED:
        logger.info("Emotion model disabled by env flag.")
        return

    try:
        from model_utils import get_model_singleton
        _model = get_model_singleton()
        MODEL_LOADED = True
        logger.info("Emotion model loaded successfully.")
    except Exception as e:
        logger.warning("Emotion model failed to load: %s", e)
        _model = None
        MODEL_LOADED = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    try_load_model_once()
    yield


app.router.lifespan_context = lifespan

# =====================================================================
# ============================ ROUTES =================================
# =====================================================================

@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_enabled": MODEL_ENABLED,
        "emotion_model_loaded": MODEL_LOADED,
    }


@app.get("/", response_class=HTMLResponse)
async def index_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h3>Heartify backend running</h3>")


# -------------------- EMOTION (UNCHANGED) -----------------------------

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        pil = decode_image_b64(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if MODEL_ENABLED and MODEL_LOADED and _model is not None:
        try:
            res = _model.predict(pil)
            probs = res.get("probs", res) if isinstance(res, dict) else res
            stress = calculate_stress_score(probs) if isinstance(probs, dict) else None
            return {
                "ok": True,
                "model": "real",
                "emotion": probs,
                "stress_score": stress,
            }
        except Exception as e:
            logger.exception("Emotion inference failed: %s", e)
            raise HTTPException(status_code=500, detail="Inference error")

    # No dummy — fail clearly
    raise HTTPException(
        status_code=503,
        detail="Emotion model not loaded. Set ENABLE_MODEL=1 and ensure TensorFlow is installed."
    )


# -------------------- HEART RATE (NEW & SAFE) --------------------------

@app.post("/predict_hr")
async def predict_hr(req: HRRequest):
    if len(req.frames) < req.fps * 5:
        raise HTTPException(
            status_code=400,
            detail="Need at least 5 seconds of frames",
        )

    try:
        frames = [decode_frame(f) for f in req.frames]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = predict_heart_rate(frames)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "mode": "rPPG",
        "bpm": result["bpm"],
        "signal_quality": result["signal_quality"],
        "ppg_length": result["ppg_length"],
    }


# --------------------- AI Agent ----------------------------------------

@app.post("/analyze")
async def analyze_results(req: AnalyzeRequest):
    from agents.agent_chronic import predict_chronic_disease
    from agents.agent_diet import suggest_diet_plan
    from agents.agent_exercise import suggest_exercise

    chronic_raw = predict_chronic_disease(req.heart_rate, req.stress, req.emotions)
    diet_raw = suggest_diet_plan(req.heart_rate, req.stress, req.emotions, chronic_raw)
    exercise_raw = suggest_exercise(req.heart_rate, req.stress, req.emotions, chronic_raw, diet_raw)

    # Log to Excel (includes chronic_prob from AI Agent 1)
    log_to_excel(
    heart_rate=req.heart_rate,
    stress=req.stress,
    emotions=req.emotions,
    chronic_prediction=chronic_raw,
    ppg_mean_signal=req.ppg_mean_signal or 0,
    ppg_variance=req.ppg_variance or 0,
    ppg_buffer_length=req.ppg_buffer_length or 0
)

    return {
        "ok": True,
        "chronic_prediction": chronic_raw,
        "diet_plan": diet_raw,
        "exercise_suggestion": exercise_raw
    }
# -------------------- SPA FALLBACK ------------------------------------

@app.get("/{full_path:path}")
async def spa_fallback(full_path: str, request: Request):
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not Found")

# ------------------------ data collection --------------------------------
def log_to_excel(heart_rate, stress, emotions, chronic_prediction, ppg_mean_signal=0, ppg_variance=0, ppg_buffer_length=0):
    file_path = os.path.join(BASE_DIR, "data", "training_data.xlsx")
    
    # Extract chronic_prob from AI Agent 1
    chronic_prob = 0.5  # Default
    try:
        chronic_data = json.loads(chronic_prediction)
        chronic_prob = chronic_data.get("chronic_risk_probability", 0.5)
        logger.info(f"Extracted chronic probability: {chronic_prob}")
    except Exception as e:
        logger.warning(f"Failed to parse chronic probability: {e}")

    data = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "heart_rate": [heart_rate],
        "stress": [stress],
        "angry": [emotions.get("angry", 0)],
        "disgust": [emotions.get("disgust", 0)],
        "fear": [emotions.get("fear", 0)],
        "happy": [emotions.get("happy", 0)],
        "sad": [emotions.get("sad", 0)],
        "surprise": [emotions.get("surprise", 0)],
        "neutral": [emotions.get("neutral", 0)],
        "ppg_mean_signal": [ppg_mean_signal],
        "ppg_variance": [ppg_variance],
        "ppg_buffer_length": [ppg_buffer_length],
        "chronic_risk_probability": [chronic_prob]  # Real AI value
    }

    df_new = pd.DataFrame(data)

    try:
        if os.path.exists(file_path):
            df_existing = pd.read_excel(file_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_excel(file_path, index=False)
        logger.info(f"Data logged to Excel (chronic_prob: {chronic_prob})")
    except Exception as e:
        logger.error(f"Failed to save to Excel: {e}")
# =====================================================================
# ============================ RUN ====================================
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)