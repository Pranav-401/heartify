# backend/rppg/rppg_inference.py

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import List, Dict

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS = 30
LOW_HZ = 0.7    # 42 BPM
HIGH_HZ = 4.0   # 240 BPM


# ---------------- MINIMAL PHYSNET ----------------
class PhysNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, (3, 5, 5), padding=(1, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (3, 3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)  # (B, 32, T)
        x = x.permute(0, 2, 1)         # (B, T, 32)
        x = self.fc(x).squeeze(-1)     # (B, T)
        return x


# ---------------- MODEL LOADER ----------------
_model = None

def load_model():
    global _model
    if _model is None:
        _model = PhysNet().to(DEVICE)
        _model.eval()
    return _model


# ---------------- SIGNAL UTILS ----------------
def bandpass(signal, fs):
    nyq = 0.5 * fs
    low = LOW_HZ / nyq
    high = HIGH_HZ / nyq
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


def estimate_bpm(ppg, fs):
    filtered = bandpass(ppg, fs)
    peaks, _ = find_peaks(filtered, distance=fs * 0.4)
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / fs
    return round(60.0 / np.mean(intervals), 1)


# ---------------- MAIN API ----------------
def predict_heart_rate(frames: List[np.ndarray]) -> Dict:
    if len(frames) < FPS * 5:
        raise ValueError("Need at least 5 seconds of video")

    video = np.stack(frames).astype(np.float32) / 255.0
    video = torch.from_numpy(video).permute(3, 0, 1, 2)
    video = video.unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        ppg = model(video).cpu().numpy()[0]

    bpm = estimate_bpm(ppg, FPS)

    return {
        "bpm": bpm,
        "signal_quality": "good" if bpm else "poor",
        "ppg_length": len(ppg)
    }
