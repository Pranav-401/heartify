# backend/processing.py
# processing.py
# - contains modular functions for image/frame processing with OpenCV
# - importable by server.py or by tests/other scripts
from typing import Tuple, Optional, Dict
import numpy as np
import cv2

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

def decode_jpeg_bytes_to_bgr(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decode JPEG/PNG binary to BGR numpy array."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def detect_largest_face(bgr_img: np.ndarray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60)) -> Optional[Tuple[int,int,int,int]]:
    """
    Detect faces and return the bounding box of the largest face as (x,y,w,h).
    Returns None if no face found.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    if len(faces) == 0:
        return None
    # choose largest face area
    largest = max(faces, key=lambda r: r[2]*r[3])
    x,y,w,h = [int(v) for v in largest]
    return x,y,w,h

def expand_bbox(x:int,y:int,w:int,h:int, img_shape:Tuple[int,int], expand_ratio:float=0.25) -> Tuple[int,int,int,int]:
    """Expand bbox by expand_ratio while staying inside image bounds."""
    h_img, w_img = img_shape[0], img_shape[1]
    ex_w = int(w * expand_ratio)
    ex_h = int(h * expand_ratio)
    x1 = max(0, x - ex_w)
    y1 = max(0, y - ex_h)
    x2 = min(w_img, x + w + ex_w)
    y2 = min(h_img, y + h + ex_h)
    return x1, y1, x2 - x1, y2 - y1

def compute_mean_green_of_roi(bgr_img: np.ndarray, bbox: Tuple[int,int,int,int]) -> float:
    """
    Compute mean green-channel intensity inside ROI defined by bbox (x,y,w,h).
    Returns float mean (0..255). If ROI invalid, returns 0.0.
    """
    x,y,w,h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    roi = bgr_img[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    mean_g = float(np.mean(roi[:,:,1]))
    return mean_g

def annotate_frame_with_bbox(bgr_img: np.ndarray, bbox: Tuple[int,int,int,int], color=(0,255,0), thickness=2) -> np.ndarray:
    """Return a copy of image annotated with rectangle."""
    img = bgr_img.copy()
    x,y,w,h = bbox
    cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness)
    return img

def process_frame_bytes(img_bytes: bytes) -> Dict:
    """
    High-level processing function:
    - decode bytes
    - detect face (largest)
    - expand bbox a bit
    - compute mean green
    Returns a dict ready to JSONify.
    """
    img = decode_jpeg_bytes_to_bgr(img_bytes)
    if img is None:
        return {"ok": False, "error": "decode_failed"}
    face = detect_largest_face(img)
    if face is None:
        return {"ok": True, "face_found": False, "mean_green": None}
    x,y,w,h = face
    ex_bbox = expand_bbox(x,y,w,h, img.shape, expand_ratio=0.25)
    mean_green = compute_mean_green_of_roi(img, ex_bbox)
    # normalized mean in 0..1
    mean_green_norm = mean_green / 255.0
    return {
        "ok": True,
        "face_found": True,
        "bbox": {"x": int(ex_bbox[0]), "y": int(ex_bbox[1]), "w": int(ex_bbox[2]), "h": int(ex_bbox[3])},
        "mean_green": mean_green,            # 0..255
        "mean_green_norm": mean_green_norm  # 0..1
    }