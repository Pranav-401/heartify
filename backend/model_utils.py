# backend/model_utils.py
import os
import logging
from typing import Dict
import numpy as np
from PIL import Image

logger = logging.getLogger("heartify.model_utils")
logger.setLevel(logging.INFO)

MODEL_ID_DEFAULT = os.environ.get("HF_MODEL_ID", "steph2502/face-emotion-model")
HF_CACHE = os.environ.get("HF_HOME")  # optional custom HF cache dir

# Lazy imports
_tensorflow_available = False
try:
    import tensorflow as tf
    from huggingface_hub import hf_hub_download
    _tensorflow_available = True
except Exception as e:
    logger.warning("TensorFlow/huggingface-hub not available: %s", e)
    _tensorflow_available = False

class HFModelWrapper:
    def __init__(self, model_id: str = MODEL_ID_DEFAULT, cache_dir: str = None):
        if not _tensorflow_available:
            raise RuntimeError("tensorflow and huggingface-hub not installed in the venv.")
        self.model_id = model_id
        self.cache_dir = cache_dir
        logger.info("Downloading .h5 model from %s (cache_dir=%s)...", model_id, cache_dir or "default")
        
        model_path = hf_hub_download(repo_id=model_id, filename="face_emotionModel.h5", cache_dir=cache_dir)
        
        logger.info("Loading Keras model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        
        # Standard FER2013 labels (assuming based on common models)
        self.labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def predict(self, pil_image: Image.Image) -> Dict[str, float]:
        """
        Accepts PIL.Image, returns dict of label -> probability (floats, sum=1).
        Assumes standard 48x48 grayscale input for FER models.
        """
        if not isinstance(pil_image, Image.Image):
            raise ValueError("predict expects a PIL.Image")

        # Preprocess: grayscale, resize to 48x48, normalize
        img = pil_image.convert("L")  # grayscale
        img = img.resize((48, 48))
        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 48, 48, 1)  # batch, h, w, ch

        # Predict
        probs = self.model.predict(arr, verbose=0)[0]

        # Map to labels
        result = {label: float(p) for label, p in zip(self.labels, probs)}
        return result

# Singleton holder
_singleton = None
def get_model_singleton():
    global _singleton
    if _singleton is not None:
        return _singleton
    if not _tensorflow_available:
        raise RuntimeError("tensorflow and huggingface-hub not installed (pip install tensorflow huggingface-hub).")
    cache_dir = HF_CACHE if HF_CACHE else None
    try:
        _singleton = HFModelWrapper(MODEL_ID_DEFAULT, cache_dir=cache_dir)
        logger.info("Model singleton created.")
        return _singleton
    except Exception as e:
        logger.exception("Failed to load HF model: %s", e)
        _singleton = None
        raise