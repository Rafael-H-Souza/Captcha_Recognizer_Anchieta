from __future__ import annotations
import os
import glob
import numpy as np
import tensorflow as tf
import cv2
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.config.settings import config
from src.utils.logger import get_logger


# ---------------------------------------
# LOGGER
# ---------------------------------------
logger = get_logger("Predictor")
if logger is None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Predictor")


# ---------------------------------------
# Custom Layer (para compatibilidade)
# ---------------------------------------
class GetItemCompat(tf.keras.layers.Layer):
    def __init__(self, index=None, **kw):
        super().__init__(**kw)
        self.index = index

    def call(self, x):
        if self.index is None:
            return x
        return tf.gather(x, self.index, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg["index"] = self.index
        return cfg


# ---------------------------------------
# Helper
# ---------------------------------------
def safe_mkdir(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def ensure_models_dirs():
    base = config.get("models_dir", "models")
    apurados = os.path.join(base, "apurados")
    exported = os.path.join(base, "exported")
    safe_mkdir(apurados)
    safe_mkdir(exported)
    return apurados, exported


# ---------------------------------------
# PREDICTOR COMPLETO
# ---------------------------------------
class Predictor:

    def __init__(self, model_path: Optional[str] = None, debug: bool = False):
        self.debug = debug
        self.logger = logger

        self.models_dir, self.exported_dir = ensure_models_dirs()

        if "charset" not in config:
            raise KeyError("Config precisa de 'charset'")

        self.charset = config["charset"]
        self.num_to_char = {i: c for i, c in enumerate(self.charset)}

        self.model = None
        self.model_path = model_path

        if model_path:
            self.model = self._load_model(model_path)
        else:
            raise RuntimeError("Precisa de um único modelo carregado.")

    # ---------------------------
    # LOAD MODELS
    # ---------------------------
    def _compile(self, model):
        try:
            model.compile(optimizer="adam", loss="categorical_crossentropy")
        except:
            pass
        return model

    def _load_model(self, path: str):
        try:
            with tf.keras.utils.custom_object_scope({"GetItemCompat": GetItemCompat}):
                m = tf.keras.models.load_model(path, compile=False)
            return self._compile(m)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {path}: {e}")
            return None

    # ---------------------------
    # PREPROCESS
    # ---------------------------
    def _preprocess(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_path)

        # CLAHE
        img = cv2.createCLAHE(2.0, (8, 8)).apply(img)

        # Threshold
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morfologia
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        # Remoção de linhas
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if abs(y1 - y2) < 5:
                    cv2.line(img, (x1, y1), (x2, y2), 0, 2)

        # Padronizar tamanho
        img = cv2.resize(img, (config["image_width"], config["image_height"]))
        img = img.astype("float32") / 255.0

        return np.expand_dims(img, axis=(0, -1))

    # ---------------------------
    # EXTRACT ATTEMPTS (TOP-K)
    # ---------------------------
    def _extract_attempts(self, preds, top_k=10):

        preds = np.array(preds)

        # Caso 1: modelo retorna (1, seq_len, charset)
        if preds.ndim == 3:
            seq_len = preds.shape[1]
            outputs = [preds[0, i] for i in range(seq_len)]

        # Caso 2: lista de arrays (multi-output)
        elif isinstance(preds, list):
            outputs = [np.squeeze(p) for p in preds]

        # Caso 3: modelo errado (1, charset)
        elif preds.ndim == 2:
            outputs = [preds[0]]

        else:
            outputs = [np.squeeze(preds)]

        details = []

        for pos, logits in enumerate(outputs):
            logits = np.array(logits)

            idxs = logits.argsort()[::-1][:top_k]

            attempts = []
            for i in idxs:
                char = self.num_to_char.get(int(i), "?")
                tipo = "num" if char.isdigit() else "letra" if char.isalpha() else "outro"

                attempts.append({
                    "char": char,
                    "tipo": tipo,
                    "confidence": float(logits[i])
                })

            details.append({
                "pos": pos,
                "predicted": attempts[0]["char"],
                "attempts": attempts
            })

        return details

    # ---------------------------
    # PUBLIC API
    # ---------------------------
    def predict_with_details(self, image_path: str):
        batch = self._preprocess(image_path)
        preds = self.model.predict(batch, verbose=0)

        if not isinstance(preds, list):
            preds = [preds]

        details = self._extract_attempts(preds, top_k=12)
        text = "".join(d["predicted"] for d in details)

        return text, details

    # ---------------------------
    # DEBUG MODE
    # ---------------------------
    def predict_with_debug(self, image_path: str, top_k=8):

        batch, debug_steps = self._preprocess_debug(image_path)
        preds = self.model.predict(batch, verbose=0)

        if not isinstance(preds, list):
            preds = [preds]

        details = self._extract_attempts(preds, top_k)
        text = "".join(d["predicted"] for d in details)

        return text, debug_steps, details

    def _preprocess_debug(self, image_path: str):

        steps = []
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        steps.append({"step": "Original", "image": img})

        img1 = cv2.createCLAHE(2.0, (8, 8)).apply(img)
        steps.append({"step": "CLAHE", "image": img1})

        img2 = cv2.adaptiveThreshold(
            img1, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        steps.append({"step": "Adaptive Threshold", "image": img2})

        img3 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        steps.append({"step": "Morfologia", "image": img3})

        img4 = cv2.resize(img3, (config["image_width"], config["image_height"]))
        steps.append({"step": "Resize", "image": img4})

        batch = np.expand_dims(img4.astype("float32") / 255.0, axis=(0, -1))

        return batch, steps
