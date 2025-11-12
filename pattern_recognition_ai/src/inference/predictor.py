from __future__ import annotations
import os
import glob
import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
import numpy as np
import cv2
import tensorflow as tf
from src.config.settings import config
from src.utils.logger import get_logger

# ---------------------------
# Logger
# ---------------------------
logger = get_logger("Predictor")
if logger is None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Predictor")

# ======================================================
# Camada compatível para GetItem antigo
# ======================================================
class GetItemCompat(tf.keras.layers.Layer):
    def __init__(self, index: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        if self.index is None:
            return inputs
        return tf.gather(inputs, indices=self.index, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"index": self.index})
        return cfg

# ======================================================
# Helpers
# ======================================================
def safe_mkdir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        logger.warning(f"Não foi possível criar pasta: {path}", exc_info=True)

def ensure_models_dirs() -> Tuple[str, str]:
    base_dir = config.get("models_dir", "models")
    models_apurados = os.path.join(base_dir, "apurados")
    models_exported = os.path.join(base_dir, "exported")
    safe_mkdir(models_apurados)
    safe_mkdir(models_exported)
    return models_apurados, models_exported

# ======================================================
# Classe Principal
# ======================================================
class Predictor:
    """
    Predictor para CAPTCHAs.
    Carrega somente modelos .h5 ou .keras.
    """

    def __init__(self, model_path: Optional[str] = None, debug: bool = False):
        self.logger = logger
        self.debug = bool(debug)
        self.models_dir, self.exported_dir = ensure_models_dirs()

        self.model: Optional[tf.keras.Model] = None
        self.model_path: Optional[str] = None
        self.multi_models: Dict[str, tf.keras.Model] = {}

        if "charset" not in config:
            raise KeyError("Config: 'charset' não encontrado em settings.config")
        self.num_to_char = {idx: char for idx, char in enumerate(config["charset"])}

        # Carregamento
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo principal não encontrado: {model_path}")
            self.model_path = model_path
            self.model = self._load_model(model_path)
        else:
            self.multi_models.update(self._load_all_models(self.models_dir))
            self.multi_models.update(self._load_all_models(self.exported_dir))

        if not self.model and not self.multi_models:
            self.logger.warning("⚠️ Nenhum modelo carregado para predição!")

    # ----------------------------
    # Compilação segura
    # ----------------------------
    def _compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        try:
            if isinstance(model.outputs, (list, tuple)):
                metrics = ["accuracy"] * len(model.outputs)
            else:
                metrics = ["accuracy"]
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=metrics)
        except Exception:
            self.logger.warning("Falha ao compilar o modelo.", exc_info=True)
        return model

    # ----------------------------
    # Carregamento seguro
    # ----------------------------
    def _safe_load(self, path: str) -> Optional[tf.keras.Model]:
        try:
            with tf.keras.utils.custom_object_scope({"GetItemCompat": GetItemCompat}):
                model = tf.keras.models.load_model(path, compile=False)
            model = self._compile_model(model)
            self.logger.info(f"✅ Carregado: {os.path.basename(path)}")
            return model
        except Exception as ex:
            self.logger.error(f"Falha ao carregar {path}: {ex}", exc_info=True)
            return None

    def _load_model(self, path: str) -> Optional[tf.keras.Model]:
        return self._safe_load(path)

    def _load_all_models(self, dir_path: str) -> Dict[str, tf.keras.Model]:
        loaded: Dict[str, tf.keras.Model] = {}
        if not os.path.isdir(dir_path):
            self.logger.debug(f"Pasta de modelos inexistente: {dir_path}")
            return loaded

        files = sorted(glob.glob(os.path.join(dir_path, "*.h5")) + glob.glob(os.path.join(dir_path, "*.keras")))
        for fpath in files:
            model = self._safe_load(fpath)
            if model:
                loaded[fpath] = model
        self.logger.info(f"Modelos carregados de {dir_path}: {len(loaded)}")
        return loaded

    # ----------------------------
    # Pré-processamento
    # ----------------------------
    def _preprocess(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.logger.error(f"Imagem não encontrada: {image_path}")
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            edges = cv2.Canny(img, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    if abs(y2 - y1) < 5:
                        cv2.line(img, (x1, y1), (x2, y2), 0, 2)

            img_resized = cv2.resize(img, (config["image_width"], config["image_height"]))
            img_norm = img_resized.astype("float32") / 255.0
            batch = np.expand_dims(img_norm, axis=(0, -1))
            return batch
        except Exception:
            self.logger.error("Erro no pré-processamento da imagem", exc_info=True)
            raise

    # ----------------------------
    # Normalização predictions
    # ----------------------------
    def _normalize_predictions_to_chars(self, predictions) -> List[str]:
        chars: List[str] = []
        try:
            if isinstance(predictions, (list, tuple)):
                for pred in predictions:
                    arr = np.asarray(pred)
                    if arr.ndim == 2:
                        idx = int(np.argmax(arr, axis=-1).flatten()[0])
                    else:
                        idx = int(np.argmax(arr))
                    chars.append(self.num_to_char.get(idx, "?"))
                return chars

            pred = np.asarray(predictions)
            if pred.ndim == 2:
                idx = int(np.argmax(pred, axis=-1).flatten()[0])
                chars.append(self.num_to_char.get(idx, "?"))
                return chars
            if pred.ndim == 3:
                for step in pred[0]:
                    idx = int(np.argmax(step))
                    chars.append(self.num_to_char.get(idx, "?"))
                return chars

            idx = int(np.argmax(pred))
            chars.append(self.num_to_char.get(idx, "?"))
            return chars
        except Exception:
            self.logger.error("Erro ao normalizar predictions para chars", exc_info=True)
            return ["?"]

    # ----------------------------
    # Predição single model
    # ----------------------------
    def _predict_single_model(self, model: tf.keras.Model, image_path: str) -> List[Tuple[str, str]]:
        batch = self._preprocess(image_path)
        try:
            preds = model.predict(batch, verbose=0)
            chars = self._normalize_predictions_to_chars(preds)
            return [(c, ("num" if c.isdigit() else "letra" if c.isalpha() else "outro")) for c in chars]
        except Exception:
            self.logger.error("Erro na predição do modelo", exc_info=True)
            raise

    # ----------------------------
    # API pública
    # ----------------------------
    def predict_with_details(self, image_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        if self.model:
            decoded = self._predict_single_model(self.model, image_path)
            text = "".join(c for c, _ in decoded)
            return text, [{"model": os.path.basename(self.model_path) if self.model_path else "model", "text": text, "types": decoded}]

        if not self.multi_models:
            self.logger.error("Nenhum modelo disponível para predição.")
            raise RuntimeError("Nenhum modelo disponível para predição.")

        results: Dict[str, List[Tuple[str, str]]] = {}
        for path, model in self.multi_models.items():
            try:
                decoded = self._predict_single_model(model, image_path)
                results[path] = decoded
            except Exception:
                self.logger.error(f"Erro ao predizer com {path}", exc_info=True)

        if not results:
            self.logger.error("Falha em todas as predições.")
            raise RuntimeError("Falha em todas as predições.")

        max_len = max(len(r) for r in results.values())
        ensemble_chars: List[str] = []
        for i in range(max_len):
            votes = [r[i][0] for r in results.values() if i < len(r)]
            ensemble_chars.append(Counter(votes).most_common(1)[0][0] if votes else "?")

        ensemble = "".join(ensemble_chars)
        details = [{"model": os.path.basename(p), "text": "".join(c for c, _ in d), "types": d} for p, d in results.items()]
        details.append({"ensemble": ensemble})
        return ensemble, details

    # ----------------------------
    # Conveniência
    # ----------------------------
    def predict_text(self, image_path: str) -> str:
        text, _ = self.predict_with_details(image_path)
        return text
    
    # ======================================================
    # Previsão com Debug / Visualização de Steps
    # ======================================================
    def _preprocess_debug(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Retorna a imagem processada e lista de steps intermediários para debug visual.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.logger.error(f"Imagem não encontrada: {image_path}")
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        debug_steps: List[Dict[str, Any]] = [{"step": "original", "image": img.copy()}]

        # 1️⃣ CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        debug_steps.append({"step": "CLAHE", "image": img.copy()})

        # 2️⃣ Threshold adaptativo
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        debug_steps.append({"step": "Adaptive Threshold", "image": img.copy()})

        # 3️⃣ Morfologia
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        debug_steps.append({"step": "Morfologia", "image": img.copy()})

        # 4️⃣ Detecção de linhas (remover linhas horizontais)
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
        img_lines = img.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if abs(y2 - y1) < 5:
                    cv2.line(img_lines, (x1, y1), (x2, y2), 0, 2)
        debug_steps.append({"step": "Remoção de linhas horizontais", "image": img_lines.copy()})
        img = img_lines

        # 5️⃣ Resize e normalização
        img_resized = cv2.resize(img, (config["image_width"], config["image_height"]))
        img_norm = img_resized.astype("float32") / 255.0
        batch = np.expand_dims(img_norm, axis=(0, -1))
        debug_steps.append({"step": "Resize e Normalização", "image": img_resized.copy()})

        return batch, debug_steps

    # ======================================================
    # Predição com Debug
    # =====================================================
    def predict_with_debug(self, image_path: str, top_k=5):
        debug_steps = []

        # Carregamento original
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        debug_steps.append({"step": "Original", "image": img.copy()})

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        debug_steps.append({"step": "CLAHE", "image": img_clahe})

        # Threshold adaptativo
        img_thresh = cv2.adaptiveThreshold(
            img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        debug_steps.append({"step": "Adaptive Threshold", "image": img_thresh})

        # Morfologia
        kernel = np.ones((2,2), np.uint8)
        img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
        debug_steps.append({"step": "Morfologia", "image": img_morph})

        # Predição
        batch = np.expand_dims(img_morph.astype("float32")/255.0, axis=(0,-1))
        preds = self.model.predict(batch, verbose=0)

        # Para cada caractere, pega top-k tentativas
        all_attempts = []
        for step in preds if isinstance(preds, list) else [preds]:
            arr = np.asarray(step)
            if arr.ndim == 2:
                arr = arr[0]  # só primeira amostra
            top_indices = arr.argsort()[-top_k:][::-1]
            attempts = [(self.num_to_char.get(idx, "?"), float(arr[idx])) for idx in top_indices]
            all_attempts.append(attempts)

        # Construir detalhes
        details = all_attempts
        result = "".join([attempts[0][0] for attempts in all_attempts])  # pega a primeira tentativa como principal
        return result, debug_steps, details
