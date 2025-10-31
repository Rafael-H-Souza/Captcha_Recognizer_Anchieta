import os
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.config.settings import config
from src.utils.logger import get_logger


class Predictor:
    def __init__(self, model_path: Optional[str] = None, debug: bool = False):
        self.logger = get_logger(__name__)
        self.model_path = model_path or config["final_model_path"]
        self.model = self._load_model(self.model_path)
        self.num_to_char = {idx: char for idx, char in enumerate(config["charset"])}
        self.debug = debug
        self.last_details: List[Dict[str, Any]] = []

    def _load_model(self, model_path: str) -> Optional[tf.keras.Model]:
        if not os.path.exists(model_path):
            self.logger.error("Modelo não encontrado em %s", model_path)
            return None
        try:
            model = tf.keras.models.load_model(model_path)
            self.logger.info("Modelo carregado de %s", model_path)
            return model
        except Exception as exc:
            self.logger.error("Falha ao carregar o modelo: %s", exc)
            return None

    def _preprocess(self, image_path: str) -> Dict[str, Any]:
        debug_steps = {}
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada em {image_path}")
        debug_steps["original"] = img.copy()

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        debug_steps["contraste"] = img.copy()

        # Threshold adaptativo
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        debug_steps["binarizada"] = img.copy()

        # Remover ruídos
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        debug_steps["limpa"] = img.copy()

        # Remover linhas horizontais
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:
                    cv2.line(img, (x1, y1), (x2, y2), 0, 2)
        debug_steps["sem_linha"] = img.copy()

        # Redimensionar e normalizar
        img_resized = cv2.resize(img, (config["image_width"], config["image_height"]))
        img_norm = img_resized.astype("float32") / 255.0
        img_norm = np.expand_dims(img_norm, axis=-1)
        img_norm = np.expand_dims(img_norm, axis=0)
        debug_steps["final_input"] = img_resized.copy()

        return {"processed": img_norm, "debug_steps": debug_steps}

    def _decode_prediction(self, predictions: np.ndarray) -> str:
        decoded_text = ""
        for pred in predictions:
            char_index = int(np.argmax(pred, axis=1)[0])
            decoded_text += self.num_to_char.get(char_index, "?")
        return decoded_text

    def predict(self, image_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Modelo não carregado.")
        processed = self._preprocess(image_path)["processed"]
        predictions = self.model.predict(processed)
        result = self._decode_prediction(predictions)
        self.logger.info("Resultado da predição para '%s': %s", image_path, result)
        return result

    def predict_with_details(self, image_path: str, min_confidence: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
        if self.model is None:
            raise RuntimeError("Modelo não carregado.")

        processed = self._preprocess(image_path)["processed"]
        predictions = self.model.predict(processed)
        decoded_text = ""
        details = []

        for i, pred in enumerate(predictions):
            char_index = int(np.argmax(pred, axis=1)[0])
            char = self.num_to_char.get(char_index, "?")
            confidence = float(np.max(pred, axis=1)[0])

            confidence_alpha = float(np.sum(pred[0, [idx for idx, c in self.num_to_char.items() if c.isalpha()]]))
            confidence_number = float(np.sum(pred[0, [idx for idx, c in self.num_to_char.items() if c.isdigit()]]))

            if char.isdigit() and confidence >= min_confidence:
                char_type = "number"
            elif char.isalpha() and confidence >= min_confidence:
                char_type = "alpha"
            else:
                char_type = "alpha-numeric"

            decoded_text += char
            details.append({
                "position": i + 1,
                "char": char,
                "confidence": confidence,
                "type": char_type,
                "confidence_alpha": confidence_alpha,
                "confidence_number": confidence_number
            })

        self.last_details = details
        self.logger.info("Predição detalhada para '%s': %s", image_path, decoded_text)
        return decoded_text, details

    def predict_with_debug(self, image_path: str) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predição completa com debug:
        - Retorna texto, debug_steps (imagens intermediárias) e detalhes de cada caractere
        """
        if self.model is None:
            raise RuntimeError("Modelo não carregado.")

        preprocess_data = self._preprocess(image_path)
        processed = preprocess_data["processed"]
        debug_steps = preprocess_data["debug_steps"]

        decoded_text, details = self.predict_with_details(image_path)
        self.last_details = details

        return decoded_text, debug_steps, details
