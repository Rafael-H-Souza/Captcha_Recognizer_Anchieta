import os
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

from src.config.settings import config
from src.utils.logger import get_logger


class Predictor:
    def __init__(self, model_path: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.model_path = model_path or config["final_model_path"]
        self.model = self._load_model(self.model_path)
        self.num_to_char = {idx: char for idx, char in enumerate(config["charset"])}

    def _load_model(self, model_path: str) -> Optional[tf.keras.Model]:
        if not os.path.exists(model_path):
            self.logger.error("Modelo não encontrado em %s", model_path)
            return None
        try:
            model = tf.keras.models.load_model(model_path)
            self.logger.info("Modelo carregado de %s", model_path)
            return model
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Falha ao carregar o modelo: %s", exc)
            return None

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada em {image_path}")

        img = cv2.resize(img, (config["image_width"], config["image_height"]))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img

    def _decode_prediction(self, predictions: list[np.ndarray]) -> str:
        decoded_text = ""
        for pred in predictions:
            char_index = int(np.argmax(pred, axis=1)[0])
            decoded_text += self.num_to_char.get(char_index, "?")
        return decoded_text

    def predict(self, image_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Modelo não carregado. Treine ou forneça um caminho válido.")

        processed_image = self._preprocess(image_path)
        predictions = self.model.predict(processed_image)
        result = self._decode_prediction(predictions)
        self.logger.info("Resultado da predição para '%s': %s", image_path, result)
        return result

    def predict_with_details(self, image_path: str):
        """
        ---- Retorna o texto completo + detalhes (caractere e confiança por posição).
        """
        if self.model is None:
            raise RuntimeError("Modelo não carregado. Treine ou forneça um caminho válido.")

        processed_image = self._preprocess(image_path)
        predictions = self.model.predict(processed_image)
        
        decoded_text = ""
        details = []

        for i, pred in enumerate(predictions):
            char_index = int(np.argmax(pred, axis=1)[0])
            confidence = float(np.max(pred, axis=1)[0])
            char = self.num_to_char.get(char_index, "?")

            decoded_text += char
            details.append({
                "position": i + 1,
                "char": char,
                "confidence": confidence
            })

        self.logger.info("Predição detalhada para '%s': %s", image_path, decoded_text)
        return decoded_text, details
