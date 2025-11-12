import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

class UnifiedPredictor:
    def __init__(self, models_dir="models/apurados"):
        self.models = {}
        self._load_all(models_dir)

    def _load_all(self, models_dir):
        for path in os.listdir(models_dir):
            if path.endswith(".keras"):
                name = os.path.splitext(path)[0]
                try:
                    self.models[name] = models.load_model(os.path.join(models_dir, path))
                    print(f"✅ Modelo carregado: {name}")
                except Exception as e:
                    print(f"⚠️ Falha ao carregar {path}: {e}")

    def predict(self, image):
        """Executa predição unificada de acordo com o tipo do modelo"""
        results = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
                results[name] = pred
            except Exception as e:
                results[name] = f"Erro: {e}"

        # Estratégia simples: pegar o modelo que mais confia
        text = self._combine_predictions(results)
        return text

    def _combine_predictions(self, results):
        combined = ""
        for name, pred in results.items():
            if isinstance(pred, np.ndarray):
                decoded = self._decode_prediction(pred)
                combined += decoded
        return combined[:5]  # se for captcha de 5 chars

    def _decode_prediction(self, pred):
        if pred.ndim == 3:
            pred = np.argmax(pred, axis=-1)[0]
            return ''.join(chr(97 + p) for p in pred)  # exemplo simplificado (a-z)
        elif pred.ndim == 2:
            return chr(97 + np.argmax(pred))
        return ""

