"""
src/inference/predictor.py
Versão reescrita e hardening do Predictor de CAPTCHAs.

Funcionalidades:
- Carregamento robusto de modelos (.h5/.keras) com fallback (safe_mode=False).
- Registro de erros completo (traceback).
- Compatibilidade com modelos antigos que usaram camadas GetItem / Lambda.
- Normalização de diversas formas de saída do model.predict().
- Pré-processamento sólido com OpenCV.
- Suporte para ensemble (multiples modelos) e predição single-model.
- Export helper para converter .h5 -> .keras quando possível.
"""

from __future__ import annotations

import os
import glob
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

import numpy as np
import cv2
import tensorflow as tf

from src.config.settings import config
from src.utils.logger import get_logger

# ---------------------------
# Configuração local de logger
# ---------------------------
logger = get_logger("Predictor")
# fallback: se get_logger retornar None, crie um logger básico
if logger is None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Predictor")


# ======================================================
# Camada compatível para substituir GetItem antigo
# ======================================================
class GetItemCompat(tf.keras.layers.Layer):
    """Camada substituta que evita slicing posicional problemático."""
    def __init__(self, index: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        if self.index is None:
            return inputs
        # tf.gather garante compatibilidade com Keras moderno
        return tf.gather(inputs, indices=self.index, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"index": self.index})
        return cfg


# ======================================================
# Helper functions
# ======================================================
def safe_mkdir(path: str) -> None:
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
# Classe principal Predictor
# ======================================================
class Predictor:
    """
    Predictor para CAPTCHAs.
    - Se model_path fornecido, usa um modelo específico.
    - Caso contrário, carrega todos os modelos nas pastas (apurados, exported)
      e permite predição por ensemble.
    """

    def __init__(self, model_path: Optional[str] = None, debug: bool = False):
        self.logger = logger
        self.debug = bool(debug)

        # diretórios de modelos (padrão vindo do settings)
        self.models_dir, self.exported_dir = ensure_models_dirs()

        self.model: Optional[tf.keras.Model] = None
        self.model_path: Optional[str] = None
        self.multi_models: Dict[str, tf.keras.Model] = {}

        # charset: lista/string vindo do settings
        if "charset" not in config:
            raise KeyError("Config: 'charset' não encontrado em src.config.settings.config")
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
        """
        Compila o modelo com a quantidade correta de métricas para multi-output.
        Não altera o estado caso a compilação falhe (retorna o modelo mesmo assim).
        """
        try:
            if isinstance(model.outputs, (list, tuple)):
                metrics = ["accuracy"] * len(model.outputs)
            else:
                metrics = ["accuracy"]
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=metrics)
            self.logger.debug(f"Modelo compilado com metrics={metrics}")
        except Exception:
            # registra e retorna o modelo sem lançar
            self.logger.warning("Falha ao compilar o modelo (continuando sem compile).", exc_info=True)
        return model

    # ----------------------------
    # Carregamento robusto com traceback
    # ----------------------------
    def _safe_load(self, path: str) -> Optional[tf.keras.Model]:
        """
        Tenta carregar um modelo usando as estratégias:
        1) load_model(path, compile=False) com custom_object_scope vazio
        2) load_model(path, compile=False, safe_mode=False) (mais permissivo)
        Também registra traceback completo em caso de falhas.
        """
        try:
            with tf.keras.utils.custom_object_scope({"GetItem": GetItemCompat, "GetItemCompat": GetItemCompat}):
                model = tf.keras.models.load_model(path, compile=False)
            model = self._compile_model(model)
            self.logger.info(f"✅ Carregado (modo padrão): {os.path.basename(path)}")
            return model
        except Exception as ex1:
            self.logger.warning(f"Falha ao carregar (modo padrão) {path}: {ex1}", exc_info=True)

        # tentativa com safe_mode=False (desativa checagem de lambdas)
        try:
            with tf.keras.utils.custom_object_scope({"GetItem": GetItemCompat, "GetItemCompat": GetItemCompat}):
                model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
            model = self._compile_model(model)
            self.logger.info(f"✅ Carregado (safe_mode=False): {os.path.basename(path)}")
            return model
        except Exception as ex2:
            self.logger.error(f"Falha definitiva ao carregar {path}: {ex2}", exc_info=True)
            return None

    # ----------------------------
    # Helpers de carregamento
    # ----------------------------
    def _load_model(self, path: str) -> Optional[tf.keras.Model]:
        model = self._safe_load(path)
        if not model:
            self.logger.error(f"❌ Erro ao carregar modelo: {path}")
        return model

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
    # Pré-processamento de imagens
    # ----------------------------
    def _preprocess(self, image_path: str) -> np.ndarray:
        """
        Abre a imagem, converte para grayscale, aplica CLAHE, binariza, limpa ruído,
        remove linhas horizontais e redimensiona para config image_width x image_height.
        Retorna array shape (1, H, W, 1), dtype float32.
        """
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
    # Normalização de saída do predict -> lista de chars
    # ----------------------------
    def _normalize_predictions_to_chars(self, predictions) -> List[str]:
        """
        Suporta:
         - list/tuple de arrays (multi-output, cada saída um char)
         - ndarray com ndim==2 -> (1, classes) -> único char
         - ndarray com ndim==3 -> (1, seq, classes) -> sequência de chars
        Retorna lista de chars na ordem.
        """
        chars: List[str] = []

        try:
            # multi-output (lista de arrays)
            if isinstance(predictions, (list, tuple)):
                for pred in predictions:
                    arr = np.asarray(pred)
                    # suportar (batch, classes) ou (classes,)
                    if arr.ndim == 2:
                        idx = int(np.argmax(arr, axis=-1).flatten()[0])
                    else:
                        idx = int(np.argmax(arr))
                    chars.append(self.num_to_char.get(idx, "?"))
                return chars

            pred = np.asarray(predictions)

            if pred.ndim == 2:
                # (1, classes)
                idx = int(np.argmax(pred, axis=-1).flatten()[0])
                chars.append(self.num_to_char.get(idx, "?"))
                return chars

            if pred.ndim == 3:
                # (1, seq, classes) -> iterate seq
                seq = pred[0]
                for step in seq:
                    idx = int(np.argmax(step))
                    chars.append(self.num_to_char.get(idx, "?"))
                return chars

            # fallback
            idx = int(np.argmax(pred))
            chars.append(self.num_to_char.get(idx, "?"))
            return chars
        except Exception:
            self.logger.error("Erro ao normalizar predictions para chars", exc_info=True)
            return ["?"]

    # ----------------------------
    # Predição com um modelo
    # ----------------------------
    def _predict_single_model(self, model: tf.keras.Model, image_path: str) -> List[Tuple[str, str]]:
        """
        Retorna uma lista de tuplas (char, tipo) onde tipo é 'num'/'letra'/'outro'.
        """
        batch = self._preprocess(image_path)
        try:
            preds = model.predict(batch, verbose=0)
            chars = self._normalize_predictions_to_chars(preds)
            decoded = [(c, ("num" if c.isdigit() else "letra" if c.isalpha() else "outro")) for c in chars]
            return decoded
        except Exception:
            self.logger.error("Erro na predição do modelo", exc_info=True)
            raise

    # ----------------------------
    # API pública
    # ----------------------------
    def predict_with_details(self, image_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Se foi fornecido um modelo único (self.model), usa ele.
        Caso contrário, faz predição com todos os modelos carregados e aplica ensemble por voto.
        Retorna (texto_predito, detalhes) onde detalhes é lista com info por modelo + item final 'ensemble'.
        """
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

        # ensemble por posição
        max_len = max(len(r) for r in results.values())
        ensemble_chars: List[str] = []
        for i in range(max_len):
            votes = [r[i][0] for r in results.values() if i < len(r)]
            if not votes:
                ensemble_chars.append("?")
            else:
                ensemble_chars.append(Counter(votes).most_common(1)[0][0])

        ensemble = "".join(ensemble_chars)
        details = [{"model": os.path.basename(p), "text": "".join(c for c, _ in d), "types": d} for p, d in results.items()]
        details.append({"ensemble": ensemble})
        return ensemble, details

    # ----------------------------
    # Conveniência: retorna só o texto
    # ----------------------------
    def predict_text(self, image_path: str) -> str:
        text, _ = self.predict_with_details(image_path)
        return text

    # ----------------------------
    # Util: tenta reexportar .h5 -> .keras (útil para compatibilidade futura)
    # ----------------------------
    def convert_h5_to_keras(self, h5_path: str) -> Optional[str]:
        """
        Se possível, tenta reabrir o .h5 e salvar no formato native Keras (.keras).
        Retorna caminho .keras gerado ou None.
        """
        try:
            with tf.keras.utils.custom_object_scope({"GetItem": GetItemCompat}):
                m = tf.keras.models.load_model(h5_path, compile=False)
            out_path = h5_path.replace(".h5", ".keras")
            tf.keras.saving.save_model(m, out_path)  # usa formato Keras nativo
            self.logger.info(f"Modelo convertido: {h5_path} -> {out_path}")
            return out_path
        except Exception:
            self.logger.error("Falha ao converter h5 para keras", exc_info=True)
            return None


# ======================================================
# Execução de teste rápido (quando executado diretamente)
# ======================================================
if __name__ == "__main__":
    print("Módulo predictor.py - teste rápido")
    try:
        p = Predictor(debug=True)
        print("Modelos carregados:", len(p.multi_models), "modelo único:", bool(p.model))
    except Exception as e:
        print("Erro inicializando Predictor:", e)
        traceback.print_exc()
