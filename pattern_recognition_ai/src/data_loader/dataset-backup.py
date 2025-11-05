import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CaptchaDataLoader:
    """Carrega imagens e rótulos para o pipeline de treinamento."""

    def __init__(self, config: Dict):
        self.config = config
        self.image_dir = config["processed_data_dir"]
        self.labels_path = config["labels_path"]
        self.charset = config["charset"]
        self.char_to_num = {char: idx for idx, char in enumerate(self.charset)}

    def _encode_label(self, label: str) -> np.ndarray:
        encoded = np.zeros((self.config["max_length"], self.config["num_classes"]), dtype=np.float32)
        for i, char in enumerate(label):
            if i >= self.config["max_length"]:
                break
            if char in self.char_to_num:
                encoded[i, self.char_to_num[char]] = 1.0
        return encoded

    def load_data(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        logger.info("Carregando dados e rótulos...")
        if not os.path.exists(self.labels_path):
            logger.error("Arquivo de rótulos não encontrado em: %s", self.labels_path)
            return None, None, None, None

        df = pd.read_csv(self.labels_path)
        images = []
        labels = []

        for _, row in df.iterrows():
            img_path = os.path.join(self.image_dir, row["filename"])
            if not os.path.exists(img_path):
                logger.warning("Imagem não encontrada: %s", img_path)
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Falha ao carregar imagem: %s", img_path)
                continue

            img = cv2.resize(img, (self.config["image_width"], self.config["image_height"]))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            labels.append(self._encode_label(str(row["label"])))

        if not images:
            logger.error("Nenhuma imagem válida foi carregada.")
            return None, None, None, None

        images = np.asarray(images)
        labels = np.asarray(labels)

        label_splits = [labels[:, i, :] for i in range(self.config["max_length"])]

        split = train_test_split(
            images,
            *label_splits,
            test_size=self.config["validation_split"],
            random_state=42,
            shuffle=True,
        )

        x_train = split[0]
        x_val = split[1]

        y_train_splits = split[2::2]
        y_val_splits = split[3::2]

        y_train_dict = {f"char_{i + 1}": np.asarray(y_train_splits[i]) for i in range(self.config["max_length"])}
        y_val_dict = {f"char_{i + 1}": np.asarray(y_val_splits[i]) for i in range(self.config["max_length"])}

        logger.info("Dados carregados: %d para treino, %d para validação.", len(x_train), len(x_val))
        return x_train, x_val, y_train_dict, y_val_dict

    def get_dataset(self) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
        x_train, x_val, y_train, y_val = self.load_data()
        if x_train is None:
            return None, None

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(self.config["batch_size"]).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(self.config["batch_size"]).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset
