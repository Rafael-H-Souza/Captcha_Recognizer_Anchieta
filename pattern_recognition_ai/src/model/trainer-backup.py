import os
from typing import Optional

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from src.config.gpu_utils import setup_gpu
from src.model.architecture import build_cnn_gru_model
from src.data_loader.dataset import CaptchaDataLoader
from src.utils.logger import get_logger


class Trainer:
    def __init__(self, config: dict, use_gpu: bool = True):
        self.config = config
        self.use_gpu = use_gpu
        self.logger = get_logger(__name__)
        self.model: Optional[tf.keras.Model] = None

    def _setup_environment(self) -> None:
        if self.use_gpu:
            setup_gpu(use_gpu=True)
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        os.makedirs(os.path.join(self.config["logs_dir"], "training"), exist_ok=True)
        os.makedirs(self.config["exported_model_dir"], exist_ok=True)

    def run(self) -> Optional[tf.keras.callbacks.History]:
        self._setup_environment()

        data_loader = CaptchaDataLoader(self.config)
        train_dataset, val_dataset = data_loader.get_dataset()
        if train_dataset is None:
            self.logger.error("Falha ao carregar o dataset. Abortando treinamento.")
            return None

        self.model = build_cnn_gru_model(self.config)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"])
        losses = {f"char_{i + 1}": "categorical_crossentropy" for i in range(self.config["max_length"])}
        metrics = {f"char_{i + 1}": "accuracy" for i in range(self.config["max_length"])}

        self.model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        self.model.summary(print_fn=self.logger.info)

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.config["checkpoint_dir"], "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        tensorboard_callback = TensorBoard(log_dir=os.path.join(self.config["logs_dir"], "training"))

        self.logger.info("🚀 Iniciando treinamento...")
        history = self.model.fit(
            train_dataset,
            epochs=self.config["epochs"],
            validation_data=val_dataset,
            callbacks=[checkpoint, early_stopping, tensorboard_callback],
        )

        self.logger.info("✅ Treinamento concluído.")
        self.model.save(self.config["final_model_path"])
        self.logger.info("Modelo final salvo em: %s", self.config["final_model_path"])

        return history
