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

    def _setup_environment(self):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        os.makedirs(self.config["exported_model_dir"], exist_ok=True)
        os.makedirs(self.config["models_by_type_dir"], exist_ok=True)

    def run(self) -> Optional[tf.keras.callbacks.History]:
        self._setup_environment()

        # Carrega dataset
        data_loader = CaptchaDataLoader(self.config)
        train_dataset, val_dataset = data_loader.get_dataset()
        if train_dataset is None:
            self.logger.error("Dataset vazio ou inválido. Abortando treinamento.")
            return None

        # Cria modelo
        self.model = build_cnn_gru_model(self.config)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config["learning_rate"]),
            loss={f"char_{i+1}": "categorical_crossentropy" for i in range(self.config["max_length"])},
            metrics={f"char_{i+1}": "accuracy" for i in range(self.config["max_length"])},
        )

        # Callbacks
        model_save_path = self.config.get("model_save_path", os.path.join(self.config["models_by_type_dir"], "model.h5"))
        checkpoint_cb = ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=[checkpoint_cb]
        )

        # Salva backup em exported
        backup_path = os.path.join(self.config["exported_model_dir"], Path(model_save_path).name + "-backup.h5")
        self.model.save(backup_path)
        self.logger.info("Backup do modelo salvo em: %s", backup_path)

        return history
