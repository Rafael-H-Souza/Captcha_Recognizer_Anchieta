import os
from typing import Optional
from pathlib import Path

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

    # =====================================================
    # 🔹 Prepara diretórios antes do treinamento
    # =====================================================
    def _setup_environment(self):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        os.makedirs(self.config["exported_model_dir"], exist_ok=True)
        os.makedirs(self.config["models_by_type_dir"], exist_ok=True)

    # =====================================================
    # 🔹 Cria e compila o modelo CNN + GRU
    # =====================================================
    def build_model(self) -> tf.keras.Model:
        """Cria e compila o modelo CNN + GRU"""
        self.model = build_cnn_gru_model(self.config)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config["learning_rate"]),
            loss={f"char_{i+1}": "categorical_crossentropy" for i in range(self.config["max_length"])},
            metrics={f"char_{i+1}": "accuracy" for i in range(self.config["max_length"])},
        )
        self.logger.info("✅ Modelo CNN+GRU criado e compilado com sucesso.")
        return self.model

    # =====================================================
    # 🔹 Executa o fluxo completo de treinamento
    # =====================================================
    def run(self) -> Optional[tf.keras.callbacks.History]:
        if self.use_gpu:
            setup_gpu()

        self._setup_environment()

        # Carrega dataset
        data_loader = CaptchaDataLoader(self.config)
        train_dataset, val_dataset = data_loader.get_dataset()
        if train_dataset is None:
            self.logger.error("❌ Dataset vazio ou inválido. Abortando treinamento.")
            return None

        # Cria o modelo (mantendo compatibilidade com MultiTrainer)
        if self.model is None:
            self.build_model()

        # Callbacks
        model_save_path = self.config.get(
            "model_save_path",
            os.path.join(self.config["models_by_type_dir"], "model.h5")
        )

        checkpoint_cb = ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        tensorboard_cb = TensorBoard(
            log_dir=os.path.join(self.config["logs_dir"], "training_logs"),
            histogram_freq=1
        )

        # Carrega dados como arrays/dicionários
        x_train, x_val, y_train_dict, y_val_dict = data_loader.load_data_from_path(self.config["dataset_dir"])

        # Treinamento correto
        history = self.model.fit(
            x=x_train,                 # apenas as imagens
            y=y_train_dict,             # dicionário de saídas
            validation_data=(x_val, y_val_dict),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=[checkpoint_cb, early_stop_cb, tensorboard_cb]
        )

        # Backup final do modelo
        backup_path = os.path.join(
            self.config["exported_model_dir"],
            Path(model_save_path).stem + "-backup.h5"
        )
        self.model.save(backup_path)
        self.logger.info("✅ Backup do modelo salvo em: %s", backup_path)

        return history
