import os
from datetime import datetime

import pandas as pd
import tensorflow as tf

from src.config import settings
from src.data_loader.dataset import CaptchaDataLoader
from src.utils.logger import get_logger


def evaluate_model() -> None:
    logger = get_logger("EvaluationScript")

    model_path = settings.config["final_model_path"]
    if not os.path.exists(model_path):
        logger.error("Modelo não encontrado em %s. Execute o treinamento primeiro.", model_path)
        return

    logger.info("Carregando modelo de %s...", model_path)
    model = tf.keras.models.load_model(model_path)

    data_loader = CaptchaDataLoader(settings.config)
    _, val_dataset = data_loader.get_dataset()
    if val_dataset is None:
        logger.error("Falha ao carregar dados de validação. Abortando avaliação.")
        return

    val_count = sum(1 for _ in val_dataset.unbatch())

    logger.info("Iniciando avaliação do modelo...")
    results = model.evaluate(val_dataset, verbose=0)

    num_outputs = settings.config["max_length"]
    accuracies = results[1 + num_outputs : 1 + (2 * num_outputs)]
    average_accuracy = sum(accuracies) / num_outputs
    total_loss = results[0]

    logger.info("Resultados da avaliação - loss: %.4f | acurácia média: %.2f%%", total_loss, average_accuracy * 100)

    log_file = os.path.join(settings.config["logs_dir"], "evaluation", "evaluation_log.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    new_entry = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_path": model_path,
                "loss": total_loss,
                "accuracy": average_accuracy,
                "image_count": val_count,
            }
        ]
    )

    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        df_log = pd.concat([df_log, new_entry], ignore_index=True)
    else:
        df_log = new_entry

    df_log.to_csv(log_file, index=False)
    logger.info("Resultados salvos em %s", log_file)


if __name__ == "__main__":
    evaluate_model()
