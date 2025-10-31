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
    batch_size = settings.config.get("batch_size", 32)

    evaluation_results = []

    # Avaliação por tipo/subtipo
    for t in data_loader.TYPES:
        for st in data_loader.SUBTYPES:
            dataset_dir = os.path.join(settings.config["dataset_dir"], t, st)
            if not os.path.exists(dataset_dir):
                continue

            # Carrega dados
            x_train, x_val, y_train_dict, y_val_dict = data_loader.load_data_from_path(dataset_dir)
            if len(x_val) == 0:
                logger.warning("Nenhuma imagem encontrada em %s. Pulando...", dataset_dir)
                continue

            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val_dict))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            val_count = len(x_val)

            # Avalia
            results = model.evaluate(val_dataset, verbose=0)
            # Supondo que results[0] = loss, results[1:] = métricas (acurácias por output)
            num_outputs = settings.config["max_length"]
            total_loss = results[0]
            if len(results) > 1:
                accuracies = results[1:1 + num_outputs]
                average_accuracy = sum(accuracies) / len(accuracies)
            else:
                average_accuracy = 0.0

            logger.info(
                "%s_%s: loss=%.4f | acurácia média=%.2f%% | imagens=%d",
                t, st, total_loss, average_accuracy * 100, val_count
            )

            evaluation_results.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": t,
                "subtype": st,
                "model_path": model_path,
                "loss": total_loss,
                "accuracy": average_accuracy,
                "image_count": val_count
            })

    # Salva CSV
    log_dir = os.path.join(settings.config["logs_dir"], "evaluation")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluation_log.csv")

    df_new = pd.DataFrame(evaluation_results)
    if os.path.exists(log_file):
        df_old = pd.read_csv(log_file)
        df_log = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_log = df_new

    df_log.to_csv(log_file, index=False)
    logger.info("✅ Avaliação concluída. Resultados salvos em %s", log_file)


if __name__ == "__main__":
    evaluate_model()
