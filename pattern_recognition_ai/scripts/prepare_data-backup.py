import os

import cv2
import pandas as pd
from tqdm import tqdm

from src.preprocessing.filters import preprocess_image
from src.config import settings
from src.utils.logger import get_logger


def run_preprocessing() -> None:
    logger = get_logger("DataPreparation")

    labels_path = settings.config["labels_path"]
    raw_dir = settings.config["raw_data_dir"]
    processed_dir = settings.config["processed_data_dir"]

    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(labels_path):
        logger.error("Arquivo de rótulos não encontrado em %s. Crie um CSV com colunas 'filename' e 'label'.", labels_path)
        return

    df = pd.read_csv(labels_path)
    logger.info("Processando %d imagens de %s...", len(df), raw_dir)

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row["filename"]
        raw_path = os.path.join(raw_dir, filename)
        processed_path = os.path.join(processed_dir, filename)

        try:
            processed_img = preprocess_image(raw_path)
            cv2.imwrite(processed_path, processed_img)
        except FileNotFoundError:
            logger.warning("Arquivo não encontrado: %s. Pulando.", raw_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Falha ao processar %s: %s", filename, exc)

    logger.info("✅ Pré-processamento concluído. Imagens salvas em %s.", processed_dir)


if __name__ == "__main__":
    run_preprocessing()
