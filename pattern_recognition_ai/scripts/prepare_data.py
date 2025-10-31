import os
import shutil
import cv2
from tqdm import tqdm

from src.preprocessing.filters import preprocess_image
from src.config import settings
from src.utils.logger import get_logger

def run_preprocessing() -> None:
    logger = get_logger("DataPreparation")

    raw_dir = settings.config["raw_data_dir"]
    processed_dir = settings.config["processed_data_dir"]

    # Apaga pasta processed se existir
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    # Cria a estrutura completa de pastas no processed
    types = ["raw", "single_char", "five_char"]
    subtypes = ["numb", "world", "alphanumeric", "alphanumeric_case"]
    for t in types:
        for st in subtypes:
            os.makedirs(os.path.join(processed_dir, t, st), exist_ok=True)

    logger.info("✅ Estrutura de pastas criada em %s", processed_dir)

    # Processamento das imagens
    for t in types:
        for st in subtypes:
            src_folder = os.path.join(raw_dir, t, st)
            dst_folder = os.path.join(processed_dir, t, st)

            if not os.path.exists(src_folder):
                logger.warning("Pasta não encontrada: %s. Pulando.", src_folder)
                continue

            files = [f for f in os.listdir(src_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            total_files = len(files)
            if total_files == 0:
                logger.warning("Nenhuma imagem encontrada em %s. Pulando.", src_folder)
                continue

            success_count = 0
            fail_count = 0

            logger.info("Processando %d imagens de %s...", total_files, src_folder)

            for filename in tqdm(files, desc=f"Processando {t}/{st}", unit="img"):
                raw_path = os.path.join(src_folder, filename)
                processed_path = os.path.join(dst_folder, filename)
                try:
                    processed_img = preprocess_image(raw_path)
                    cv2.imwrite(processed_path, processed_img)
                    success_count += 1
                except FileNotFoundError:
                    logger.warning("Arquivo não encontrado: %s. Pulando.", raw_path)
                    fail_count += 1
                except Exception as exc:
                    logger.error("Falha ao processar %s: %s", filename, exc)
                    fail_count += 1

            logger.info(
                "✅ Pasta %s/%s concluída. Sucesso: %d, Falhas: %d, Total: %d",
                t, st, success_count, fail_count, total_files
            )

    logger.info("✅ Pré-processamento completo. Imagens salvas em %s.", processed_dir)


if __name__ == "__main__":
    run_preprocessing()
