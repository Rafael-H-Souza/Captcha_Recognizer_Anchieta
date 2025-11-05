import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.preprocessing.filters import preprocess_image
from src.config import settings
from src.utils.logger import get_logger


# ======================================================
# Função para salvar imagens em .npy
# ======================================================
def save_npy(files, folder_path: Path, x_file: Path, y_file: Path, max_length: int):
    """
    Converte imagens de uma lista de arquivos em arrays e gera x.npy e y.npy.
    Cada imagem deve ter o label no nome (ex: "a_01.png" ou "12345.png").
    """
    x_data, y_data = [], []

    for f in files:
        img_path = folder_path / f
        try:
            img = preprocess_image(str(img_path))
            x_data.append(img.astype(np.float32) / 255.0)  # normaliza 0-1

            # Extrai label do nome do arquivo
            label_str = Path(f).stem
            label = [ord(c) for c in label_str]
            label += [0] * (max_length - len(label))
            y_data.append(label[:max_length])

        except Exception as e:
            print(f"[ERRO] ❌ Erro processando {f}: {e}")

    if not x_data:
        return False

    np.save(x_file, np.array(x_data))
    np.save(y_file, np.array(y_data))
    return True


# ======================================================
# Função principal de pré-processamento
# ======================================================
def run_preprocessing() -> None:
    logger = get_logger("Data-Preparation")

    raw_dir = Path(settings.config["raw_data_dir_base"])
    processed_dir = Path(settings.config["processed_data_dir"])
    max_length = settings.config.get("max_length", 5)

    # Limpa pasta processed se existir
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    # Percorre todos os subdiretórios recursivamente
    for subdir, _, files in os.walk(raw_dir):
        files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue

        relative = Path(subdir).relative_to(raw_dir)
        dst_folder = processed_dir / relative
        dst_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"[INFO] 🔄 Processando {len(files)} imagens de {relative}...")

        # Processa imagens com tqdm
        for filename in tqdm(files, desc=str(relative), unit="img", ncols=100, ascii=True):
            src_path = Path(subdir) / filename
            dst_path = dst_folder / filename
            try:
                processed_img = preprocess_image(str(src_path))
                cv2.imwrite(str(dst_path), processed_img)
            except Exception as exc:
                logger.error(f"[ERRO] ❌ Falha ao processar {filename}: {exc}")

        # Divide treino e validação
        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # Cria datasets .npy
        x_train_file = dst_folder / "x_train.npy"
        y_train_file = dst_folder / "y_train.npy"
        x_val_file = dst_folder / "x_val.npy"
        y_val_file = dst_folder / "y_val.npy"

        save_npy(train_files, dst_folder, x_train_file, y_train_file, max_length)
        save_npy(val_files, dst_folder, x_val_file, y_val_file, max_length)

        logger.info(f"[SUCESSO] ✅ {relative} processado.")
        logger.info(f"   [TREINO] 🟢 {x_train_file.name}, {y_train_file.name}")
        logger.info(f"   [VAL] 🔵 {x_val_file.name}, {y_val_file.name}")

        summary.append((relative, len(files), len(train_files), len(val_files)))

    # ======================================================
    # Resumo final
    # ======================================================
    logger.info("[RESUMO] 🎉 Pré-processamento completo!")
    logger.info(f"[RESUMO] 📂 Arquivos gerados em: {processed_dir}")
    logger.info("[RESUMO] 📊 Resumo por categoria:")
    for rel, total, train, val in summary:
        logger.info(f"   - {rel}: total={total}, treino={train}, validação={val}")


if __name__ == "__main__":
    run_preprocessing()
