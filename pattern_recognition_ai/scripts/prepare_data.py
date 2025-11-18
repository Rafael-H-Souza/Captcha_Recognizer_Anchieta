#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 Pré-processamento de imagens CAPTCHA
"""

import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

from src.config import settings
from src.utils.logger import get_logger

def preprocess_image(image_path: str):
    IMG_HEIGHT = 50
    IMG_WIDTH = 200

    # 1) Ler em escala de cinza
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # 2) Ajustar tamanho fixo
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

    # 3) Suavização leve (NÃO destrói bordas)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 4) Binarização segura (NÃO usa threshold adaptativo)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # 5) Remover pequenos pontos de ruído (muito seguro)
    img = cv2.medianBlur(img, 3)

    # 6) Erros pequenos são removidos, texto intacto
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img


def cleanup_processed_dataset(type_name: str | None = None, subtype: str | None = None) -> None:
    processed_dir = Path(settings.config["processed_data_dir"])
    if not processed_dir.exists():
        print("⚠️ Diretório 'data/processed' não encontrado — nada a limpar.\n")
        return

    if type_name and subtype:
        target = processed_dir / type_name / subtype
        if target.exists():
            print(f"🧹 Limpando subset: {target}")
            shutil.rmtree(target)
            print("✅ Subset removido com sucesso.")
        else:
            print(f"⚠️ Subset {target} não encontrado.")
    else:
        print("🧹 Limpando completamente a pasta 'data/processed'...")
        shutil.rmtree(processed_dir, ignore_errors=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Pasta 'data/processed' limpa com sucesso.\n")

def save_npy(files, folder: Path, x_file: Path, y_file: Path, max_length: int, valid_chars: str):
    X, Y = [], []

    for f in files:
        img_path = folder / f
        try:
            img = preprocess_image(str(img_path))
            img = img.astype(np.float32) / 255.0
            X.append(img)

            label_str = Path(f).stem
            label_str = "".join([c for c in label_str if c in valid_chars])
            if not label_str:
                continue

            label = [ord(c) for c in label_str[:max_length]]
            label += [0] * (max_length - len(label))
            Y.append(label)
        except Exception as e:
            print(f"❌ Erro processando {f}: {e}")

    if not X:
        print(f"⚠️ Nenhuma imagem válida encontrada em {folder}")
        return False

    np.save(x_file, np.array(X))
    np.save(y_file, np.array(Y))
    print(f"💾 Arquivos salvos: {x_file.name}, {y_file.name}")
    return True

def run_preprocessing(type_name: str | None = None, subtype: str | None = None) -> None:
    logger = get_logger("Data-Preparation")

    raw_dir = Path(settings.config["raw_data_dir_base"])
    processed_dir = Path(settings.config["processed_data_dir"])
    max_length = settings.config.get("max_length", 5)
    valid_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    print("\n🚀 INICIANDO PRÉ-PROCESSAMENTO DE IMAGENS CAPTCHA")
    print("=" * 70)

    cleanup_processed_dataset(type_name, subtype)

    total_images = 0
    summary = []

    if not raw_dir.exists():
        print("❌ Diretório data/raw não encontrado.")
        return

    type_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"📂 Encontrados {len(type_dirs)} tipos para processamento.\n")

    for subdir, _, files in os.walk(raw_dir):
        files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue

        relative = Path(subdir).relative_to(raw_dir)
        if type_name and relative.parts[0] != type_name:
            continue
        if subtype and len(relative.parts) > 1 and relative.parts[1] != subtype:
            continue

        dst_folder = processed_dir / relative
        dst_folder.mkdir(parents=True, exist_ok=True)

        print(f"📸 Processando {len(files)} imagens em {relative}...")
        logger.info(f"Processando {len(files)} imagens de {relative}...")

        for filename in tqdm(files, desc=str(relative), unit="img", ncols=90):
            src_path = Path(subdir) / filename
            dst_path = dst_folder / filename
            try:
                processed = preprocess_image(str(src_path))
                cv2.imwrite(str(dst_path), processed)
                total_images += 1
            except Exception as exc:
                logger.error(f"Falha ao processar {filename}: {exc}")

        split_idx = int(len(files) * 0.8)
        train_files, val_files = files[:split_idx], files[split_idx:]

        x_train, y_train = dst_folder / "x_train.npy", dst_folder / "y_train.npy"
        x_val, y_val = dst_folder / "x_val.npy", dst_folder / "y_val.npy"

        save_npy(train_files, dst_folder, x_train, y_train, max_length, valid_chars)
        save_npy(val_files, dst_folder, x_val, y_val, max_length, valid_chars)

        summary.append((str(relative), len(files), len(train_files), len(val_files)))
        print(f"✅ {relative}: concluído.\n")

    print("=" * 70)
    print("🎉 PRÉ-PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"📊 Total de imagens processadas: {total_images}")
    print(f"📁 Datasets salvos em: {processed_dir.resolve()}\n")
    print("📋 RESUMO POR CATEGORIA:")
    print("-" * 70)
    for rel, total, train, val in summary:
        print(f"📦 {rel:<30} | Total: {total:<5} | Treino: {train:<5} | Validação: {val}")
    print("-" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processa imagens CAPTCHA.")
    parser.add_argument("--type", type=str, default=None, help="Tipo (ex: five_char, single_char)")
    parser.add_argument("--subtype", type=str, default=None, help="Subtipo (ex: world, numb)")
    args = parser.parse_args()

    run_preprocessing(args.type, args.subtype)