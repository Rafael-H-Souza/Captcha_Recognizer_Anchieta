#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Geração automatizada de dataset CAPTCHA + limpeza prévia.
Versão 2.0 — remove datasets antigos (imagens e rótulos) antes de cada nova geração.
"""

import argparse
import csv
import os
import random
from pathlib import Path
from tqdm import tqdm
from captcha.image import ImageCaptcha

from src.config import settings

# -----------------------------------------
# Configurações gerais
# -----------------------------------------
DEFAULT_NUM_IMAGES = 100  # imagens por pasta
RAW_DATA_DIR = Path("data/raw")
LABELS_BASE_DIR = Path("data/labels")

# -----------------------------------------
# Charset por subtipo
# -----------------------------------------
CHARSETS = {
    "numb": "0123456789",
    "world": "abcdefghijklmnopqrstuvwxyz",
    "alphanumeric": "abcdefghijklmnopqrstuvwxyz0123456789",
    "alphanumeric_case": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
}

# -----------------------------------------
# Função auxiliar: limpeza de datasets
# -----------------------------------------
def cleanup_dataset(type_name: str | None = None, subtype: str | None = None) -> None:
    """
    Apaga imagens (.png) e rótulos (.csv) antes de uma nova geração.
    Se 'type_name' e 'subtype' forem fornecidos, limpa apenas esse subconjunto.
    """
    total_deleted = 0

    # --- Limpar imagens
    if type_name and subtype:
        image_pattern = RAW_DATA_DIR / type_name / subtype / "*.png"
        image_files = list(Path(image_pattern.parent).glob("*.png"))
    else:
        image_files = list(RAW_DATA_DIR.rglob("*.png"))

    for img_path in tqdm(image_files, desc="🧹 Apagando imagens antigas", leave=False):
        try:
            img_path.unlink()
            total_deleted += 1
        except Exception as e:
            print(f"⚠️ Erro ao apagar {img_path}: {e}")

    # --- Limpar rótulos CSV
    if type_name and subtype:
        label_pattern = LABELS_BASE_DIR / f"{type_name}_{subtype}_labels.csv"
        label_files = [label_pattern] if label_pattern.exists() else []
    else:
        label_files = list(LABELS_BASE_DIR.glob("*.csv"))

    for csv_path in tqdm(label_files, desc="🧽 Apagando arquivos de rótulos", leave=False):
        try:
            csv_path.unlink()
            total_deleted += 1
        except Exception as e:
            print(f"⚠️ Erro ao apagar {csv_path}: {e}")

    print(f"🗑️  Total de arquivos removidos: {total_deleted}\n")


# -----------------------------------------
# Funções auxiliares para geração
# -----------------------------------------
def get_charset(subtype: str) -> str:
    """Retorna o charset de acordo com o subtipo."""
    return CHARSETS.get(subtype, CHARSETS["alphanumeric"])


def get_length(type_name: str) -> int:
    """Retorna o tamanho do CAPTCHA com base no tipo."""
    return 1 if "single_char" in type_name else 5


# -----------------------------------------
# Geração do dataset
# -----------------------------------------
def generate_captcha_dataset(num_images: int = DEFAULT_NUM_IMAGES) -> None:
    """Gera dataset completo e limpa versões anteriores antes de começar."""
    generator = ImageCaptcha(
        width=settings.config["image_width"],
        height=settings.config["image_height"],
    )

    TYPES = ["single_char", "five_char"]
    SUBTYPES = ["numb", "world", "alphanumeric", "alphanumeric_case"]

    print(f"\n🚀 Iniciando geração de {num_images} imagens para cada tipo/subtipo...\n")

    # --- Limpeza global antes de iniciar
    print("🧹 Limpando datasets antigos antes da nova geração...\n")
    cleanup_dataset()  # remove tudo

    for t in TYPES:
        captcha_length = get_length(t)

        for st in SUBTYPES:
            # Diretório de saída
            output_dir = RAW_DATA_DIR / t / st
            os.makedirs(output_dir, exist_ok=True)

            # Arquivo CSV de rótulos
            os.makedirs(LABELS_BASE_DIR, exist_ok=True)
            labels_file = LABELS_BASE_DIR / f"{t}_{st}_labels.csv"

            charset = get_charset(st)

            with open(labels_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["filename", "label"])

                for idx in tqdm(range(num_images), desc=f"{t}/{st}", leave=False):
                    captcha_text = "".join(random.choices(charset, k=captcha_length))
                    filename = f"{t}_{st}_{idx + 1}.png"
                    file_path = output_dir / filename

                    # Gera e salva imagem
                    generator.write(captcha_text, str(file_path))
                    writer.writerow([filename, captcha_text])

            print(f"✅ {t}/{st}: {num_images} imagens geradas em {output_dir}")
            print(f"   ↳ Rótulos: {labels_file}")

    print("\n🏁 Dataset completo gerado com sucesso!")
    print(f"📂 Estrutura base: {RAW_DATA_DIR.resolve()}")


# -----------------------------------------
# CLI (linha de comando)
# -----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera dataset sintético de CAPTCHAs conforme tipo/subtipo de pasta, limpando versões antigas antes."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Quantidade de imagens por pasta (default: 100).",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Limita a geração e limpeza a um tipo específico (ex: five_char ou single_char).",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default=None,
        help="Limita a geração e limpeza a um subtipo específico (ex: world, numb, alphanumeric_case).",
    )

    args = parser.parse_args()

    # Se o usuário quiser limpar apenas parte
    if args.type or args.subtype:
        print(f"🧹 Limpando dataset específico: {args.type or '*'} / {args.subtype or '*'}")
        cleanup_dataset(args.type, args.subtype)

    generate_captcha_dataset(num_images=args.num_images)
