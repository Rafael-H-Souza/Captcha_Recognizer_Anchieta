import argparse
import csv
import os
import random
from pathlib import Path

from captcha.image import ImageCaptcha
from tqdm import tqdm

from src.config import settings


DEFAULT_NUM_IMAGES = 100  # imagens por pasta
RAW_DATA_DIR = Path("data/raw")
LABELS_BASE_DIR = Path("data/labels")

# -------------------------------
#  Charset por subtipo
# -------------------------------
CHARSETS = {
    "numb": "0123456789",
    "world": "abcdefghijklmnopqrstuvwxyz",
    "alphanumeric": "abcdefghijklmnopqrstuvwxyz0123456789",
    "alphanumeric_case": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
}


def get_charset(subtype: str) -> str:
    """Retorna o charset de acordo com o subtipo."""
    return CHARSETS.get(subtype, CHARSETS["alphanumeric"])


def get_length(type_name: str) -> int:
    """Retorna o tamanho do CAPTCHA com base no tipo."""
    return 1 if "single_char" in type_name else 5


def generate_captcha_dataset(num_images: int = DEFAULT_NUM_IMAGES) -> None:
    """Gera dataset completo de acordo com a estrutura de pastas."""
    generator = ImageCaptcha(
        width=settings.config["image_width"],
        height=settings.config["image_height"],
    )

    TYPES = ["single_char", "five_char"]
    SUBTYPES = ["numb", "world", "alphanumeric", "alphanumeric_case"]

    print(f"\n🚀 Iniciando geração de {num_images} imagens para cada tipo/subtipo...\n")

    for t in TYPES:
        captcha_length = get_length(t)

        for st in SUBTYPES:
            # Diretório de saída: data/raw/<tipo>/<subtipo>
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

            print(f"✅ {t}/{st}: {num_images} imagens salvas em {output_dir}")
            print(f"   ↳ Rótulos: {labels_file}")

    print("\n🏁 Dataset completo gerado com sucesso!")
    print(f"📂 Estrutura base: {RAW_DATA_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera dataset sintético de CAPTCHAs conforme tipo/subtipo de pasta."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Quantidade de imagens por pasta (default: 100).",
    )

    args = parser.parse_args()
    generate_captcha_dataset(num_images=args.num_images)
