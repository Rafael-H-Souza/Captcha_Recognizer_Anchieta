import argparse
import csv
import os
import random
from typing import Optional

from captcha.image import ImageCaptcha
from tqdm import tqdm

from src.config import settings

DEFAULT_NUM_IMAGES = 1000
DEFAULT_CHARSET = settings.config["charset"]
DEFAULT_CAPTCHA_LENGTH = settings.config["max_length"]
RAW_DATA_DIR = settings.config["raw_data_dir"]
LABELS_FILE = settings.config["labels_path"]


def generate_captcha_dataset(
    num_images: int = DEFAULT_NUM_IMAGES,
    charset: Optional[str] = None,
    captcha_length: Optional[int] = None,
) -> None:
    """Gera imagens de CAPTCHA e salva os rótulos correspondentes em CSV."""
    charset = charset or DEFAULT_CHARSET
    captcha_length = captcha_length or DEFAULT_CAPTCHA_LENGTH

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)

    generator = ImageCaptcha(
        width=settings.config["image_width"],
        height=settings.config["image_height"],
    )

    print(f"Gerando {num_images} imagens de CAPTCHA...")

    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])

        for idx in tqdm(range(num_images)):
            captcha_text = "".join(random.choices(charset, k=captcha_length))
            filename = f"captcha_{idx + 1}.png"
            file_path = os.path.join(RAW_DATA_DIR, filename)

            generator.write(captcha_text, file_path)
            writer.writerow([filename, captcha_text])

    print("\n✅ Dataset gerado com sucesso!")
    print(f"Imagens salvas em: {RAW_DATA_DIR}")
    print(f"Rótulos salvos em: {LABELS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera um dataset sintético de CAPTCHAs e registra os rótulos em CSV.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Quantidade de imagens a serem geradas.",
    )
    parser.add_argument(
        "--charset",
        type=str,
        default=None,
        help="Sequência de caracteres permitidos no CAPTCHA (default: configuração do projeto).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Quantidade de caracteres por CAPTCHA (default: configuração do projeto).",
    )

    args = parser.parse_args()

    generate_captcha_dataset(
        num_images=args.num_images,
        charset=args.charset,
        captcha_length=args.length,
    )
