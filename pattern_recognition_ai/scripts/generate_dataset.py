import os
import random
import string
from pathlib import Path
from captcha.image import ImageCaptcha
from tqdm import tqdm
import shutil

# Configurações
NUM_IMAGES = 10
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60

CHARS_NUM = string.digits
CHARS_WORLD = string.ascii_uppercase
CHARS_ALPHANUM = string.ascii_uppercase + string.digits
CHARS_ALPHANUM_CASE = string.ascii_letters + string.digits

BASE_DIR = Path("data/raw")
TYPES = ["raw", "single_char", "five_char"]
SUBTYPES = ["numb", "world", "alphanumeric", "alphanumeric_case"]

# Funções
def clear_folder(folder_path):
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

def save_captcha_images(folder_path, chars, length=1, num_images=NUM_IMAGES):
    used_labels = set()
    max_possible = len(chars) ** length
    num_images = min(num_images, max_possible)
    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    for _ in tqdm(range(num_images), desc=f"Gerando {folder_path}"):
        while True:
            text = "".join(random.choices(chars, k=length))
            if text not in used_labels:
                used_labels.add(text)
                break
        image = generator.generate_image(text)
        image_path = folder_path / f"{text}.png"
        image.save(image_path)
    return num_images

# Gerar dataset
for t in TYPES:
    for st in SUBTYPES:
        folder = BASE_DIR / t / st
        clear_folder(folder)

        length = 1 if t in ["raw", "single_char"] else 5

        if st == "numb":
            chars = CHARS_NUM
        elif st == "world":
            chars = CHARS_WORLD
        elif st == "alphanumeric":
            chars = CHARS_ALPHANUM
        elif st == "alphanumeric_case":
            chars = CHARS_ALPHANUM_CASE

        generated = save_captcha_images(folder, chars, length=length, num_images=NUM_IMAGES)
        print(f"✅ {generated} imagens geradas em {folder}")

print("🗂️ Dataset completo gerado com CAPTCHAs distorcidos!")
