import cv2
import numpy as np

IMG_HEIGHT = 50
IMG_WIDTH = 200

def preprocess_image(image_path: str):
    """Pré-processamento ideal e seguro para CAPTCHAs."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # 1) Redimensiona para o tamanho padrão
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

    # 2) Remove ruído leve sem destruir traços
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 3) Binarização adaptativa – mais robusta que threshold fixo
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        8
    )

    return img
