import cv2


def binarize_image(image, threshold: int = 128):
    """Aplica binarização a uma imagem em escala de cinza."""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary_image


def remove_noise(image):
    """Remove ruído aplicando um filtro de mediana."""
    return cv2.medianBlur(image, 3)


def preprocess_image(image_path: str):
    """Pipeline completo de pré-processamento para uma única imagem."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada em {image_path}")

    img_no_noise = remove_noise(img)
    img_binary = binarize_image(img_no_noise, 170)
    return img_binary
