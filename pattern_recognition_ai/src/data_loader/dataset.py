import os
from typing import Dict, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class CaptchaDataLoader:
    """Carrega imagens e rótulos da estrutura de pastas, garantindo compatibilidade com max_length."""

    TYPES = ["single_char", "five_char"]
    SUBTYPES = ["numb", "world", "alphanumeric"]

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_dir = config["dataset_dir"]
        self.charset = config["charset"]
        self.char_to_num = {char: idx for idx, char in enumerate(self.charset)}
        self._create_folders()

    def _create_folders(self):
        for t in self.TYPES:
            for st in self.SUBTYPES:
                path = os.path.join(self.dataset_dir, t, st)
                os.makedirs(path, exist_ok=True)
        print("✅ Pastas do dataset verificadas/criadas.")

    def _encode_label(self, label: str) -> np.ndarray:
        """Transforma string em one-hot de shape (max_length, num_classes)."""
        encoded = np.zeros((self.config["max_length"], self.config["num_classes"]), dtype=np.float32)
        for i, char in enumerate(label):
            if i >= self.config["max_length"]:
                break
            if char in self.char_to_num:
                encoded[i, self.char_to_num[char]] = 1.0
        return encoded

    def _process_images_and_labels(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        images, labels = [], []

        for file in os.listdir(folder_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (self.config["image_width"], self.config["image_height"]))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)

            label = os.path.splitext(file)[0]
            labels.append(self._encode_label(label))

        images = np.array(images)
        labels = np.array(labels)

        # Garantir shape 3D
        if labels.ndim == 2:  # caso só tenha 1 caractere
            labels = labels[:, :, np.newaxis]
        elif labels.ndim == 1:  # caso tenha apenas 1 imagem
            labels = labels[np.newaxis, :, np.newaxis]

        return images, labels

    def load_data_from_path(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Carrega imagens e labels de uma pasta específica, já dividindo em treino e validação."""

        images, labels = self._process_images_and_labels(folder_path)

        if len(images) == 0:
            return np.array([]), np.array([]), {}, {}

        # Separa cada caractere para classificação multi-saída
        label_splits = [labels[:, i, :] for i in range(self.config["max_length"])]

        split = train_test_split(
            images,
            *label_splits,
            test_size=self.config["validation_split"],
            random_state=42,
            shuffle=True
        )

        x_train = split[0]
        x_val = split[1]
        y_train_splits = split[2::2]
        y_val_splits = split[3::2]

        y_train_dict = {f"char_{i + 1}": np.array(y_train_splits[i]) for i in range(self.config["max_length"])}
        y_val_dict = {f"char_{i + 1}": np.array(y_val_splits[i]) for i in range(self.config["max_length"])}

        return x_train, x_val, y_train_dict, y_val_dict

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Carrega todo o dataset."""
        return self.load_data_from_path(self.dataset_dir)
    
    def get_dataset(self):
        """Retorna o dataset formatado para o treinamento Keras (x, y)"""
        x_train, x_val, y_train, y_val = self.load_data()
        if len(x_train) == 0:
            print("⚠️ Nenhuma imagem encontrada no dataset.")
            return None, None

        train_dataset = (x_train, y_train)
        val_dataset = (x_val, y_val)
        return train_dataset, val_dataset

    
    
