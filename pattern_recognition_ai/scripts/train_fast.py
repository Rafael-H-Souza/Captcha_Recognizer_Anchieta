import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from glob import glob

# ==========================================================
# ⚙️ CONFIGURAÇÕES
# ==========================================================
dataset_path = "data/dataset/five_char_alphanumeric_case.npz"
raw_path = "data/raw/five_char/alphanumeric_case/"
os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

charset = "abcdefghijklmnopqrstuvwxyz0123456789"
char_to_index = {c: i for i, c in enumerate(charset)}

# ==========================================================
# 📦 FUNÇÃO PARA GERAR O DATASET .NPZ AUTOMATICAMENTE
# ==========================================================
def generate_npz_from_raw():
    print("🧩 Gerando dataset .npz a partir de imagens em:", raw_path)

    image_paths = glob(os.path.join(raw_path, "*.png"))
    if not image_paths:
        raise FileNotFoundError("❌ Nenhuma imagem encontrada em " + raw_path)

    X, y = [], []
    for img_path in tqdm(image_paths, desc="Processando imagens"):
        label = os.path.splitext(os.path.basename(img_path))[0].lower()  # nome do arquivo = rótulo
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 50))  # ajusta tamanho padrão
        img = img / 255.0  # normaliza
        X.append(img)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    np.savez_compressed(dataset_path, X=X, y=y)
    print(f"✅ Dataset salvo em {dataset_path} com {len(X)} amostras.")

# ==========================================================
# 🧠 VERIFICA SE EXISTE O NPZ — SE NÃO, GERA AUTOMATICAMENTE
# ==========================================================
if not os.path.exists(dataset_path):
    generate_npz_from_raw()

# ==========================================================
# 🔄 CARREGA O DATASET
# ==========================================================
data = np.load(dataset_path, allow_pickle=True)
X, y = data["X"], data["y"]
X = np.expand_dims(X, -1)  # adiciona canal para CNN

# ==========================================================
# 🎯 CONVERTE RÓTULOS PARA INDICES CORRETOS
# ==========================================================
y_simple = np.array([char_to_index[label[0].lower()] for label in y])
y_simple = tf.keras.utils.to_categorical(y_simple, num_classes=len(charset))

X_train, X_test, y_train, y_test = train_test_split(X, y_simple, test_size=0.2)

# ==========================================================
# 🧩 MODELO SIMPLES E RÁPIDO
# ==========================================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(50, 200, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(charset), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==========================================================
# 🚀 TREINAMENTO RÁPIDO
# ==========================================================
print("🚀 Iniciando treinamento rápido...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ==========================================================
# 💾 SALVAR MODELO
# ==========================================================
output_path = "models_apurados/five_char_alphanumeric_case_model.h5"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
model.save(output_path)
print(f"✅ Modelo salvo em {output_path}")
