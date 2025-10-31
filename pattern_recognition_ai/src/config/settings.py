import os
from dotenv import load_dotenv

load_dotenv()

# ======================================================
# Diretório base do projeto
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================================================
# Diretórios principais
# ======================================================
DATA_DIR = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models"))
LOGS_DIR = os.getenv("LOGS_PATH", os.path.join(BASE_DIR, "logs"))

# ======================================================
# Subpastas de dados
# ======================================================
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# ======================================================
# Subpastas de modelos
# ======================================================
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")          # Para checkpoints temporários
EXPORTED_MODEL_DIR = os.path.join(MODEL_DIR, "exported")         # Para modelos exportados
MODELS_BY_TYPE_DIR = os.path.join(MODEL_DIR, "apurados")         # Para modelos por tipo/subtipo

FINAL_MODEL_PATH = os.path.join(EXPORTED_MODEL_DIR, "captcha_model.h5")
ONNX_MODEL_PATH = os.path.join(EXPORTED_MODEL_DIR, "captcha_model.onnx")

# ======================================================
# Parâmetros de imagem
# ======================================================
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", 200))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", 50))
IMAGE_CHANNELS = 1

# ======================================================
# Configuração do dataset
# ======================================================
MAX_LENGTH = 5
CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"
NUM_CLASSES = len(CHARSET)

# ======================================================
# Hiperparâmetros de treino
# ======================================================
EPOCHS = int(os.getenv("EPOCHS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
VALIDATION_SPLIT = 0.2

# ======================================================
# Configuração global
# ======================================================
config = {
    # Paths principais
    "base_dir": BASE_DIR,
    "data_dir": DATA_DIR,
    "model_dir": MODEL_DIR,
    "logs_dir": LOGS_DIR,

    # Dados
    "raw_data_dir": RAW_DATA_DIR,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "dataset_dir": PROCESSED_DATA_DIR,
    "labels_path": os.path.join(LABELS_DIR, "labels.csv"),

    # Modelos
    "checkpoint_dir": CHECKPOINT_DIR,
    "exported_model_dir": EXPORTED_MODEL_DIR,
    "models_by_type_dir": MODELS_BY_TYPE_DIR,
    "final_model_path": FINAL_MODEL_PATH,
    "onnx_model_path": ONNX_MODEL_PATH,

    # Imagem
    "image_width": IMAGE_WIDTH,
    "image_height": IMAGE_HEIGHT,
    "image_channels": IMAGE_CHANNELS,

    # Dataset
    "max_length": MAX_LENGTH,
    "num_classes": NUM_CLASSES,
    "charset": CHARSET,

    # Treino
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "validation_split": VALIDATION_SPLIT,
}
