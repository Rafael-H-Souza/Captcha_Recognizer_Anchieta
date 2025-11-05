import os
import shutil
import cv2
from tqdm import tqdm
from dotenv import load_dotenv

from src.preprocessing.filters import preprocess_image
from src.utils.logger import get_logger

# ======================================================
# Carrega variáveis de ambiente
# ======================================================
load_dotenv()

# ======================================================
# Diretório base do projeto (raiz)
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_TRAIN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================================================
# Diretórios principais
# ======================================================
DATA_DIR = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models"))
#LOGS_DIR = os.getenv("LOGS_PATH", os.path.join(BASE_DIR, "logs"))
LOGS_DIR = os.path.join(BASE_DATA_TRAIN, "logs")


# ======================================================
# Subpastas de dados
# ======================================================
RAW_DATA_DIR = os.path.join(BASE_DATA_TRAIN, "data")
RAW_DATA_DIR_BASE = os.path.join(BASE_DATA_TRAIN, "data/raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_TRAIN, "data/processed")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# ======================================================
# Subpastas de modelos
# ======================================================
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
EXPORTED_MODEL_DIR = os.path.join(MODEL_DIR, "exported")
MODELS_BY_TYPE_DIR = os.path.join(MODEL_DIR, "apurados")

# ======================================================
# Caminho principal de modelo (opcional)
# ======================================================
FINAL_MODEL_PATH = None  # Deixa None para usar ensemble automático

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
    "base_dir": BASE_DIR,
    "data_dir": DATA_DIR,
    "model_dir": MODEL_DIR,
    "logs_dir": LOGS_DIR,

    # Dados
    "raw_data_dir": RAW_DATA_DIR,
    "raw_data_dir_base": RAW_DATA_DIR_BASE,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "dataset_dir": PROCESSED_DATA_DIR,
    "labels_path": os.path.join(LABELS_DIR, "labels.csv"),

    # Modelos
    "checkpoint_dir": CHECKPOINT_DIR,
    "exported_model_dir": EXPORTED_MODEL_DIR,
    "models_by_type_dir": MODELS_BY_TYPE_DIR,
    "final_model_path": FINAL_MODEL_PATH,

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