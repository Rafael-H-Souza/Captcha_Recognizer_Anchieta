import tensorflow as tf
import numpy as np
import glob
import os
import logging
import re

# --- Configuração de logs ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("merge_models")

# --- Diretórios ---
apurados_dir = "models/apurados"
os.makedirs(apurados_dir, exist_ok=True)

# --- Função para mesclar pesos de modelos ---
def merge_models(model_paths):
    logger.info(f"Carregando {len(model_paths)} modelos para mesclagem...")
    models = [tf.keras.models.load_model(p) for p in model_paths]
    merged_model = tf.keras.models.clone_model(models[0])
    merged_weights = []
    for i in range(len(models[0].get_weights())):
        weights = np.mean([m.get_weights()[i] for m in models], axis=0)
        merged_weights.append(weights)
    merged_model.set_weights(merged_weights)
    return merged_model

# --- Função para agrupar modelos por tipo base (ignora timestamp e '_merged') ---
def group_models_by_type(model_paths):
    groups = {}
    for path in model_paths:
        filename = os.path.basename(path).replace(".keras", "")
        # Ignora '_merged' no final
        filename = re.sub(r"_merged$", "", filename)
        # Remove o timestamp final
        filename_clean = re.sub(r"_\d{14}$", "", filename)
        groups.setdefault(filename_clean, []).append(path)
    return groups

# --- Main ---
all_model_paths = glob.glob(os.path.join(apurados_dir, "*.keras"))
if not all_model_paths:
    logger.error("Nenhum modelo encontrado em 'models/apurados'.")
    exit(1)

# Agrupar por tipo base
model_groups = group_models_by_type(all_model_paths)

# Mesclar e salvar um modelo por tipo
for model_type, paths in model_groups.items():
    merged_model = merge_models(paths)
    save_path = os.path.join(apurados_dir, f"{model_type}_merged.keras")
    merged_model.save(save_path)
    logger.info(f"✅ Modelo unificado salvo: {save_path}")
