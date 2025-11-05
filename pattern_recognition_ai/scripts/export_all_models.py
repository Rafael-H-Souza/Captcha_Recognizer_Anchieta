import os
import shutil
from glob import glob
from src.utils.logger import get_logger

# ======================================================
# ⚙️ Configuração inicial
# ======================================================
logger = get_logger("Export-Models")

# Diretórios base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APURADOS_DIR = os.path.join(BASE_DIR, "models", "apurados")
EXPORT_DIR = os.path.join(BASE_DIR, "models", "exported")

# ======================================================
# 📂 Preparação dos diretórios
# ======================================================
os.makedirs(EXPORT_DIR, exist_ok=True)

logger.info(f"📁 Diretório de origem: {APURADOS_DIR}")
logger.info(f"📦 Diretório de destino: {EXPORT_DIR}")

# ======================================================
# 🔍 Busca e cópia dos modelos (.keras e .h5)
# ======================================================
def export_models():
    # Procura arquivos .keras e .h5
    keras_models = glob(os.path.join(APURADOS_DIR, "*.keras"))
    h5_models = glob(os.path.join(APURADOS_DIR, "*.h5"))
    models = keras_models + h5_models

    if not models:
        logger.error(f"❌ Nenhum modelo encontrado em: {APURADOS_DIR}")
        return

    total = 0
    for model_path in models:
        model_name = os.path.basename(model_path)
        dest_path = os.path.join(EXPORT_DIR, model_name)

        try:
            shutil.copy2(model_path, dest_path)
            logger.info(f"✅ Copiado: {model_name} → {EXPORT_DIR}")
            total += 1
        except Exception as e:
            logger.error(f"❌ Falha ao copiar {model_name}: {e}")

    logger.info(f"\n📦 {total} modelos exportados com sucesso para: {EXPORT_DIR}")


# ======================================================
# 🎯 Execução direta
# ======================================================
if __name__ == "__main__":
    logger.info("🚀 Iniciando exportação de modelos...")
    export_models()
    logger.info("✅ Processo de exportação concluído.")
