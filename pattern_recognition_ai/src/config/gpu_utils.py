import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_gpu(use_gpu: bool = True) -> None:
    """Configura o TensorFlow para utilizar GPU quando disponível."""
    if not use_gpu:
        logger.info("GPU desabilitada por parâmetro. Forçando uso de CPU.")
        tf.config.set_visible_devices([], "GPU")
        return

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        logger.warning("⚠️ Nenhuma GPU compatível com CUDA encontrada. Usando CPU.")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], "GPU")
        logger.info("✅ GPU configurada com sucesso: %s dispositivo(s) detectado(s).", len(gpus))
    except RuntimeError as exc:
        logger.error("Erro ao configurar a GPU: %s", exc)
