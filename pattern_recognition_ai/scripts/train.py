import argparse
from pathlib import Path

from src.model.trainer import Trainer
from src.config.settings import config
from src.utils.logger import get_logger

TYPES = ["raw", "single_char", "five_char"]
SUBTYPES = ["numb", "world", "alphanumeric", "alphanumeric_case"]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Treinamento de múltiplos modelos de CAPTCHA por tipo/subtipo."
    )
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 10))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 32))
    parser.add_argument("--learning-rate", type=float, default=config.get("learning_rate", 0.001))
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--processed-dir", type=str, default=config.get("processed_data_dir"))
    parser.add_argument("--models-dir", type=str, default="models/apurados")
    return parser.parse_args()

def main():
    logger = get_logger("MultiTrainer")
    args = parse_arguments()

    # Atualiza configuração global
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })

    processed_dir = Path(args.processed_dir)
    models_output_dir = Path(args.models_dir)
    models_output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("🚀 Iniciando treinamento apurado por tipo/subtipo...")

    for t in TYPES:
        for st in SUBTYPES:
            folder = processed_dir / t / st
            if not folder.exists() or not any(folder.iterdir()):
                logger.warning("Pasta vazia ou inexistente: %s. Pulando.", folder)
                continue

            # Nome do modelo baseado no tipo/subtipo
            model_name = f"{t}_{st}_model.h5"
            model_path = models_output_dir / model_name

            logger.info("Treinando modelo: %s com imagens de %s", model_name, folder)

            # Atualiza o config com dataset e caminho de saída
            config["dataset_dir"] = str(folder)
            config["model_save_path"] = str(model_path)

            try:
                trainer = Trainer(config=config, use_gpu=args.use_gpu)
                trainer.run()
                logger.info("✅ Modelo %s treinado com sucesso!", model_name)
            except Exception as e:
                logger.error("💥 Falha ao treinar modelo %s: %s", model_name, e)

    # Atualiza config global para avaliação, usando o dataset completo se necessário
    config.setdefault("dataset_dir", str(processed_dir))
    config.setdefault("model_save_path", str(models_output_dir / "captcha_model_final.h5"))

    logger.info("🎯 Treinamento apurado concluído para todos os tipos/subtipos.")

if __name__ == "__main__":
    main()
