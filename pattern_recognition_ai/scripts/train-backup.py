import argparse
import sys
from src.model.trainer import Trainer
from src.config.settings import config
from src.utils.logger import get_logger


def parse_arguments() -> argparse.Namespace:
    """Faz o parsing dos argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treinamento do modelo de reconhecimento de CAPTCHA."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.get("epochs", 10),
        help="Número de épocas de treinamento."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.get("batch_size", 32),
        help="Tamanho do batch."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.get("learning_rate", 0.001),
        help="Taxa de aprendizado."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Força o uso de GPU (se disponível)."
    )

    return parser.parse_args()


def main() -> None:
    """Executa o treinamento do modelo de reconhecimento de CAPTCHA."""
    logger = get_logger("TrainingScript")

    try:
        args = parse_arguments()

        # Atualiza a configuração global com os parâmetros fornecidos
        config.update(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
            }
        )

        logger.info(
            "🚀 Iniciando treinamento com as configurações: "
            "Épocas=%d | Batch=%d | Learning Rate=%.5f | GPU=%s",
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.use_gpu,
        )

        trainer = Trainer(config, use_gpu=args.use_gpu)
        trainer.run()

        logger.info("✅ Treinamento concluído com sucesso!")

    except KeyboardInterrupt:
        logger.warning("❌ Execução interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        logger.exception("💥 Erro durante o treinamento: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
