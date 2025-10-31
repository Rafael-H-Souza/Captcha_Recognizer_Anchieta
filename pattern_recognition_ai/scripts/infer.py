import argparse
import os

from src.inference.predictor import Predictor
from src.config.settings import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência com o modelo de CAPTCHA.")
    parser.add_argument("--image_path", required=True, help="Caminho para a imagem a ser decodificada.")
    parser.add_argument("--model_path", help="Caminho opcional para o modelo salvo.")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Arquivo não encontrado em '{args.image_path}'")

    predictor = Predictor(args.model_path or config["final_model_path"])
    result = predictor.predict(args.image_path)
    print(f"Texto decodificado: {result}")


if __name__ == "__main__":
    main()
