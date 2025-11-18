#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def listar_keras(pasta: str):
    pasta = Path(pasta)
    if not pasta.exists():
        return []
    return sorted([f for f in pasta.iterdir() if f.suffix == ".keras"])


def carregar_imagem_exemplo(args):
    """
    Se o usuário passar --image, tenta carregar.
    Senão, gera uma imagem vazia (50x200x1).
    """
    if args.image:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow (PIL) não está instalado, instale com: pip install pillow")

        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        img = Image.open(img_path).convert("L")  # escala de cinza
        img = img.resize((200, 50))             # largura=200, altura=50
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)      # (50,200,1)
        return arr
    else:
        # imagem vazia
        return np.zeros((50, 200, 1), dtype="float32")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizador/Testador de modelos Keras (.keras)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/exported",
        help="Diretório onde estão os modelos .keras (default: models/exported)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nome do modelo .keras (arquivo) dentro de models-dir. "
             "Se não informado, será exibida uma lista para escolha.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Caminho de uma imagem para teste. Se não informado, usa imagem vazia (50x200x1).",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Diretório de modelos não existe: {models_dir}")
        return

    arquivos = listar_keras(models_dir)
    if not arquivos:
        print(f"❌ Nenhum arquivo .keras encontrado em: {models_dir}")
        return

    # Se o usuário não informou --model, mostrar lista para escolha
    if args.model is None:
        print("\n📦 Modelos encontrados:")
        for i, f in enumerate(arquivos, start=1):
            print(f"{i}. {f.name}")

        escolha = input("\nDigite o número do modelo que deseja carregar: ").strip()
        try:
            idx = int(escolha)
            if idx < 1 or idx > len(arquivos):
                raise ValueError()
        except ValueError:
            print("❌ Opção inválida.")
            return

        model_path = arquivos[idx - 1]
    else:
        # Usuário passou --model: pode ser nome simples ou caminho completo
        candidate = Path(args.model)
        if candidate.is_file():
            model_path = candidate
        else:
            model_path = models_dir / args.model
        if not model_path.exists():
            print(f"❌ Modelo não encontrado: {model_path}")
            return

    print(f"\n🔍 Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("\n📑 Arquitetura do modelo:")
    model.summary()

    # Carregar imagem
    try:
        img = carregar_imagem_exemplo(args)
    except Exception as e:
        print(f"❌ Erro ao preparar imagem de teste: {e}")
        return

    print("\n🤖 Realizando predição de teste...")
    pred = model.predict(np.expand_dims(img, axis=0))
    print("\n📈 Saída da predição:")
    print(pred)
    print("\n✅ Teste concluído com sucesso.")


if __name__ == "__main__":
    main()
