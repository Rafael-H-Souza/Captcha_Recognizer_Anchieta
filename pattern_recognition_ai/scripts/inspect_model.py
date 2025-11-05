#!/usr/bin/env python3
"""
🧠 Inspeciona e testa um modelo .h5 do TensorFlow/Keras (multi-saída).
Mostra:
 - Estrutura completa do modelo
 - Total de parâmetros
 - Conjunto de caracteres reconhecidos
 - Predição de teste aleatória
 - Estatísticas por caractere se dataset fornecido
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf

# ============================
# 🎨 Cores para terminal
# ============================
class Color:
    OKGREEN = "\033[92m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

def color_text(text, color):
    return f"{color}{text}{Color.ENDC}"

# ============================
# ⚙️ Função principal
# ============================
def main():
    parser = argparse.ArgumentParser(description="Inspeciona e testa um modelo .h5")
    parser.add_argument("--model", "-m", required=True, help="Caminho do arquivo .h5")
    parser.add_argument("--dataset", "-d", help="Caminho do dataset (npz com X, y)")
    parser.add_argument("--no-predict", action="store_true", help="Desativa teste aleatório")
    args = parser.parse_args()

    # --------------------------
    # Carregar modelo
    # --------------------------
    if not os.path.exists(args.model):
        print(color_text(f"❌ Modelo não encontrado: {args.model}", Color.FAIL))
        sys.exit(1)

    print(color_text(f"🧩 Carregando modelo: {args.model}", Color.OKBLUE))
    model = tf.keras.models.load_model(args.model)
    model.summary()
    print(color_text(f"📊 Total de parâmetros: {model.count_params():,}", Color.WARNING))

    # --------------------------
    # Charset
    # --------------------------
    charset = []
    try:
        from src.config.settings import config
        charset = config.get("characters") or config.get("charset") or []
        if charset:
            print(color_text(f"🔤 Conjunto de caracteres ({len(charset)}): {' '.join(charset)}", Color.OKCYAN))
    except Exception:
        print(color_text("⚠ Charset não encontrado no config.", Color.WARNING))

    # --------------------------
    # Teste com dataset real
    # --------------------------
    if args.dataset:
        print(color_text("\n🧪 Testando com dataset real...", Color.OKGREEN))
        data = np.load(args.dataset)
        X = data["X"]
        y_true = data["y"]

        preds = model.predict(X)
        if isinstance(preds, list):
            preds = [np.argmax(p, axis=1) for p in preds]
            y_pred = np.stack(preds, axis=1)
        else:
            y_pred = np.argmax(preds, axis=1)

        total = y_true.shape[0]
        correct_total = np.sum(np.all(y_true == y_pred, axis=1))
        acc_total = (correct_total / total) * 100

        print(color_text(f"📈 Total de imagens: {total}", Color.OKBLUE))
        print(color_text(f"✅ Captchas corretos: {correct_total} ({acc_total:.2f}%)", Color.OKGREEN))
        print(color_text(f"❌ Captchas incorretos: {total - correct_total}", Color.FAIL))

        # Estatísticas por caractere
        if y_true.ndim == 2:
            print(color_text("\n📊 Acurácia por caractere:", Color.OKBLUE))
            for i in range(y_true.shape[1]):
                correct_char = np.sum(y_true[:, i] == y_pred[:, i])
                acc_char = (correct_char / total) * 100
                ch_label = f"char_{i+1}"
                print(f"→ {ch_label:<7}: {correct_char}/{total} ({acc_char:.2f}%)")

    # --------------------------
    # Predição aleatória
    # --------------------------
    elif not args.no_predict:
        print(color_text("\n🔍 Teste de predição aleatória...", Color.OKGREEN))
        _, h, w, c = model.input_shape
        sample = np.random.rand(1, h, w, c).astype("float32")
        preds = model.predict(sample)

        if isinstance(preds, list):
            for i, p in enumerate(preds):
                index = np.argmax(p)
                char = charset[index] if charset else str(index)
                print(color_text(f"→ char_{i+1}: {p.shape} | Predição: {char}", Color.OKCYAN))
        else:
            index = np.argmax(preds)
            char = charset[index] if charset else str(index)
            print(color_text(f"✔ Saída única: {preds.shape} | Predição: {char}", Color.OKCYAN))

    print(color_text("\n✅ Inspeção concluída!\n", Color.OKGREEN))

if __name__ == "__main__":
    main()
