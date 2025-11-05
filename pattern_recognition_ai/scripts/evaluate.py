import os
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import custom_object_scope

from src.config import settings
from src.data_loader.dataset import CaptchaDataLoader
from src.utils.logger import get_logger


# ======================================================
# 🧩 Camada customizada compatível com slicing antigo
# ======================================================
class GetItem(tf.keras.layers.Layer):
    def __init__(self, index=None, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        try:
            return inputs[:, self.index] if self.index is not None else inputs
        except Exception as e:
            tf.print("⚠️ Erro no GetItem:", e)
            return inputs


# ======================================================
# 🧱 Reconstrução segura de modelos antigos
# ======================================================
def rebuild_model_safely(old_model_path: str) -> str:
    logger = get_logger("Evaluation-Script")

    try:
        with custom_object_scope({'GetItem': GetItem}):
            old_model = tf.keras.models.load_model(old_model_path, compile=False)

        input_shape = old_model.input_shape[1:]
        inputs = tf.keras.layers.Input(shape=input_shape)

        try:
            outputs = old_model(inputs, training=False)
        except TypeError:
            logger.warning(f"⚠️ Slice antigo detectado em {old_model_path}, aplicando fallback...")
            logger.debug(traceback.format_exc())
            outputs = old_model(inputs)

        new_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        fixed_path = old_model_path.replace(".h5", "_fixed.h5")
        new_model.save(fixed_path)
        logger.info(f"✅ Modelo compatível salvo em: {fixed_path}")

        return fixed_path

    except Exception as e:
        logger.error(f"❌ Falha ao reconstruir {old_model_path}: {e}")
        logger.debug(traceback.format_exc())
        return old_model_path


# ======================================================
# 🚀 Avaliação dos modelos exportados
# ======================================================
def evaluate_model() -> None:
    logger = get_logger("Evaluation-Script")

    exported_dir = os.path.join(settings.BASE_RAIZ, "models", "exported")
    log_dir = os.path.join(settings.config["logs_dir"], "evaluation")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "evaluation_log.csv")

    if not os.path.exists(exported_dir):
        logger.error(f"❌ Diretório não encontrado: {exported_dir}")
        return

    model_files = [f for f in os.listdir(exported_dir) if f.endswith(".h5")]
    if not model_files:
        logger.warning(f"⚠️ Nenhum modelo encontrado em {exported_dir}")
        return

    data_loader = CaptchaDataLoader(settings.config)
    batch_size = settings.config.get("batch_size", 32)
    results = []

    for model_name in model_files:
        model_path = os.path.join(exported_dir, model_name)
        logger.info(f"\n📦 Avaliando modelo: {model_name}")

        fixed_path = rebuild_model_safely(model_path)

        try:
            with custom_object_scope({'GetItem': GetItem}):
                model = tf.keras.models.load_model(fixed_path, compile=False)
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            logger.error(f"❌ Falha ao carregar modelo {model_name}: {e}")
            logger.debug(traceback.format_exc())
            continue

        for t in data_loader.TYPES:
            for st in data_loader.SUBTYPES:
                dataset_dir = os.path.join(settings.config["dataset_dir"], t, st)
                if not os.path.exists(dataset_dir):
                    continue

                try:
                    x_train, x_val, y_train_dict, y_val_dict = data_loader.load_data_from_path(dataset_dir)
                except Exception as e:
                    logger.error(f"Erro ao carregar {dataset_dir}: {e}")
                    logger.debug(traceback.format_exc())
                    continue

                if len(x_val) == 0:
                    logger.warning(f"⚠️ Nenhuma imagem em {dataset_dir}")
                    continue

                if isinstance(y_val_dict, dict):
                    y_val = tuple(y_val_dict[k] for k in sorted(y_val_dict.keys())) \
                        if len(y_val_dict) > 1 else list(y_val_dict.values())[0]
                else:
                    y_val = y_val_dict

                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                try:
                    results_eval = model.evaluate(val_dataset, verbose=0)
                    if isinstance(results_eval, list):
                        loss, acc = results_eval[0], results_eval[-1]
                    else:
                        loss, acc = 0.0, results_eval

                    logger.info(f"✅ {t}/{st}: loss={loss:.4f} | acc={acc*100:.2f}%")

                    results.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model_name,
                        "type": t,
                        "subtype": st,
                        "loss": loss,
                        "accuracy": acc
                    })

                except Exception as e:
                    logger.error(f"Erro ao avaliar {model_name} ({t}/{st}): {e}")
                    logger.debug(traceback.format_exc())
                    continue

    # ======================================================
    # 🧾 Salvar resultados + gráfico
    # ======================================================
    if results:
        df = pd.DataFrame(results)
        df.to_csv(log_path, index=False)
        logger.info(f"\n📊 Log salvo em: {log_path}")

        # Estatísticas gerais
        summary = df.groupby("model")[["loss", "accuracy"]].mean().sort_values("accuracy", ascending=False)
        logger.info("\n🏁 Resumo médio por modelo:")
        for model, row in summary.iterrows():
            logger.info(f"   • {model}: acc={row['accuracy']*100:.2f}% | loss={row['loss']:.4f}")

        # Gráficos
        fig, ax = plt.subplots(figsize=(10, 5))
        summary["accuracy"].plot(kind="bar", ax=ax)
        ax.set_title("Acurácia Média por Modelo")
        ax.set_ylabel("Acurácia")
        ax.set_xlabel("Modelos")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        plot_path = os.path.join(log_dir, "evaluation_accuracy_chart.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"📈 Gráfico gerado em: {plot_path}")

    else:
        logger.warning("\n⚠️ Nenhum resultado válido — verifique os datasets ou modelos.")


# ======================================================
# 🎯 Execução direta
# ======================================================
if __name__ == "__main__":
    evaluate_model()
