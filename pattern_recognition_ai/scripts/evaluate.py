import os
import glob
import json
import time
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models, Input, Model

# -------------------------------
# CONFIGURAR SUPRESSÃO DE WARNINGS DO TF
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error


# ===============================================================
# UNIFIED PREDICTOR - Carrega e unifica todos os modelos apurados
# ===============================================================
class UnifiedPredictor:
    def __init__(self, models_dir="models/apurados", save_path="models/final/unified_models.pkl"):
        self.models = {}
        self.models_dir = models_dir
        self.save_path = save_path
        self.backup_path = save_path.replace(".pkl", "_backup.pkl")
        self.unified_model_path = os.path.join(models_dir, "all_models_unified.keras")

        if os.path.exists(self.unified_model_path):
            print(f"\n🚀 Carregando modelo unificado existente: {self.unified_model_path}")
            self.unified_model = models.load_model(self.unified_model_path)
            print("✅ Modelo unificado carregado com sucesso.\n")
        else:
            print(f"\n⚠️ Nenhum modelo unificado encontrado. Tentando unir modelos em {models_dir}")
            self._load_all(models_dir)
            if self.models:
                self.unified_model = self._build_unified_model()
                self.unified_model.save(self.unified_model_path)
                print(f"✅ Novo modelo unificado salvo em: {self.unified_model_path}")
                self._save_backup_pkl()
            else:
                print("❌ Nenhum modelo encontrado para unificação.")

    def _loading_animation(self, message="Carregando"):
        for c in ['|', '/', '-', '\\']:
            sys.stdout.write(f'\r{message} {c}')
            sys.stdout.flush()
            time.sleep(0.1)

    def _load_all(self, models_dir):
        print(f"\n📦 Procurando modelos em: {models_dir}")
        model_paths = glob.glob(os.path.join(models_dir, "*.keras"))
        if not model_paths:
            print("❌ Nenhum modelo .keras encontrado.")
            return

        for path in model_paths:
            self._loading_animation("Carregando modelos")
            name = os.path.basename(path).replace(".keras", "")
            try:
                self.models[name] = models.load_model(path)
                print(f"\r✅ Modelo carregado: {name}        ")
            except Exception as e:
                print(f"\r⚠️ Falha ao carregar {name}: {e}")

    def _merge_with_existing(self):
        """Mescla os modelos novos com o .pkl anterior, se existir."""
        if os.path.exists(self.save_path):
            print(f"\n🔄 Mesclando aprendizado com {self.save_path} ...")
            try:
                with open(self.save_path, "rb") as f:
                    old_models = pickle.load(f)
                self.models = {**old_models, **self.models}
                print("✅ Aprendizado mesclado com sucesso.")
            except Exception as e:
                print(f"⚠️ Falha ao mesclar aprendizado anterior: {e}")

    def _save_backup_pkl(self):
        """Salva modelos individuais em .pkl como backup, sem usar na inferência."""
        if self.models:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            if os.path.exists(self.save_path):
                print(f"💾 Criando backup de {self.save_path}")
                os.rename(self.save_path, self.backup_path)

            self._merge_with_existing()

            with open(self.save_path, "wb") as f:
                pickle.dump(self.models, f)
            print(f"✅ Backup .pkl unificado salvo: {self.save_path}")

    def _build_unified_model(self):
        """Cria um modelo ensemble concatenando as saídas dos modelos carregados."""
        if not self.models:
            raise ValueError("Nenhum modelo carregado para unificação.")

        sample_model = list(self.models.values())[0]
        input_shape = sample_model.input_shape[1:]
        unified_input = Input(shape=input_shape, name="unified_input")

        flattened_outputs = []
        for i, (name, model) in enumerate(self.models.items()):
            model._name = f"model_{i}"
            output = model(unified_input)
            if isinstance(output, (list, tuple)):
                for j, out in enumerate(output):
                    flat = layers.Flatten(name=f"flat_{i}_{j}")(out)
                    flattened_outputs.append(flat)
            else:
                flat = layers.Flatten(name=f"flat_{i}")(output)
                flattened_outputs.append(flat)

        merged = layers.Concatenate(axis=-1, name="merged")(flattened_outputs)
        x = layers.Dense(256, activation="relu", name="dense_1")(merged)
        x = layers.Dropout(0.3)(x)
        final_output = layers.Dense(36, activation="softmax", name="final_output")(x)

        unified_model = Model(inputs=unified_input, outputs=final_output, name="all_models_unified")
        unified_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return unified_model

    def predict_batch(self, images):
        """Predição em batch usando o modelo unificado real (.keras)."""
        if hasattr(self, "unified_model"):
            preds = self.unified_model.predict(np.array(images), verbose=0)
            decoded = [chr(97 + int(np.argmax(p))) for p in preds]  # ajuste para charset real
            return decoded
        else:
            print("⚠️ Nenhum modelo unificado carregado.")
            return ["-----"] * len(images)


# ===============================================================
# AVALIADOR PRINCIPAL
# ===============================================================
class Evaluator:
    def __init__(self, dataset_dir, apurados_dir="models/apurados", logs_dir="logs"):
        self.dataset_dir = dataset_dir
        self.apurados_dir = apurados_dir
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

    def load_data(self, type_name, subtype=None):
        if subtype:
            data_path = os.path.join(self.dataset_dir, type_name, subtype)
        else:
            data_path = os.path.join(self.dataset_dir)

        x_val_path = os.path.join(data_path, "x_val.npy")
        y_val_path = os.path.join(data_path, "y_val.npy")

        if not os.path.exists(x_val_path) or not os.path.exists(y_val_path):
            raise FileNotFoundError(f"Arquivos de validação não encontrados em {data_path}.")

        X_val = np.load(x_val_path).astype("float32") / 255.0
        y_val = np.load(y_val_path, allow_pickle=True)

        if X_val.ndim == 3:
            X_val = np.expand_dims(X_val, -1)

        if type_name == "single_char":
            y_val = np.array([str(c) for c in y_val])
        else:
            y_val = np.array([''.join(str(ch) for ch in y).ljust(5)[:5] for y in y_val])

        print(f"📂 Dataset carregado: {data_path} | Tipo: {type_name} | Imagens: {len(X_val)}")
        return X_val, y_val

    def save_report(self, report_dict, model_name, model_type, model_subtype, suffix="", image_count=0):
        report_subdir = os.path.join(self.logs_dir, "evaluation", model_type, model_subtype)
        os.makedirs(report_subdir, exist_ok=True)

        json_path = os.path.join(report_subdir, f"{model_name}_report{suffix}.json")
        txt_path = os.path.join(report_subdir, f"{model_name}_report{suffix}.txt")

        report_dict["timestamp"] = datetime.now().isoformat()
        report_dict["type"] = model_type
        report_dict["subtype"] = model_subtype
        report_dict["accuracy"] = np.mean([v.get("f1-score", 0) for k, v in report_dict.items() if isinstance(v, dict)])
        report_dict["loss"] = 0.0
        report_dict["image_count"] = image_count

        with open(json_path, "w") as f_json:
            json.dump(report_dict, f_json, indent=4)

        with open(txt_path, "w") as f_txt:
            for key, metrics in report_dict.items():
                f_txt.write(f"{key}: {metrics}\n")

        self.update_csv(report_dict, model_name)

    def update_csv(self, report_data, model_name):
        reports_dir = os.path.join(self.logs_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        csv_path = os.path.join(reports_dir, "evaluation_report.csv")
        history_path = os.path.join(reports_dir, "report_history.csv")

        new_entry = {
            "timestamp": report_data.get("timestamp"),
            "model_name": model_name,
            "type": report_data.get("type"),
            "subtype": report_data.get("subtype"),
            "accuracy": report_data.get("accuracy"),
            "loss": report_data.get("loss"),
            "image_count": report_data.get("image_count")
        }

        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        df_hist = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()
        df_hist = pd.concat([df_hist, pd.DataFrame([new_entry])], ignore_index=True)
        df_hist.to_csv(history_path, index=False)

        print(f"🧩 Métricas atualizadas em: {csv_path}")

    def evaluate_model(self, model_path, type_name):
        model_name = os.path.basename(model_path).replace(".keras", "")
        parts = model_name.split("_")
        model_type = parts[0] if len(parts) < 3 else "_".join(parts[:2])
        model_subtype = parts[1] if len(parts) == 2 else "_".join(parts[2:-1]) if len(parts) > 2 else "unknown"

        print(f"\n📝 Avaliando modelo: {model_name} (Tipo: {model_type}, Subtipo: {model_subtype})")

        model = models.load_model(model_path)
        X_val, y_val = self.load_data(model_type, model_subtype)

        num_classes = model.output_shape[-1] if type_name == "single_char" else \
            (model.output_shape[0][-1] if isinstance(model.output_shape, list) else model.output_shape[-1])

        unique_chars = sorted(list(set(''.join(y_val) + ' ')))
        if len(unique_chars) > num_classes:
            unique_chars = unique_chars[:num_classes]
        char_to_index = {c: i for i, c in enumerate(unique_chars)}

        try:
            if type_name == "single_char":
                y_val_enc = tf.keras.utils.to_categorical([char_to_index.get(c, 0) for c in y_val], num_classes)
                model.evaluate(X_val, y_val_enc, verbose=0)
                y_pred = np.argmax(model.predict(X_val), axis=-1)
                report = classification_report(
                    y_val, [unique_chars[i] for i in y_pred],
                    labels=unique_chars, output_dict=True, zero_division=0
                )
                self.save_report(report, model_name, model_type, model_subtype, image_count=X_val.shape[0])
            else:
                y_val_split = np.array([[char_to_index.get(c, 0) for c in text] for text in y_val])
                y_val_enc = [tf.keras.utils.to_categorical(y_val_split[:, i], num_classes) for i in range(5)]
                model.evaluate(X_val, y_val_enc, verbose=0)
                y_pred_list = model.predict(X_val)
                for i, y_pred_out in enumerate(y_pred_list):
                    y_pred_i = np.argmax(y_pred_out, axis=-1)
                    y_true_i = y_val_split[:, i]
                    report = classification_report(
                        y_true_i, y_pred_i,
                        labels=list(range(num_classes)),
                        target_names=unique_chars,
                        output_dict=True, zero_division=0
                    )
                    self.save_report(report, model_name, model_type, model_subtype, suffix=f"_saida_{i}", image_count=X_val.shape[0])

        except Exception as e:
            print(f"❌ Erro durante a avaliação: {e}")

        print("✅ Avaliação concluída.\n")

    def evaluate_all_models(self):
        print("\n🧠 Iniciando avaliação unificada (all_models_unified)...")
        unified = UnifiedPredictor(self.apurados_dir)

        type_name = "five_char"
        subtypes_path = os.path.join(self.dataset_dir, type_name)
        subtypes = [d for d in os.listdir(subtypes_path) if os.path.isdir(os.path.join(subtypes_path, d))] if os.path.exists(subtypes_path) else []
        if not subtypes:
            print("❌ Nenhum dataset de five_char encontrado para avaliação unificada.")
            return

        subtype = subtypes[0]
        X_val, y_val = self.load_data(type_name, subtype)
        preds = unified.predict_batch(X_val)

        y_true = [''.join(str(ch) for ch in y) for y in y_val]
        report = classification_report(y_true, preds, output_dict=True, zero_division=0)

        self.save_report(report, "all_models_unified", "multi", "ensemble", image_count=len(X_val))
        print("✅ Avaliação unificada concluída.\n")


# ===============================================================
# EXECUÇÃO PRINCIPAL
# ===============================================================
if __name__ == "__main__":
    base_dir = "data/processed"
    apurados_dir = "models/apurados"
    logs_dir = "logs"

    evaluator = Evaluator(base_dir, apurados_dir, logs_dir)

    for type_name in ["single_char", "five_char"]:
        type_path = os.path.join(base_dir, type_name)
        if not os.path.exists(type_path):
            continue
        for subtype in os.listdir(type_path):
            dataset_dir = os.path.join(type_path, subtype)
            if not os.path.isdir(dataset_dir):
                continue

            model_pattern = os.path.join(apurados_dir, f"{type_name}_{subtype}_model.keras")
            model_files = glob.glob(model_pattern)
            if not model_files:
                print(f"⚠️ Nenhum modelo encontrado para {type_name} / {subtype}")
                continue

            model_path = model_files[0]
            evaluator.evaluate_model(model_path, type_name)

    evaluator.evaluate_all_models()

    print("\n✅ Todas as métricas foram atualizadas automaticamente!\n")
