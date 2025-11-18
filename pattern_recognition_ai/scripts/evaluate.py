#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 Avaliação automática de modelos CAPTCHA (versão aprimorada)
Autor: Rafael H. Souza

- Detecta automaticamente modelos .keras
- Avalia modelos single_char e five_char (multi-saída)
- Evita avisos do sklearn em datasets com 1 classe
- Mostra distribuição de classes em y_val.npy
- Gera relatórios JSON e CSV consolidados
"""

import os
import glob
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras import models

# ⚙️ Suprimir apenas warnings desnecessários
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -------------------------------
# Utilidades
# -------------------------------
def device_name() -> str:
    return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_subdirs(path: str):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def latest_file(paths: list[str]) -> str | None:
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


# -------------------------------
# Carregamento de dados
# -------------------------------
def load_data(dataset_root: str, type_name: str, subtype: str | None):
    data_path = os.path.join(dataset_root, type_name, subtype or "")
    x_val_path = os.path.join(data_path, "x_val.npy")
    y_val_path = os.path.join(data_path, "y_val.npy")

    if not (os.path.exists(x_val_path) and os.path.exists(y_val_path)):
        raise FileNotFoundError(f"Arquivos de validação não encontrados em {data_path}")

    X = np.load(x_val_path).astype("float32") / 255.0
    y = np.load(y_val_path, allow_pickle=True)

    if X.ndim == 3:
        X = np.expand_dims(X, -1)

    if type_name == "single_char":
        y = np.array([str(c) for c in y])
    else:
        y = np.array(["".join(str(ch) for ch in row).ljust(5)[:5] for row in y])

    # 🧠 Mostra distribuição de classes para depuração
    unique, counts = np.unique(list("".join(y)), return_counts=True)
    print(f"📂 Dataset: {data_path} | {len(X)} imagens | {len(unique)} classes únicas")
    print("   Distribuição de classes:", dict(zip(unique, counts)))

    return X, y


# -------------------------------
# Persistência de relatórios
# -------------------------------
def save_json_report(base_logs: str, model_name: str, model_type: str, model_subtype: str,
                     report_dict: dict, suffix: str = ""):
    folder = os.path.join(base_logs, "evaluation", model_type, model_subtype)
    ensure_dir(folder)
    path = os.path.join(folder, f"{model_name}_report{suffix}.json")
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"📊 Relatório salvo: {os.path.basename(path)}")


def update_csv(base_logs: str, row: dict):
    path = os.path.join(base_logs, "reports", "evaluation_report.csv")
    ensure_dir(os.path.dirname(path))
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


# -------------------------------
# Avaliação single_char
# -------------------------------
def evaluate_single_char(model, X_val, y_val, model_name, model_type, model_subtype, logs_dir):
    unique = sorted(list(set("".join(y_val) + " ")))
    num_classes = model.output_shape[-1]
    unique = unique[:num_classes] if len(unique) > num_classes else unique
    idx = {c: i for i, c in enumerate(unique)}
    y_true_enc = np.array([idx.get(c, 0) for c in y_val])

    ds_val = tf.data.Dataset.from_tensor_slices(X_val).batch(64)
    t0 = time.time()
    y_pred_raw = model.predict(ds_val, verbose=0)
    duration = time.time() - t0

    y_pred = np.argmax(y_pred_raw, axis=-1)
    loss, _ = model.evaluate(X_val, tf.keras.utils.to_categorical(y_true_enc, num_classes), verbose=0)

    acc = accuracy_score(y_true_enc, y_pred)
    f1w = f1_score(y_true_enc, y_pred, average="weighted")
    cm = confusion_matrix(y_true_enc, y_pred, labels=list(range(len(unique)))).tolist()

    # ⚙️ Se houver apenas uma classe, evita warnings
    report = classification_report(
        y_true_enc, y_pred, labels=list(range(len(unique))),
        target_names=unique, output_dict=True, zero_division=0
    )

    report.update({
        "timestamp": datetime.now().isoformat(),
        "type": model_type,
        "subtype": model_subtype,
        "model_name": model_name,
        "num_images": int(X_val.shape[0]),
        "num_classes_detected": len(unique),
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
        "loss": float(loss),
        "confusion_matrix": cm,
        "runtime_sec": round(duration, 2),
        "device": device_name(),
        "tf_version": tf.__version__,
    })

    save_json_report(logs_dir, model_name, model_type, model_subtype, report)
    update_csv(logs_dir, {
        "timestamp": report["timestamp"],
        "model_name": model_name,
        "type": model_type,
        "subtype": model_subtype,
        "accuracy": report["accuracy"],
        "f1_weighted": report["f1_weighted"],
        "loss": report["loss"],
        "device": report["device"],
    })


# -------------------------------
# Avaliação five_char (multi-output)
# -------------------------------
def evaluate_five_char(model, X_val, y_val, model_name, model_type, model_subtype, logs_dir):
    if not isinstance(model.output_shape, (list, tuple)):
        return evaluate_single_char(model, X_val, y_val, model_name, model_type, model_subtype, logs_dir)

    unique = sorted(list(set("".join(y_val) + " ")))
    num_classes = model.output_shape[0][-1]
    unique = unique[:num_classes] if len(unique) > num_classes else unique
    idx = {c: i for i, c in enumerate(unique)}

    y_true_idx = np.array([[idx.get(c, 0) for c in s] for s in y_val])
    y_true_onehot = [tf.keras.utils.to_categorical(y_true_idx[:, i], num_classes) for i in range(5)]

    ds_val = tf.data.Dataset.from_tensor_slices(X_val).batch(64)
    t0 = time.time()
    y_pred_list = model.predict(ds_val, verbose=0)
    duration = time.time() - t0
    losses = model.evaluate(X_val, y_true_onehot, verbose=0)

    total_loss = float(losses[0]) if isinstance(losses, (list, tuple)) else float(losses)
    pos_acc, pos_f1w = [], []

    for i, y_pred_out in enumerate(y_pred_list):
        y_pred_i = np.argmax(y_pred_out, axis=-1)
        acc_i = accuracy_score(y_true_idx[:, i], y_pred_i)
        f1_i = f1_score(y_true_idx[:, i], y_pred_i, average="weighted")
        pos_acc.append(acc_i)
        pos_f1w.append(f1_i)

    seq_exact = float(np.mean(np.all(
        np.stack([np.argmax(p, axis=-1) for p in y_pred_list], axis=1) == y_true_idx, axis=1
    )))

    overall = {
        "timestamp": datetime.now().isoformat(),
        "type": model_type,
        "subtype": model_subtype,
        "model_name": model_name,
        "num_images": int(X_val.shape[0]),
        "sequence_exact_accuracy": seq_exact,
        "positions_accuracy_mean": float(np.mean(pos_acc)),
        "positions_f1_weighted_mean": float(np.mean(pos_f1w)),
        "loss_total": total_loss,
        "runtime_sec": round(duration, 2),
        "num_classes_detected": len(unique),
        "device": device_name(),
        "tf_version": tf.__version__,
    }

    save_json_report(logs_dir, model_name, model_type, model_subtype, overall, suffix="_overall")
    update_csv(logs_dir, {
        "timestamp": overall["timestamp"],
        "model_name": f"{model_name}_overall",
        "type": model_type,
        "subtype": model_subtype,
        "accuracy": overall["sequence_exact_accuracy"],
        "f1_weighted": overall["positions_f1_weighted_mean"],
        "loss": overall["loss_total"],
        "device": overall["device"],
    })


# -------------------------------
# Execução principal
# -------------------------------
if __name__ == "__main__":
    dataset_root = "data/processed"
    models_dir = "models/apurados"
    logs_dir = "logs"

    if not (os.path.exists(models_dir) and os.listdir(models_dir)):
        models_dir = "models/exported"

    ensure_dir(logs_dir)
    print(f"\n📁 Diretório de modelos detectado: {models_dir}\n")

    for type_name in ["single_char", "five_char"]:
        type_dir = os.path.join(dataset_root, type_name)
        if not os.path.exists(type_dir):
            continue

        for subtype in list_subdirs(type_dir):
            pattern = os.path.join(models_dir, f"{type_name}_{subtype}_model*.keras")
            candidates = glob.glob(pattern)
            if not candidates:
                print(f"⚠️ Nenhum modelo encontrado para {type_name}/{subtype}")
                continue

            model_path = latest_file(candidates)
            model_name = os.path.basename(model_path).replace(".keras", "")
            print(f"🧠 Avaliando modelo: {model_name} ...")

            try:
                model = models.load_model(model_path)
                X_val, y_val = load_data(dataset_root, type_name, subtype)
                if type_name == "single_char":
                    evaluate_single_char(model, X_val, y_val, model_name, type_name, subtype, logs_dir)
                else:
                    evaluate_five_char(model, X_val, y_val, model_name, type_name, subtype, logs_dir)
            except Exception as e:
                print(f"❌ Erro avaliando {model_name}: {e}")

    print("\n✅ Avaliação concluída. Relatórios gerados em logs/evaluation e logs/reports/")
