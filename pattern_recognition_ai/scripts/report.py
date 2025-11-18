#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report Consolidado – Compatível com TODOS os JSONs do Rafael
Aceita:
 - single_char
 - five_char
 - ensemble
 - formatos antigos e novos
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "reports"
EVAL_DIRS = [
    PROJECT_ROOT / "logs" / "evaluation",
    PROJECT_ROOT / "logs" / "evaluatio",
    PROJECT_ROOT / "logs" / "eval",
]

LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "evaluation_report.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("report")


# ==========================================================
# Encontrar JSONs
# ==========================================================
def collect_json_reports() -> list[Path]:
    found = []
    for d in EVAL_DIRS:
        if d.exists():
            found.extend(d.rglob("*.json"))
    return sorted(set(found))


# ==========================================================
# NORMALIZAR JSON → transforma qualquer estrutura em padrão único
# ==========================================================
def normalize_json(data: dict, file_path: Path) -> dict | None:

    # ========== 1️⃣ SINGLE_CHAR ==========
    if "accuracy" in data and "loss" in data:

        return {
            "timestamp": data.get("timestamp"),
            "type": data.get("type", "single_char"),
            "subtype": data.get("subtype", "unknown"),
            "model_name": data.get("model_name", file_path.stem),
            "accuracy": float(data.get("accuracy", 0)),
            "val_accuracy": float(data.get("val_accuracy", 0)),
            "loss": float(data.get("loss", 0)),
            "val_loss": float(data.get("val_loss", 0)),
            "images": int(data.get("num_images", data.get("image_count", 0))),
            "epochs": int(data.get("epochs", 0)),
        }

    # ========== 2️⃣ FIVE_CHAR ==========
    if "sequence_exact_accuracy" in data:

        return {
            "timestamp": data.get("timestamp"),
            "type": data.get("type", "five_char"),
            "subtype": data.get("subtype", "unknown"),
            "model_name": data.get("model_name", file_path.stem),

            # não existe accuracy real → usamos a exata
            "accuracy": float(data.get("sequence_exact_accuracy", 0)),
            "val_accuracy": float(data.get("positions_accuracy_mean", 0)),

            "loss": float(data.get("loss_total", 0)),
            "val_loss": 0.0,

            "images": int(data.get("num_images", 0)),
            "epochs": 0,   # five_char não usa epochs
        }

    # ========== 3️⃣ ENSEMBLE ==========
    if "ensemble_accuracy" in data:
        return {
            "timestamp": data.get("timestamp"),
            "type": "multi",
            "subtype": "ensemble",
            "model_name": data.get("model_name", file_path.stem),
            "accuracy": float(data.get("ensemble_accuracy", 0)),
            "val_accuracy": 0.0,
            "loss": 0.0,
            "val_loss": 0.0,
            "images": int(data.get("num_images", 0)),
            "epochs": 0,
        }

    # ========== SEM PADRÃO ==========
    return None


# ==========================================================
# Consolidar relatórios
# ==========================================================
def consolidate_reports() -> pd.DataFrame:

    files = collect_json_reports()
    logger.info(f"📂 Encontrados {len(files)} JSONs brutos...")

    records = []

    for json_file in files:
        try:
            data = json.loads(Path(json_file).read_text(encoding="utf-8"))
            norm = normalize_json(data, json_file)

            if norm is None:
                logger.warning(f"⛔ Ignorado (não reconhecido): {json_file.name}")
                continue

            records.append(norm)

        except Exception as e:
            logger.warning(f"⚠ Erro lendo {json_file.name}: {e}")

    if not records:
        logger.error("❌ Nenhum JSON válido.")
        return pd.DataFrame()

    logger.info(f"✅ {len(records)} JSONs válidos consolidados.")
    return pd.DataFrame(records)


# ==========================================================
# Gerar relatório final
# ==========================================================
def generate_report(target_accuracy: float, deadline_str: str):

    df = consolidate_reports()
    if df.empty:
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["progress"] = (df["accuracy"] / target_accuracy * 100).clip(0, 200)

    # Agrupado
    grouped = (
        df.groupby(["type", "subtype"])
          .agg(
              models=("model_name", "count"),
              images=("images", "sum"),
              acc_mean=("accuracy", "mean"),
              val_mean=("val_accuracy", "mean"),
              loss_mean=("loss", "mean"),
          )
          .reset_index()
    )

    print("\n" + "="*70)
    print("📊 RELATÓRIO FINAL – JSONs Compatíveis")
    print("="*70)

    latest = df.sort_values("timestamp", ascending=False).iloc[0]

    print(f"🕒 Último:       {latest['timestamp']:%d/%m/%Y %H:%M}")
    print(f"🎯 Accuracy:     {latest['accuracy']:.2%}")
    print(f"📉 Loss:         {latest['loss']:.4f}")
    print(f"📈 Progresso:    {latest['progress']:.1f}%")

    print("\n📌 AGRUPADO POR TIPO")
    print(tabulate(
        grouped,
        headers=["Tipo", "Subtipo", "Modelos", "Imagens",
                 "Acc Média", "Val Média", "Loss Média"],
        tablefmt="fancy_grid",
        floatfmt=(".0f",".0f",".2%",".2%",".4f"),
    ))

    csv_out = LOGS_DIR / "evaluation_report.csv"
    grouped.to_csv(csv_out, index=False)
    print(f"\n💾 CSV salvo em: {csv_out}")


# ==========================================================
# MAIN
# ==========================================================
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--deadline", type=str, default="2025-11-14")
    args = parser.parse_args()

    generate_report(args.target, args.deadline)


if __name__ == "__main__":
    main()
