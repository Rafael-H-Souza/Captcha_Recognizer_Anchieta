#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relatório consolidado de performance de modelos de IA (CAPTCHA predictor).
Versão hardening — coleta automática de todos os JSONs em qualquer subpasta (type/subtype).
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from tabulate import tabulate

# ==============================================
# Configuração de diretórios
# ==============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "reports"
EVAL_DIRS = [
    PROJECT_ROOT / "logs" / "evaluation",
    PROJECT_ROOT / "logs" / "evaluatio",
    PROJECT_ROOT / "logs" / "eval"
]

LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "evaluation_report.log"

# ==============================================
# Configuração de logging
# ==============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("report")

# ==============================================
# Função: coletar relatórios JSON
# ==============================================
def collect_json_reports() -> list[Path]:
    """Busca recursivamente relatórios JSON em múltiplas pastas possíveis."""
    found = []
    for directory in EVAL_DIRS:
        if directory.exists():
            found.extend(directory.rglob("*.json"))
    return list({f.resolve() for f in found})  # remove duplicados


# ==============================================
# Função: consolidar relatórios JSON
# ==============================================
def consolidate_reports() -> pd.DataFrame:
    """Lê todos os relatórios JSON e extrai métricas com detecção de tipo/subtipo."""
    records = []
    json_files = collect_json_reports()

    if not json_files:
        logger.warning("❌ Nenhum relatório JSON encontrado em logs/evaluation/.")
        return pd.DataFrame()

    logger.info(f"📂 Encontrados {len(json_files)} relatórios JSON...")

    for report_file in json_files:
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Inferência de type e subtype pela estrutura de pastas
            rel_path = report_file.relative_to(PROJECT_ROOT)
            parts = rel_path.parts
            type_, subtype = "unknown", "unknown"
            for i, p in enumerate(parts):
                if p in ("evaluation", "evaluatio", "eval") and i + 2 < len(parts):
                    type_, subtype = parts[i + 1], parts[i + 2]
                    break

            record = {
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "model_name": data.get("model_name", report_file.stem),
                "accuracy": float(data.get("accuracy", data.get("acc", 0.0))),
                "val_accuracy": float(data.get("val_accuracy", 0.0)),
                "loss": float(data.get("loss", 0.0)),
                "val_loss": float(data.get("val_loss", 0.0)),
                "image_count": int(data.get("images", data.get("image_count", 0))),
                "epochs": int(data.get("epochs", 0)),
                "type": type_,
                "subtype": subtype,
                "source_file": str(rel_path),
            }
            records.append(record)

        except Exception as e:
            logger.warning(f"⚠️ Erro ao processar {report_file.name}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"✅ {len(df)} relatórios válidos consolidados.")
    return df


# ==============================================
# Geração de relatório consolidado
# ==============================================
def generate_report(target_accuracy: float, project_deadline_str: str):
    report_csv = LOGS_DIR / "evaluation_report.csv"
    report_history = LOGS_DIR / "report_history.csv"

    df = consolidate_reports()
    if df.empty:
        logger.error("❌ Nenhum relatório válido encontrado.")
        return

    # Conversões e cálculos
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(datetime.now())
    df["progress_percentage"] = (df["accuracy"] / target_accuracy * 100).round(2)

    try:
        deadline = datetime.strptime(project_deadline_str, "%Y-%m-%d")
        df["days_remaining"] = (deadline - df["timestamp"]).dt.days.clip(lower=0)
    except Exception:
        df["days_remaining"] = None

    # Agrupamento
    grouped = (
        df.groupby(["type", "subtype"], dropna=False)
        .agg(
            total_models=("model_name", "count"),
            total_images=("image_count", "sum"),
            avg_accuracy=("accuracy", "mean"),
            avg_val_acc=("val_accuracy", "mean"),
            avg_loss=("loss", "mean"),
            avg_val_loss=("val_loss", "mean"),
            avg_progress=("progress_percentage", "mean"),
        )
        .reset_index()
    )

    # Salvar CSV principal
    grouped.to_csv(report_csv, index=False)
    logger.info(f"📁 CSV consolidado salvo em: {report_csv}")

    # Atualizar histórico
    summary = {
        "timestamp": datetime.now().isoformat(),
        "acc_mean": df["accuracy"].mean(),
        "val_acc_mean": df["val_accuracy"].mean(),
        "loss_mean": df["loss"].mean(),
        "val_loss_mean": df["val_loss"].mean(),
        "progress_mean": df["progress_percentage"].mean(),
    }
    pd.DataFrame([summary]).to_csv(
        report_history, mode="a", header=not report_history.exists(), index=False
    )

    # ==========================================
    # Impressão de resumo
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO CONSOLIDADO DE MODELOS")
    print("=" * 70)
    latest = df.sort_values("timestamp", ascending=False).iloc[0]
    print(f"🕒 Última Atualização: {latest['timestamp']:%d/%m/%Y %H:%M}")
    print(f"🎯 Acurácia Atual:    {latest['accuracy']:.2%}")
    print(f"🎯 Acurácia Alvo:     {target_accuracy:.2%}")
    print(f"📈 Progresso:         {latest['progress_percentage']:.2f}%")
    print(f"📉 Loss Atual:        {latest['loss']:.4f}")
    print(f"🖼️  Imagens:           {int(latest['image_count'])}")
    print(f"🔢 Épocas:             {int(latest['epochs'])}")

    print("\n📌 MÉTRICAS AGRUPADAS POR TIPO E SUBTIPO")
    print(tabulate(
        grouped,
        headers=[
            "Tipo", "Subtipo", "Modelos", "Total Imagens",
            "Acurácia Média", "Val Acc Média", "Loss Média", "Val Loss Média", "Progresso (%)"
        ],
        tablefmt="fancy_grid",
        floatfmt=(".0f", ".0f", ".2%", ".2%", ".4f", ".4f", ".2f"),
    ))

    print("\n📈 MÉTRICAS GERAIS DO PROJETO")
    print(tabulate([[
        df["image_count"].sum(),
        df["accuracy"].mean(),
        df["val_accuracy"].mean(),
        df["loss"].mean(),
        df["val_loss"].mean(),
        df["progress_percentage"].mean()
    ]],
        headers=["Total Imagens", "Acc Média", "Val Acc Média", "Loss Média", "Val Loss Média", "Progresso (%)"],
        tablefmt="fancy_grid",
        floatfmt=(".0f", ".2%", ".2%", ".4f", ".4f", ".2f"),
    ))

    print(f"\n✅ Histórico salvo em: {report_history.resolve()}")


# ==============================================
# MAIN
# ==============================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gera relatório consolidado de progresso dos modelos.")
    parser.add_argument("--target", type=float, default=0.95, help="Acurácia alvo (ex: 0.95)")
    parser.add_argument("--deadline", type=str, default="2025-11-14", help="Prazo final (YYYY-MM-DD)")
    args = parser.parse_args()

    logger.info("🚀 Iniciando geração de relatório...")
    try:
        generate_report(target_accuracy=args.target, project_deadline_str=args.deadline)
    except Exception as e:
        logger.exception(f"❌ Erro ao gerar relatório: {e}")
        print(f"❌ Erro ao gerar relatório: {e}")


if __name__ == "__main__":
    main()
