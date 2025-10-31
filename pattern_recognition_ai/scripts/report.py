import argparse
import os
from datetime import datetime

import pandas as pd
from tabulate import tabulate
from src.config import settings


def generate_report(target_accuracy: float, project_deadline_str: str) -> None:
    # Caminho do log original
    log_file = os.path.join(settings.config["logs_dir"], "evaluation", "evaluation_log.csv")
    if not os.path.exists(log_file):
        print("❌ Arquivo de log 'evaluation_log.csv' não encontrado. Execute 'make evaluate'.")
        return

    # Lê o log completo
    df = pd.read_csv(log_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["type"] = df.get("type", "N/A")
    df["subtype"] = df.get("subtype", "N/A")

    # Calcula métricas adicionais
    df["progress_percentage"] = df["accuracy"] / target_accuracy * 100

    # Calcula prazo e status
    try:
        project_deadline = datetime.strptime(project_deadline_str, "%Y-%m-%d")
        df["days_remaining"] = (project_deadline - df["timestamp"]).dt.days
        df["required_daily_gain"] = df.apply(
            lambda row: (target_accuracy - row["accuracy"]) / row["days_remaining"]
            if row["days_remaining"] > 0 else 0,
            axis=1,
        )
        df["status"] = df["days_remaining"].apply(lambda x: "Dentro do Prazo" if x > 0 else "Atrasado")
    except ValueError:
        project_deadline = None
        df["days_remaining"] = None
        df["required_daily_gain"] = None
        df["status"] = "Prazo Inválido"

    # Última avaliação
    latest_eval = df.sort_values(by="timestamp", ascending=False).iloc[0]

    # Relatório geral
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE PROGRESSO DO MODELO DE IA")
    print("=" * 60)
    print(f"Última Avaliação: {latest_eval['timestamp'].strftime('%d/%m/%Y %H:%M')}")
    print("-" * 60)
    print("🎯 DESEMPENHO ATUAL")
    print(f"  - Acurácia Atual: {latest_eval['accuracy']:.2%}")
    print(f"  - Acurácia Alvo:   {target_accuracy:.2%}")
    print(f"  - Progresso para o Alvo: {latest_eval['progress_percentage']:.2f}%")
    print(f"  - Loss Atual: {latest_eval.get('loss', 0):.4f}")
    print(f"  - Imagens Processadas: {latest_eval.get('image_count', 0)}")

    print("\n🗓️ ACOMPANHAMENTO DE PRAZOS")
    if project_deadline is None:
        print("  - Prazo inválido informado. Use o formato YYYY-MM-DD.")
    else:
        print(f"  - Prazo Final: {project_deadline.strftime('%d/%m/%Y')}")
        print(f"  - Dias Restantes: {latest_eval['days_remaining']}")
        print(f"  - Ganho Diário Necessário: {latest_eval['required_daily_gain']:.3%}")
        print(f"  - Status: {latest_eval['status']}")

    # Agrupamento por tipo/subtipo
    grouped = df.groupby(["type", "subtype"]).agg(
        total_images=pd.NamedAgg(column="image_count", aggfunc="sum"),
        avg_accuracy=pd.NamedAgg(column="accuracy", aggfunc="mean"),
        avg_loss=pd.NamedAgg(column="loss", aggfunc="mean"),
        avg_progress=pd.NamedAgg(column="progress_percentage", aggfunc="mean")
    ).reset_index()

    print("\n📌 MÉTRICAS POR TIPO E SUBTIPO")
    print(tabulate(
        grouped,
        headers=["Tipo", "Subtipo", "Imagens", "Acurácia Média", "Loss Média", "Progresso Médio (%)"],
        tablefmt="fancy_grid",
        floatfmt=(".0f", ".0f", ".0f", ".4f", ".2f")
    ))

    # Média total
    total_images = df["image_count"].sum()
    overall_accuracy = df["accuracy"].mean()
    overall_loss = df["loss"].mean()
    overall_progress = df["progress_percentage"].mean()
    print("\n📊 MÉDIA TOTAL")
    print(tabulate([[
        total_images, overall_accuracy, overall_loss, overall_progress
    ]],
        headers=["Total Imagens", "Acurácia Média", "Loss Média", "Progresso Médio (%)"],
        tablefmt="fancy_grid",
        floatfmt=(".0f", ".2%", ".4f", ".2f")
    ))

    # Salva histórico completo
    history_dir = os.path.join(settings.config["logs_dir"], "evaluation")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, "report_history.csv")
    if os.path.exists(history_file):
        df.to_csv(history_file, mode="a", header=False, index=False)
    else:
        df.to_csv(history_file, index=False)
    print(f"\n✅ Histórico completo gravado em: {history_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera relatório de progresso do treinamento.")
    parser.add_argument("--target", type=float, default=0.98, help="Acurácia alvo (ex: 0.98).")
    parser.add_argument("--deadline", type=str, default="2026-12-31", help="Prazo final (YYYY-MM-DD).")
    args = parser.parse_args()

    generate_report(target_accuracy=args.target, project_deadline_str=args.deadline)


if __name__ == "__main__":
    main()
