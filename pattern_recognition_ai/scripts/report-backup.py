import argparse
import os
from datetime import datetime

import pandas as pd
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

    # Calcula métricas adicionais para cada linha
    df["progress_percentage"] = df["accuracy"] / target_accuracy * 100

    # Calcula prazo e status
    try:
        project_deadline = datetime.strptime(project_deadline_str, "%Y-%m-%d")
        df["days_remaining"] = (project_deadline - df["timestamp"]).dt.days
        df["required_daily_gain"] = (target_accuracy - df["accuracy"]) / df["days_remaining"]
        df["status"] = df["days_remaining"].apply(lambda x: "Dentro do Prazo" if x > 0 else "Atrasado")
    except ValueError:
        df["days_remaining"] = None
        df["required_daily_gain"] = None
        df["status"] = "Prazo Inválido"

    # Última avaliação
    latest_eval = df.sort_values(by="timestamp", ascending=False).iloc[0]

    # Relatório no console
    print("\n" + "=" * 50)
    print("📊 RELATÓRIO DE PROGRESSO DO MODELO DE IA")
    print("=" * 50)
    print(f"Última Avaliação: {latest_eval['timestamp'].strftime('%d/%m/%Y %H:%M')}")
    print("-" * 50)
    print("🎯 DESEMPENHO ATUAL")
    print(f"  - Acurácia Atual: {latest_eval['accuracy']:.2%}")
    print(f"  - Acurácia Alvo:   {target_accuracy:.2%}")
    print(f"  - Progresso para o Alvo: {latest_eval['progress_percentage']:.2f}%")
    if "loss" in latest_eval:
        print(f"  - Loss Atual: {latest_eval.get('loss', 0):.4f}")
    if "image_count" in latest_eval:
        print(f"  - Imagens Processadas: {latest_eval.get('image_count', 0)}")
    print("\n🗓️ ACOMPANHAMENTO DE PRAZOS")
    if project_deadline is None:
        print("  - Prazo inválido informado. Use o formato YYYY-MM-DD.")
    else:
        print(f"  - Prazo Final: {project_deadline.strftime('%d/%m/%Y')}")
        print(f"  - Dias Restantes: {latest_eval['days_remaining']}")
        print(f"  - Ganho Diário Necessário: {latest_eval['required_daily_gain']:.3%}")
        print(f"  - Status: {latest_eval['status']}")

    print("\n📈 HISTÓRICO DE EVOLUÇÃO COMPLETO")
    print(df.to_string(index=False))
    print("=" * 50)

    # Salva histórico completo
    history_file = os.path.join(settings.config["logs_dir"], "evaluation", "report_history.csv")
    if os.path.exists(history_file):
        df.to_csv(history_file, mode="a", header=False, index=False)
    else:
        df.to_csv(history_file, index=False)
    print(f"\n✅ Histórico completo gravado em: {history_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera relatório de progresso do treinamento.")
    parser.add_argument("--target", type=float, default=0.95, help="Acurácia alvo (ex: 0.95).")
    parser.add_argument("--deadline", type=str, default="2025-11-14", help="Prazo final (YYYY-MM-DD).")
    args = parser.parse_args()

    generate_report(target_accuracy=args.target, project_deadline_str=args.deadline)


if __name__ == "__main__":
    main()
