import argparse
import os
import sys
import logging
from datetime import datetime
import pandas as pd
from tabulate import tabulate
from src.config import settings


# ==============================================
# 🧩 CONFIGURAÇÃO DE LOGGING
# ==============================================
LOG_DIR = os.path.join(settings.config["logs_dir"], "reports")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "report_generation.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def generate_report(target_accuracy: float, project_deadline_str: str) -> None:
    log_file = os.path.join(settings.config["logs_dir"], "evaluation", "evaluation_log.csv")

    logger.info("Iniciando geração de relatório de progresso.")
    logger.info(f"Arquivo de log esperado: {log_file}")

    # 1️⃣ Verifica existência
    if not os.path.exists(log_file):
        logger.error("Arquivo 'evaluation_log.csv' não encontrado.")
        print("❌ Arquivo de log 'evaluation_log.csv' não encontrado.")
        print("💡 Execute 'make evaluate' para gerar o log antes do relatório.")
        return

    # 2️⃣ Verifica se está vazio
    if os.path.getsize(log_file) == 0:
        logger.warning("Arquivo 'evaluation_log.csv' está vazio.")
        print("⚠️ O arquivo 'evaluation_log.csv' está vazio.")
        print("💡 Execute 'make evaluate' para gerar as métricas primeiro.")
        return

    # 3️⃣ Leitura segura
    try:
        df = pd.read_csv(log_file)
        logger.info(f"{len(df)} registros carregados do log.")
    except pd.errors.EmptyDataError:
        logger.error("Nenhum dado válido encontrado no CSV.")
        print("⚠️ Nenhum dado válido encontrado em 'evaluation_log.csv'.")
        return
    except Exception as e:
        logger.exception(f"Erro ao ler o CSV: {e}")
        print(f"❌ Erro ao ler '{log_file}': {e}")
        return

    # 4️⃣ Colunas obrigatórias
    required_cols = {"timestamp", "accuracy", "loss", "image_count"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Colunas ausentes: {missing_cols}")
        print(f"⚠️ Arquivo incompleto. Faltando colunas: {', '.join(missing_cols)}")
        return

    # 5️⃣ Conversões e cálculos
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["type"] = df.get("type", "N/A")
    df["subtype"] = df.get("subtype", "N/A")
    df["progress_percentage"] = (df["accuracy"] / target_accuracy * 100).round(2)

    # 6️⃣ Prazo e status
    try:
        project_deadline = datetime.strptime(project_deadline_str, "%Y-%m-%d")
        df["days_remaining"] = (project_deadline - df["timestamp"]).dt.days
        df["required_daily_gain"] = df.apply(
            lambda row: ((target_accuracy - row["accuracy"]) / row["days_remaining"])
            if row["days_remaining"] > 0 else 0,
            axis=1
        )
        df["status"] = df["days_remaining"].apply(
            lambda x: "Dentro do Prazo" if x > 0 else "Atrasado"
        )
    except ValueError:
        logger.warning("Prazo inválido. Use o formato YYYY-MM-DD.")
        project_deadline = None
        df["days_remaining"] = None
        df["required_daily_gain"] = None
        df["status"] = "Prazo Inválido"

    # 7️⃣ Última avaliação
    latest_eval = df.sort_values(by="timestamp", ascending=False).iloc[0]

    # ===============================================
    # 🧾 EXIBIÇÃO DO RELATÓRIO
    # ===============================================
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE PROGRESSO DO MODELO DE IA")
    print("=" * 60)
    print(f"Última Avaliação: {latest_eval['timestamp'].strftime('%d/%m/%Y %H:%M')}")
    print("-" * 60)
    print("🎯 DESEMPENHO ATUAL")
    print(f"  - Acurácia Atual:         {latest_eval['accuracy']:.2%}")
    print(f"  - Acurácia Alvo:          {target_accuracy:.2%}")
    print(f"  - Progresso para o Alvo:  {latest_eval['progress_percentage']:.2f}%")
    print(f"  - Loss Atual:             {latest_eval['loss']:.4f}")
    print(f"  - Imagens Processadas:    {int(latest_eval['image_count'])}")

    print("\n🗓️ ACOMPANHAMENTO DE PRAZOS")
    if project_deadline is None:
        print("  - ❌ Prazo inválido informado.")
    else:
        print(f"  - Prazo Final:            {project_deadline.strftime('%d/%m/%Y')}")
        print(f"  - Dias Restantes:         {int(latest_eval['days_remaining'])}")
        print(f"  - Ganho Diário Necessário: {latest_eval['required_daily_gain']:.3%}")
        print(f"  - Status:                 {latest_eval['status']}")

    # 8️⃣ Agrupamento
    grouped = df.groupby(["type", "subtype"], dropna=False).agg(
        total_images=("image_count", "sum"),
        avg_accuracy=("accuracy", "mean"),
        avg_loss=("loss", "mean"),
        avg_progress=("progress_percentage", "mean")
    ).reset_index()

    print("\n📌 MÉTRICAS POR TIPO E SUBTIPO")
    print(tabulate(
        grouped,
        headers=["Tipo", "Subtipo", "Total Imagens", "Acurácia Média", "Loss Média", "Progresso Médio (%)"],
        tablefmt="fancy_grid",
        floatfmt=(".0f", ".0f", ".4f", ".4f", ".2f")
    ))

    # 9️⃣ Média total
    total_images = df["image_count"].sum()
    overall_accuracy = df["accuracy"].mean()
    overall_loss = df["loss"].mean()
    overall_progress = df["progress_percentage"].mean()

    print("\n📊 MÉTRICAS GERAIS")
    print(tabulate([[
        total_images, overall_accuracy, overall_loss, overall_progress
    ]],
        headers=["Total Imagens", "Acurácia Média", "Loss Média", "Progresso Médio (%)"],
        tablefmt="fancy_grid",
        floatfmt=(".0f", ".2%", ".4f", ".2f")
    ))

    # 🔟 Salva histórico completo
    history_dir = os.path.join(settings.config["logs_dir"], "evaluation")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, "report_history.csv")

    try:
        write_mode = "a" if os.path.exists(history_file) else "w"
        header_flag = not os.path.exists(history_file)
        df.to_csv(history_file, mode=write_mode, header=header_flag, index=False)
        logger.info(f"Histórico salvo: {history_file}")
        print(f"\n✅ Histórico salvo em: {history_file}")
    except Exception as e:
        logger.exception(f"Erro ao salvar histórico: {e}")
        print(f"⚠️ Erro ao salvar histórico: {e}")

    logger.info("✅ Relatório gerado com sucesso.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera relatório de progresso do modelo.")
    parser.add_argument("--target", type=float, default=0.98, help="Acurácia alvo (ex: 0.98)")
    parser.add_argument("--deadline", type=str, default="2026-12-31", help="Prazo final (YYYY-MM-DD)")
    args = parser.parse_args()

    generate_report(target_accuracy=args.target, project_deadline_str=args.deadline)


if __name__ == "__main__":
    main()
