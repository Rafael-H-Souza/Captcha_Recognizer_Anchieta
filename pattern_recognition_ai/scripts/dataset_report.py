import os
from pathlib import Path
from src.config import settings
from src.utils.logger import get_logger
from rich.console import Console
from rich.table import Table

def generate_dataset_report():
    logger = get_logger("Dataset-Report")
    console = Console()
    processed_dir = Path(settings.config["processed_data_dir"])
    
    if not processed_dir.exists():
        logger.error("⚠️ Diretório processed não encontrado: %s", processed_dir)
        return

    types = ["single_char", "five_char"]
    subtypes = ["numb", "world", "alphanumeric", "alphanumeric_case"]

    total_images = 0
    report_data = []

    # Coleta os dados
    for t in types:
        for st in subtypes:
            folder = processed_dir / t / st
            if not folder.exists():
                logger.warning("Pasta não encontrada: %s", folder)
                count = 0
            else:
                count = len([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            
            report_data.append((t, st, count))
            total_images += count

    # Cria a tabela com rich
    table = Table(title="📊 Relatório do Dataset (Processed)", show_footer=True)
    table.add_column("Tipo", style="cyan", no_wrap=True)
    table.add_column("Subtipo", style="magenta", no_wrap=True)
    table.add_column("Imagens", justify="right", style="green", footer=str(total_images))

    for t, st, count in report_data:
        table.add_row(t, st, str(count))

    console.print(table)

if __name__ == "__main__":
    generate_dataset_report()
