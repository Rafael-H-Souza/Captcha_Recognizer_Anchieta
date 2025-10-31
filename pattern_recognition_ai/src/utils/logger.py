import logging
import sys
from typing import Optional

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Retorna um logger configurado para o projeto.
    - name: nome do logger (normalmente __name__ do módulo)
    - level: nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():  # Evita adicionar handlers duplicados
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Handler para saída no console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Opcional: também salvar logs em arquivo
        # file_handler = logging.FileHandler("logs/project.log")
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
    
    return logger
