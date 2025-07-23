# src/logger.py
import logging
from pathlib import Path

def setup_logging(log_dir: str = "logs", logger_name: str = "app") -> logging.Logger:
    """
    Centralized logging configuration for all modules.
    
    Args:
        log_dir: Directory to store log files
        logger_name: Identifier for the logger
        
    Returns:
        Configured logger instance
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # File handler (debug level)
    file_handler = logging.FileHandler(log_dir_path / f'{logger_name}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler (info level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Clear existing handlers (if any)
    logger.handlers.clear()
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger