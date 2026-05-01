import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Creates and returns a configured logger.

    Args:
        name (str): Name of the logger (usually __name__)
        log_dir (str): Directory to store log files

    Returns:
        logging.Logger: Configured logger instance
    """

    # Create logs directory if not exists
    os.makedirs(log_dir, exist_ok=True)

    # Create unique log file per run
    log_file = os.path.join(
        log_dir, f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    logger = logging.getLogger(name)

    # Prevent duplicate logs
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console Handler (important for debugging)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] : %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger