# src/logging_config.py
import logging
import os
import json

def setup_logging(log_file, log_level):
    """Set up logging configuration."""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join('logs', log_file)),
            logging.StreamHandler()
        ]
    )

    # Set specific logging levels for libraries
    httpx_logger = logging.getLogger("httpx")
    openai_logger = logging.getLogger("openai")

    httpx_logger.setLevel(logging.WARNING)
    openai_logger.setLevel(logging.WARNING)

    # Optional: silence any other noisy loggers
    logging.getLogger("httpx._api").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)