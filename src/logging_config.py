# src/logging_config.py
import logging
import os
import json

def load_config():
    """Load the configuration from a JSON file."""
    try:
        with open('config.json') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

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

# Load the configuration and set up logging
config = load_config()
if config:
    setup_logging(config.get('LOG_FILE', 'project_log.log'), config.get('LOG_LEVEL', 'INFO'))
