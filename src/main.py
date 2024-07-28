#src/main.py
import logging
from src.logging_config import setup_logging
from src.utils import load_config
from src.data_collection import run_data_collection
from src.llm_analysis import run_llm_analysis

config = load_config()
if not config:
    raise Exception("Configuration could not be loaded. Exiting.")

setup_logging(config.get('LOG_FILE'), config.get('LOG_LEVEL'))

def data_collection():
    try:
        logging.info("Starting data collection...")
        run_data_collection()
        logging.info("Data collection completed successfully.")
    except Exception as e:
        logging.error(f"Data collection failed: {e}")
        raise

def llm_analysis():
    try:
        logging.info("Starting LLM analysis...")
        run_llm_analysis()
        logging.info("LLM analysis completed successfully.")
    except Exception as e:
        logging.error(f"LLM analysis failed: {e}")
        raise

def main():
    try:
        data_collection()
        llm_analysis()
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
