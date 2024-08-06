#src/main.py
import logging
import os
from src.logging_config import setup_logging
from src.utils import load_config, create_assistant
from src.data_collection import run_data_collection
from src.llm_analysis import run_llm_analysis
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def run_analysis_iteration(max_books, max_workers, output_path=None):
    try:
        logging.info("Starting LLM analysis...")
        run_llm_analysis(max_books, max_workers, output_path=output_path)
        logging.info("LLM analysis completed successfully.")
    except Exception as e:
        logging.error(f"LLM analysis failed: {e}")
        raise

def main():
    # Load the configuration
    config = load_config()
    if not config:
        raise Exception("Configuration could not be loaded. Exiting.")

    # Set up logging
    setup_logging(config.get('LOG_FILE', 'project_log.log'), config.get('LOG_LEVEL', 'INFO'))

    iterations = config.get('iterations', 1)
    max_books = config.get('max_books', 3)
    max_workers = config.get('max_workers', 4)
    try:
        # data_collection()
        with ThreadPoolExecutor(max_workers=1) as executor: #Forcing 1 iteration per time due to rate limits
            futures = []
            for i in range(iterations):
                logging.info(f"Starting iteration {i + 1}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_directory = f"data/processed/run_{i+1}_{timestamp}"
                os.makedirs(run_directory, exist_ok=True)
                output_path = os.path.join(run_directory, "analysis_results.csv")

                future = executor.submit(run_analysis_iteration, max_books, max_workers, output_path=output_path)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                    logging.info(f"Iteration completed successfully.")
                except Exception as exc:
                    logging.error(f"Iteration generated an exception: {exc}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
