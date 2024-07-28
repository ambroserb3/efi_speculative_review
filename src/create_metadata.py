import os
import json
import pandas as pd
import logging
from src.logging_config import load_config, setup_logging

# Load the configuration
config = load_config()
if not config:
    raise Exception("Configuration could not be loaded. Exiting.")

# Extract configuration values
OUTPUT_DIR_TEXT = config['OUTPUT_DIR_TEXT']
OUTPUT_DIR_METADATA = config['OUTPUT_DIR_METADATA']
PROCESSED_DATA_DIR = "data/processed"

# Set up logging
setup_logging(config.get('LOG_FILE'), config.get('LOG_LEVEL'))

def create_dataframe(metadata_dir, text_dir):
    """Create a DataFrame from metadata files and corresponding text filenames."""
    data = []
    for filename in os.listdir(metadata_dir):
        if filename.endswith('.json'):
            metadata_path = os.path.join(metadata_dir, filename)
            with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                metadata = json.load(metadata_file)
                book_id = os.path.splitext(filename)[0]
                text_filename = f"{book_id}.txt"
                text_path = os.path.join(text_dir, text_filename)

                if os.path.exists(text_path):
                    metadata['text_filename'] = text_filename
                    data.append(metadata)
                else:
                    logging.warning(f"Text file for {book_id} not found. Skipping.")

    df = pd.DataFrame(data)
    return df

def main():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    df = create_dataframe(OUTPUT_DIR_METADATA, OUTPUT_DIR_TEXT)
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'books_metadata.csv')
    df.to_csv(dataset_path, index=False, encoding='utf-8')
    logging.info(f"Metadata dataset saved to {dataset_path}")

if __name__ == "__main__":
    main()
