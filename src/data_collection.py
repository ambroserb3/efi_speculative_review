import requests
import os
import json
import logging
from requests.exceptions import HTTPError
from src.logging_config import setup_logging
from src.utils import load_config, load_prompts, create_assistant, upload_file, apply_constraints, enforce_constraints

# Load the configuration
config = load_config()
if not config:
    raise Exception("Configuration could not be loaded. Exiting.")

# Extract configuration values
SUBJECTS = config['SUBJECTS']
OUTPUT_DIR_TEXT = config['OUTPUT_DIR_TEXT']
OUTPUT_DIR_METADATA = config['OUTPUT_DIR_METADATA']

# Set up logging
setup_logging(config.get('LOG_FILE'), config.get('LOG_LEVEL'))

GUTENBERG_API_URL = "http://gutendex.com/books"

def search_gutenberg_books(subject, max_results=40):
    """Search Project Gutenberg for public domain books by subject."""
    params = {
        'topic': subject,
        'languages': 'en',
        'limit': max_results
    }
    try:
        response = requests.get(GUTENBERG_API_URL, params=params)
        response.raise_for_status()
        logging.debug(f"API Response: {response.json()}")
        return response.json()
    except HTTPError as http_err:
        logging.error("HTTP error occurred: %s", http_err)
    except Exception as err:
        logging.error("An error occurred: %s", err)
    return {}

def download_book(gutenberg_id, output_dir):
    """Download book content from Project Gutenberg."""
    try:
        book_url = f"http://www.gutenberg.org/ebooks/{gutenberg_id}.txt.utf-8"
        response = requests.get(book_url)
        response.raise_for_status()
        book_text = response.text

        text_file_path = os.path.join(output_dir, f"{gutenberg_id}.txt")
        if os.path.exists(text_file_path):
            logging.info(f"Book {gutenberg_id} already exists. Skipping download.")
            return text_file_path
        
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(book_text)
        return text_file_path
    except Exception as e:
        logging.error(f"Error downloading book {gutenberg_id}: {e}")
        return None

def save_metadata(metadata, output_dir, identifier):
    """Save metadata to a JSON file."""
    try:
        metadata_file_path = os.path.join(output_dir, f"{identifier}.json")
        if os.path.exists(metadata_file_path):
            logging.info(f"Metadata for book {identifier} already exists. Skipping save.")
            return metadata_file_path

        with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        return metadata_file_path
    except Exception as e:
        logging.error(f"Error saving metadata for identifier {identifier}: {e}")
        return None

def log_book_info(book):
    """Log information about a book."""
    title = book.get('title', 'No title available')
    authors = [author['name'] for author in book.get('authors', [{'name': 'No authors available'}])]
    download_count = book.get('download_count', 'No download count available')
    logging.info(f"Title: {title}")
    logging.info(f"Authors: {', '.join(authors)}")
    logging.info(f"Download Count: {download_count}")
    logging.info("-" * 40)

def process_book(book):
    """Process a single book: log info, download text, and save metadata."""
    gutenberg_id = book.get('id')
    
    if gutenberg_id:
        log_book_info(book)
        logging.info(f"Processing book with ID: {gutenberg_id}")
        
        text_file_path = download_book(gutenberg_id, OUTPUT_DIR_TEXT)
        metadata_file_path = save_metadata(book, OUTPUT_DIR_METADATA, gutenberg_id)
        
        if text_file_path and metadata_file_path:
            logging.info(f"Downloaded and saved book: {gutenberg_id}")
            logging.info(f"Text file: {text_file_path}")
            logging.info(f"Metadata file: {metadata_file_path}")
    else:
        logging.warning("No recognizable ID found for this book. Skipping download.")

def run_data_collection():
    if not os.path.exists(OUTPUT_DIR_TEXT):
        os.makedirs(OUTPUT_DIR_TEXT)
    if not os.path.exists(OUTPUT_DIR_METADATA):
        os.makedirs(OUTPUT_DIR_METADATA)
    
    total_books_found = 0
    processed_books = set()

    for subject in SUBJECTS:
        logging.info(f"Searching books for subject: {subject}")
        results = search_gutenberg_books(subject)
        
        books = results.get('results', [])
        total_books_found += len(books)

        for book in books:
            gutenberg_id = book.get('id')
            if gutenberg_id in processed_books:
                logging.info(f"Book {gutenberg_id} already processed. Skipping.")
                continue

            process_book(book)
            processed_books.add(gutenberg_id)
        
        logging.info(f"Total books found for subject '{subject}': {len(books)}")

    logging.info(f"Total books found across all subjects: {total_books_found}")

if __name__ == "__main__":
    run_data_collection()
