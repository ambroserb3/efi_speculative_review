import os
import json, string
import pandas as pd
import logging
from openai import OpenAI
from src.logging_config import setup_logging
def load_config():
    """Load the configuration from a JSON file."""
    try:
        with open('config.json') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

# Load the configuration and set up logging
config = load_config()
if config:
    setup_logging(config.get('LOG_FILE', 'project_log.log'), config.get('LOG_LEVEL', 'INFO'))

MODEL = config.get('OPENAI_MODEL')
if not MODEL:
    raise KeyError("OPENAI_MODEL is not defined in the configuration file.")

def load_prompts(prompts_file):
    """Load prompts from a JSON file."""
    try:
        with open(prompts_file, 'r', encoding='utf-8') as file:
            prompts = json.load(file)['prompts']
        return prompts
    except Exception as e:
        logging.error(f"Error loading prompts: {e}")
        return []
def create_assistant(client):
    """Create a new assistant and return its ID."""
    try:
        assistant = client.beta.assistants.create(
            name="Literature Analysis Assistant",
            instructions="You are a helpful assistant specializing in analyzing literary texts. Please only provide responses that conform to specified formatting and constraints.",
            tools=[{"type": "code_interpreter"}],
            model=MODEL,
        )
        logging.debug(f"Assistant creation response: {assistant}")
        return assistant.id
    except Exception as e:
        logging.error(f"Error creating assistant: {e}")
        return None

def upload_file(file_path, client):
    """Upload file to OpenAI and return the file ID."""
    try:
        with open(file_path, 'rb') as f:
            response = client.files.create(file=f, purpose='assistants')
            logging.debug(f"File upload response: {response}")
            file_id = response.id
        return file_id
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return None

def insert_metadata(prompt, metadata):
    """Insert metadata into the prompt."""
    for key, value in metadata.items():
        placeholder = f"{{{key}}}"
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, str(value))
    return prompt

def apply_constraints(prompt, constraints, metadata=None):
    """Apply constraints to the prompt and insert metadata if provided."""
    if metadata:
        prompt = insert_metadata(prompt, metadata)
    for constraint in constraints:
        if "limit_words" in constraint:
            limit = constraint.split(":")[1]
            if int(limit) > 1:
                prompt = f"{prompt} (You must limit your response to less than {limit} words.)"
            else:
                prompt = f"{prompt} (You must limit your response to a single word.)"
        elif constraint == "yes_no":
            prompt = f"{prompt} (You must answer with either 'yes' or 'no' only. Any other response will be considered invalid.)"
        elif constraint == "gender":
            prompt = f"{prompt} (You must answer with one of the following: 'male', 'female', 'trans', or 'other' only. Any other response will be considered invalid.)"
        elif constraint == "pov":
            prompt = f"{prompt} (You must answer with one of the following: 'first-person', 'third-person' or 'other' only. Any other response will be considered invalid.)"
        elif constraint == "orientation":
            prompt = f"{prompt} (You must answer with one of the following: 'straight', 'gay', 'bisexual', 'asexual', or 'unknown' only. Any other response will be considered invalid.)"
        elif constraint == "class":
            prompt = f"{prompt} (You must answer with one of the following: 'upper-class', 'lower-class', 'middle-class', or 'unknown' only. Any other response will be considered invalid.)"
        elif constraint == "comma_separated_list":
            prompt = f"{prompt} (You must provide the answer as a comma-separated list.)"
    return prompt

def enforce_constraints(response, constraints):
    """Enforce constraints on the response."""
    # Keep commas while stripping other punctuation
    response_cleaned = response.strip().lower().translate(str.maketrans('', '', string.punctuation.replace(',', '')))
    for constraint in constraints:
        if "limit_words" in constraint:
            limit = int(constraint.split(":")[1])
            if len(response_cleaned.split()) > limit:
                logging.warning(f"Response exceeds word limit of {limit} words.")
                return False
        elif constraint == "yes_no":
            if response_cleaned not in ["yes", "no"]:
                logging.warning("Response is not a 'yes' or 'no' answer.")
                return False
        elif constraint == "comma_separated_list":
            if not isinstance(response, str) or ',' not in response:
                logging.warning("Response is not a comma-separated list.")
                return False
    return True

def clean_text(text):
    # Implement text cleaning steps here if necessary
    return text.strip()
