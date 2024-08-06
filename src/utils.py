import os
import json, string
import pandas as pd
import logging
from openai import OpenAI
from src.logging_config import setup_logging


def load_config():
    """Load the configuration from a JSON file.
    :return: A dictionary containing the configuration settings.
    """
    try:
        with open("config.json") as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


config = load_config()
if config:
    setup_logging(
        config.get("LOG_FILE", "project_log.log"), config.get("LOG_LEVEL", "INFO")
    )

MODEL = config.get("OPENAI_MODEL")
TEMPERATURE = config.get("temperature", 1)
TOP_P = config.get("top_p", 1)

if not MODEL:
    raise KeyError("OPENAI_MODEL is not defined in the configuration file.")


def load_prompts(prompts_file):
    """Load prompts from a JSON file.
    :param prompts_file: The path to the JSON file containing prompts.
    """
    try:
        with open(prompts_file, "r", encoding="utf-8") as file:
            prompts = json.load(file)["prompts"]
        return prompts
    except Exception as e:
        logging.error(f"Error loading prompts: {e}")
        return []


def create_assistant(client):
    """Create a new assistant and return its ID.
    :param client: The OpenAI client instance.
    """
    try:
        assistant = client.beta.assistants.create(
            name="Literature Analysis Assistant",
            instructions="You are a scholar specializing in analyzing literary texts. Please only provide responses which conform to specified formatting and constraints.",
            tools=[{"type": "code_interpreter"}],
            model=MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        logging.debug(f"Assistant creation response: {assistant}")
        return assistant.id
    except Exception as e:
        logging.error(f"Error creating assistant: {e}")
        return None


def upload_file(file_path, client):
    """Upload file to OpenAI and return the file ID.
    :param file_path: The path to the file to upload.
    :param client: The OpenAI client instance.
    """
    try:
        with open(file_path, "rb") as f:
            response = client.files.create(file=f, purpose="assistants")
            logging.debug(f"File upload response: {response}")
            file_id = response.id
        return file_id
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return None


def insert_metadata(prompt, metadata):
    """Insert metadata into the prompt.
    :param prompt: The prompt string.
    :param metadata: A dictionary containing metadata values.
    :return: The prompt with metadata values inserted.
    """
    for key, value in metadata.items():
        placeholder = f"{{{key}}}"
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, str(value))
    return prompt


def apply_constraints(prompt, constraints, metadata=None):
    """Apply constraints to the prompt and insert metadata if provided.
    :param prompt: The prompt string.
    :param constraints: A list of constraints to apply.
    :param metadata: A dictionary containing metadata values.
    :return: The prompt with constraints applied.
    """
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
            prompt = f"{prompt} (You must answer with either: 'male' or 'female' only. Any other response will be considered invalid.)"
        elif constraint == "pov":
            prompt = f"{prompt} (You must answer with one of the following: 'first-person', 'third-person', or 'other' only. Any other response will be considered invalid.)"
        elif constraint == "orientation":
            prompt = f"{prompt} (You must answer with one of the following: 'straight', 'gay', 'bisexual', 'asexual', or 'unknown' only. Any other response will be considered invalid.)"
        elif constraint == "class":
            prompt = f"{prompt} (You must answer with one of the following: 'upperclass', 'lowerclass', 'middleclass', or 'unknown' only. Any other response will be considered invalid.)"
        elif constraint == "comma_separated_list":
            prompt = (
                f"{prompt} (You must provide the answer as a comma-separated list.)"
            )
    return prompt


def enforce_constraints(response, constraints):
    """Enforce constraints on the response.
    :param response: The response to check.
    :param constraints: A list of constraints to enforce.
    :return: True if the response meets all constraints, False otherwise.
    """
    normalized_response = (
        response.strip().lower().translate(str.maketrans("", "", string.punctuation))
    )
    for constraint in constraints:
        if "limit_words" in constraint:
            limit = int(constraint.split(":")[1])
            if len(normalized_response.split()) > limit:
                logging.info(f"Response exceeds word limit of {limit} words.")
                return False
        elif constraint == "yes_no":
            if normalized_response not in ["yes", "no"]:
                logging.warning("Response is not a 'yes' or 'no' answer.")
                return False
        elif constraint == "gender":
            if normalized_response not in ["male", "female"]:
                logging.warning("Response is not a valid gender answer.")
                return False
        elif constraint == "pov":
            if normalized_response not in ["first-person", "third-person", "other"]:
                logging.warning("Response is not a valid point of view answer.")
                return False
        elif constraint == "orientation":
            if normalized_response not in [
                "straight",
                "gay",
                "bisexual",
                "asexual",
                "unknown",
            ]:
                logging.warning("Response is not a valid orientation answer.")
                return False
        elif constraint == "class":
            if normalized_response not in [
                "upperclass",
                "lowerclass",
                "middleclass",
                "unknown",
            ]:
                logging.warning("Response is not a valid class answer.")
                return False
        elif constraint == "comma_separated_list":
            if not isinstance(response, str) or "," not in response:
                logging.warning("Response is not a comma-separated list.")
                return False
    return True


def check_token_usage(current_usage, max_usage):
    """Check if the current token usage exceeds the maximum allowed.
    :param current_usage: The current token usage.
    :param max_usage: The maximum token usage allowed.
    :return: True if the current usage exceeds the maximum, False otherwise.
    """
    if current_usage >= max_usage:
        return True
    return False
