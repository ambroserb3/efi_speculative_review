import os
import json
import pandas as pd
import logging
from openai import OpenAI
from src.logging_config import load_config, setup_logging

# Load the configuration
config = load_config()
if not config:
    raise Exception("Configuration could not be loaded. Exiting.")

# Extract configuration values
API_KEY = config.get('OPENAI_API_KEY')
MODEL = config.get('OPENAI_MODEL')
PROCESSED_DATA_DIR = config.get('PROCESSED_DATA_DIR')
OUTPUT_DIR_TEXT = config.get('OUTPUT_DIR_TEXT')
PROMPTS_FILE = "prompts.json"

client = OpenAI(api_key=API_KEY)

if not PROCESSED_DATA_DIR:
    raise KeyError("PROCESSED_DATA_DIR is not defined in the configuration file.")
if not API_KEY:
    raise KeyError("OPENAI_API_KEY is not defined in the configuration file.")
if not MODEL:
    raise KeyError("OPENAI_MODEL is not defined in the configuration file.")
if not OUTPUT_DIR_TEXT:
    raise KeyError("OUTPUT_DIR_TEXT is not defined in the configuration file.")

# Set up logging
setup_logging(config.get('LOG_FILE'), config.get('LOG_LEVEL'))

def load_prompts(prompts_file):
    """Load prompts from a JSON file."""
    try:
        with open(prompts_file, 'r', encoding='utf-8') as file:
            prompts = json.load(file)['prompts']
        return prompts
    except Exception as e:
        logging.error(f"Error loading prompts: {e}")
        return []

def upload_file(file_path):
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

def create_assistant():
    """Create a new assistant and return its ID."""
    try:
        assistant = client.beta.assistants.create(
            name="Literature Analysis Assistant",
            instructions="You are a helpful assistant specializing in analyzing literary texts.",
            tools=[{"type": "code_interpreter"}],
            model=MODEL,
        )
        logging.debug(f"Assistant creation response: {assistant}")
        return assistant.id
    except Exception as e:
        logging.error(f"Error creating assistant: {e}")
        return None

def create_thread(file_id, initial_message):
    """Create a new chat session in OpenAI and return the chat ID."""
    try:
        response = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": initial_message,
                    "attachments": [
                        {
                            "file_id": file_id,
                            "tools": [{"type": "code_interpreter"}]
                        }
                    ]
                }
            ]
        )
        thread_id = response.id
        logging.debug(f"Thread creation response: {response}")
        return thread_id
    except Exception as e:
        logging.error(f"Error creating thread: {e}")
        return None

def call_openai_api(thread_id, message):
    """Call OpenAI API within a given chat session."""
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            for msg in messages:
                if msg.role == 'assistant':
                    content_blocks = msg.content
                    response_text = "".join(block.text.value for block in content_blocks if block.type == "text")
                    logging.debug(f"API call response: {response_text}")
                    return response_text
        else:
            logging.error(f"Run status: {run.status}")
            return None
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None

def apply_constraints(prompt, constraints):
    """Apply constraints to the prompt."""
    for constraint in constraints:
        if "limit_words" in constraint:
            limit = constraint.split(":")[1]
            prompt = f"{prompt} (Please limit your response to {limit} words.)"
    return prompt

def analyze_book(book_metadata, file_id, prompts):
    """Analyze a single book using the loaded prompts."""
    results = {}
    initial_message = f"Here is the full text of the book: {book_metadata.get('title', 'Unknown')}."
    thread_id = create_thread(file_id, initial_message)

    if not thread_id:
        logging.error(f"Failed to create thread for book: {book_metadata.get('title', 'Unknown')}. Skipping.")
        return results

    def recursive_prompt(question, follow_ups, constraints):
        prompt = apply_constraints(question, constraints)
        response = call_openai_api(thread_id, prompt)
        if not response:
            logging.error(f"Failed to get response for prompt: {prompt}")
            return
        results[question] = response
        logging.debug(f"Question: {question}")
        logging.debug(f"Response: {response}")

        # Add the response as a message from assistant
        try:
            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="assistant",
                content=response
            )
        except Exception as e:
            logging.error(f"Error adding assistant message: {e}")

        for condition, follow_up in follow_ups.items():
            if condition in response.lower() and follow_up:
                follow_up_question = follow_up["question"]
                follow_up_follow_ups = follow_up.get("follow_ups", {})
                follow_up_constraints = follow_up.get("constraints", [])
                recursive_prompt(follow_up_question, follow_up_follow_ups, follow_up_constraints)

    for prompt in prompts:
        question = prompt["question"]
        follow_ups = prompt["follow_ups"]
        constraints = prompt.get("constraints", [])

        recursive_prompt(question, follow_ups, constraints)

    return results

def main():
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'books_metadata.csv')

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found at {dataset_path}. Exiting.")
        return

    df = pd.read_csv(dataset_path)
    prompts = load_prompts(PROMPTS_FILE)

    if not prompts:
        logging.error("No prompts loaded. Exiting.")
        return

    results_path = os.path.join(PROCESSED_DATA_DIR, 'analysis_results.csv')

    # Initialize results DataFrame if not exists
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame()

    for _, row in df.iterrows():
        book_metadata = row.to_dict()
        title = book_metadata.get('title', 'Unknown')
        logging.info(f"Analyzing book: {title}")

        # Get the full text file path
        text_filename = book_metadata.get('text_filename')
        text_path = os.path.join(OUTPUT_DIR_TEXT, text_filename)

        if not os.path.exists(text_path):
            logging.error(f"Text file not found for book: {title}. Skipping.")
            continue

        # Upload the file to OpenAI
        file_id = upload_file(text_path)
        if not file_id:
            logging.error(f"Failed to upload file for book: {title}. Skipping.")
            continue

        analysis_result = analyze_book(book_metadata, file_id, prompts)
        analysis_result['title'] = title
        results_df = results_df.append(analysis_result, ignore_index=True)

        # Save results incrementally
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        logging.info(f"Analysis results saved to {results_path}")

if __name__ == "__main__":
    # Create assistant once, outside of the loop
    assistant_id = create_assistant()

    main()


#TODO: Add constraints to the prompts in the prompts.json file, especially yes/no and comma separated lists. 
#TODO: Add more prompts to the prompts.json file for a more comprehensive analysis.
#TODO: Add more logging statements to track the progress of the analysis.
#TODO: Add error handling to gracefully handle exceptions and continue the analysis. Namely, retry if the constraints are not met. Every constraint in Apply_constraints function should be handled by Enforce_constraints.
#TODO: Add a mechanism to resume the analysis from where it left off in case of interruptions.
#TODO: Make sure to save the results incrementally to avoid losing data in case of interruptions.
#TODO: Consider parallelizing the analysis process to speed up the analysis of multiple books.
