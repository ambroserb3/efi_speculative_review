import os
import json, string
import pandas as pd
import logging
from openai import OpenAI
from src.logging_config import setup_logging
from src.utils import load_config, load_prompts, create_assistant, upload_file, apply_constraints, enforce_constraints

config = load_config()
if not config:
    raise Exception("Configuration could not be loaded. Exiting.")

API_KEY = config.get('OPENAI_API_KEY')
PROCESSED_DATA_DIR = config.get('PROCESSED_DATA_DIR')
OUTPUT_DIR_TEXT = config.get('OUTPUT_DIR_TEXT')
PROMPTS_FILE = "prompts.json"

client = OpenAI(api_key=API_KEY)

if not PROCESSED_DATA_DIR:
    raise KeyError("PROCESSED_DATA_DIR is not defined in the configuration file.")
if not API_KEY:
    raise KeyError("OPENAI_API_KEY is not defined in the configuration file.")
if not OUTPUT_DIR_TEXT:
    raise KeyError("OUTPUT_DIR_TEXT is not defined in the configuration file.")

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
        logging.info("Added user message: {}".format(message))
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
                    logging.info(f"API call response: {response_text}")
                    return response_text
        else:
            logging.error(f"Run status: {run.status}")
            return None
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None

def analyze_book(book_metadata, file_id, prompts):
    """Analyze a single book using the loaded prompts."""
    results = {}
    initial_message = f"Here is the full text of the book: {book_metadata.get('title', 'Unknown')}."
    thread_id = create_thread(file_id, initial_message)

    if not thread_id:
        logging.error(f"Failed to create thread for book: {book_metadata.get('title', 'Unknown')}. Skipping.")
        return results

    def recursive_prompt(question, follow_ups, constraints):
        prompt = apply_constraints(question, constraints, metadata=book_metadata)
        response = call_openai_api(thread_id, prompt)
        if not response:
            logging.error(f"Failed to get response for prompt: {prompt}")
            return

        # Retry mechanism
        attempt = 0
        while not enforce_constraints(response, constraints) and attempt < 3:
            logging.debug(f"Retrying prompt: {prompt}")
            response = call_openai_api(thread_id, prompt)
            attempt += 1

        if not enforce_constraints(response, constraints):
            logging.error(f"Constraints not met for prompt: {prompt}. Skipping.")
            return

        results[question] = response
        logging.info(f"Question: {question}")
        logging.info(f"Response: {response}")

        # Add the response as a message from assistant
        try:
            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="assistant",
                content=response
            )
            logging.info("Added assistant message: {}".format(response))
        except Exception as e:
            logging.error(f"Error adding assistant message: {e}")

        for condition, follow_up in follow_ups.items():
            if condition in response.lower() and follow_up:
                if "questions" in follow_up:
                    for follow_up_item in follow_up["questions"]:
                        follow_up_question = follow_up_item["question"]
                        follow_up_follow_ups = follow_up_item.get("follow_ups", {})
                        follow_up_constraints = follow_up_item.get("constraints", [])
                        recursive_prompt(follow_up_question, follow_up_follow_ups, follow_up_constraints)
                else:
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

def run_llm_analysis():
    """Run the LLM analysis on each book in the books dataset and save the results."""
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'books_metadata.csv')
    results_path = os.path.join(PROCESSED_DATA_DIR, 'analysis_results.csv')

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found at {dataset_path}. Exiting.")
        return

    df = pd.read_csv(dataset_path)
    prompts = load_prompts(PROMPTS_FILE)

    if not prompts:
        logging.error("No prompts loaded. Exiting.")
        return

    # Extract columns from prompts.json
    columns = ['title', 'authors', 'subjects']  # Start with 'title' column
    for prompt in prompts:
        columns.append(prompt['question'])
        for follow_up in prompt.get('follow_ups', {}).values():
            if 'question' in follow_up:
                columns.append(follow_up['question'])
                for sub_follow_up in follow_up.get('follow_ups', {}).values():
                    if 'question' in sub_follow_up:
                        columns.append(sub_follow_up['question'])

    # Initialize results DataFrame if not exists
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=columns)

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
        file_id = upload_file(text_path, client)
        if not file_id:
            logging.error(f"Failed to upload file for book: {title}. Skipping.")
            continue

        analysis_result = analyze_book(book_metadata, file_id, prompts)
        analysis_result['title'] = title
        analysis_result['authors'] = book_metadata.get('authors', 'Unknown')
        analysis_result['subjects'] = book_metadata.get('subjects', 'Unknown')

        # Update or add the entry for the book
        results_df = results_df[results_df['title'] != title]  # Remove any existing entry for the book
        results_df = pd.concat([results_df, pd.DataFrame([analysis_result])], ignore_index=True)

        # Save results incrementally without duplicates
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        logging.info(f"Analysis results saved to {results_path}")
        break

if __name__ == "__main__":
    # Create assistant once, outside of the loop
    assistant_id = create_assistant(client)
    run_llm_analysis()
    #TODO: Consider parallelizing the analysis process to speed up the analysis of multiple books.
