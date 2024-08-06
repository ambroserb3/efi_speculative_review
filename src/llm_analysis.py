import os
import json, string
import pandas as pd
import logging
from openai import OpenAI
from src.logging_config import setup_logging
from src.utils import (
    load_config,
    load_prompts,
    create_assistant,
    upload_file,
    apply_constraints,
    enforce_constraints,
    check_token_usage,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

config = load_config()
if not config:
    raise Exception("Configuration could not be loaded. Exiting.")

API_KEY = config.get("OPENAI_API_KEY")
PROCESSED_DATA_DIR = config.get("PROCESSED_DATA_DIR")
OUTPUT_DIR_TEXT = config.get("OUTPUT_DIR_TEXT")
PROMPTS_FILE = "prompts.json"

client = OpenAI(api_key=API_KEY)
assistant_id = create_assistant(client)

if not PROCESSED_DATA_DIR:
    raise KeyError("PROCESSED_DATA_DIR is not defined in the configuration file.")
if not API_KEY:
    raise KeyError("OPENAI_API_KEY is not defined in the configuration file.")
if not OUTPUT_DIR_TEXT:
    raise KeyError("OUTPUT_DIR_TEXT is not defined in the configuration file.")


def create_thread(file_id, initial_message):
    """Create a new chat session in OpenAI and return the chat ID.
    :param file_id: The ID of the uploaded text file.
    :param initial_message: The initial message to send to the API.
    :return: The ID of the chat session
    """
    try:
        response = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": initial_message,
                    "attachments": [
                        {"file_id": file_id, "tools": [{"type": "code_interpreter"}]}
                    ],
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
    """Call OpenAI API within a given chat session.
    :param thread_id: The ID of the chat session.
    :param message: The message to send to the API.
    :return: The response from the API.
    """
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=message, timeout=60
        )
    except Exception as e:
        logging.error(f"Error creating message: {e}")
    try:
        logging.info("Added user message: {}".format(message))
        attempt = 0
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            timeout=60,
        )

        if run.status == "completed":
            logging.debug("Run completed successfully.")

        while run.status != "completed" and attempt < 3:
            logging.error(f"Unexpected run status: {run.status} for message: {message}")
            logging.error(run.usage)
            logging.error(run.last_error.code)
            if run.last_error.code == "rate_limit_exceeded":
                logging.error(f"Rate Limits Exceeded, exiting...")
                quit()
            logging.info(f"Retrying in 10 seconds...")
            time.sleep(10)
            run = client.beta.thread.runs.create_and_poll(
                thread_id=thread_id, assistant_id=assistant_id, timeout=60
            )
            attempt += 1

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            for msg in messages:
                if msg.role == "assistant":
                    content_blocks = msg.content
                    response_text = "".join(
                        block.text.value
                        for block in content_blocks
                        if block.type == "text"
                    )
                    logging.info(f"API call response: {response_text}")
                    logging.info(run.usage)
                    return response_text, run.usage.total_tokens
        else:
            logging.error(f"Run status: {run.status}")
            return None
    except TimeoutError as e:
        logging.error(f"Request timed out: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None


def analyze_book(book_metadata, file_id, prompts, max_workers):
    """Analyze a single book using the loaded prompts.
    :param book_metadata: The metadata for the book.
    :param file_id: The ID of the uploaded text file.
    :param prompts: The list of prompts to use for analysis.
    :param max_workers: The maximum number of workers to use for analysis.
    :return: The analysis results for the book.
    """
    results = {}
    initial_message = (
        f"Here is the full text of the book: {book_metadata.get('title', 'Unknown')}."
    )
    thread_id = create_thread(file_id, initial_message)

    if not thread_id:
        logging.error(
            f"Failed to create thread for book: {book_metadata.get('title', 'Unknown')}. Skipping."
        )
        return results

    max_tokens_per_minute = 20000 / max_workers
    total_tokens_used = 0

    def recursive_prompt(question, follow_ups, constraints, total_tokens_used):
        prompt = apply_constraints(question, constraints, metadata=book_metadata)
        response, tokens_used = call_openai_api(thread_id, prompt)
        total_tokens_used += tokens_used

        if check_token_usage(total_tokens_used, max_tokens_per_minute):
            # Sleep until the next minute to reset the token count
            time.sleep(60)
            total_tokens_used = 0

        if not response:
            logging.error(f"Failed to get response for prompt: {prompt}")
            return total_tokens_used

        attempt = 0
        while not enforce_constraints(response, constraints) and attempt < 3:
            if attempt == 3:
                prompt = f"{prompt} (You have not been answering with the correct format. You MUST use the specified format or constraints to provide a valid response.)"
            logging.info(f"Retrying prompt: {prompt}")
            response = call_openai_api(thread_id, prompt)
            logging.error(f"Failed to get response for prompt: {prompt}")
            attempt += 1

        if not enforce_constraints(response, constraints):
            logging.error(f"Constraints not met for prompt: {prompt}. Skipping.")
            return total_tokens_used

        results[question] = response
        logging.info(f"Question: {question}")
        logging.info(f"Response: {response}")

        try:
            client.beta.threads.messages.create(
                thread_id=thread_id, role="assistant", content=response
            )
            logging.debug("Added assistant message: {}".format(response))
        except Exception as e:
            logging.error(f"Error adding assistant message: {e}")

        for condition, follow_up in follow_ups.items():
            if condition in response.lower() and follow_up:
                if "questions" in follow_up:
                    for follow_up_item in follow_up["questions"]:
                        follow_up_question = follow_up_item["question"]
                        follow_up_follow_ups = follow_up_item.get("follow_ups", {})
                        follow_up_constraints = follow_up_item.get("constraints", [])
                        total_tokens_used = recursive_prompt(
                            follow_up_question,
                            follow_up_follow_ups,
                            follow_up_constraints,
                            total_tokens_used,
                        )
                else:
                    follow_up_question = follow_up["question"]
                    follow_up_follow_ups = follow_up.get("follow_ups", {})
                    follow_up_constraints = follow_up.get("constraints", [])
                    total_tokens_used = recursive_prompt(
                        follow_up_question,
                        follow_up_follow_ups,
                        follow_up_constraints,
                        total_tokens_used,
                    )

        return total_tokens_used

    for prompt in prompts:
        question = prompt["question"]
        follow_ups = prompt["follow_ups"]
        constraints = prompt.get("constraints", [])
        total_tokens_used = recursive_prompt(
            question, follow_ups, constraints, total_tokens_used
        )
        logging.info(f"Total tokens used after request: {total_tokens_used}")

    logging.info(f"Total tokens used: {total_tokens_used}")
    return results


def initialize_analysis_results(prompts, output_path):
    """Initialize the analysis results file with all possible columns.
    :param prompts: The list of prompts to use for analysis.
    :param output_path: The path to the output file.
    """
    columns = ["title", "authors", "subjects"]

    def add_questions(prompt_list):
        for prompt in prompt_list:
            columns.append(prompt["question"])
            for follow_up in prompt.get("follow_ups", {}).values():
                follow_up_questions = follow_up.get("questions", [])
                if follow_up_questions:
                    add_questions(follow_up_questions)

    # Add prompt questions and follow-ups
    add_questions(prompts)

    df = pd.DataFrame(columns=columns)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logging.info(f"Initialized analysis_results.csv with columns: {columns}")


def process_book(row, prompts, results_path, results_df_columns, max_workers):
    """Process a single book using the loaded prompts.
    :param row: The book metadata.
    :param prompts: The list of prompts to use for analysis.
    :param results_path: The path to the results file.
    :param results_df_columns: The columns of the results DataFrame.
    :param max_workers: The maximum number of workers to use for analysis.
    :return: The analysis results for the book.
    """
    book_metadata = row.to_dict()
    title = book_metadata.get("title")
    authors = book_metadata.get("authors", "Unknown")
    subjects = book_metadata.get("subjects", "Unknown")

    logging.info(f"Analyzing book: {title}")

    text_filename = book_metadata.get("text_filename")
    text_path = os.path.join(OUTPUT_DIR_TEXT, text_filename)

    if not os.path.exists(text_path):
        logging.error(f"Text file not found for book: {title}. Skipping.")
        return None

    file_id = upload_file(text_path, client)
    if not file_id:
        logging.error(f"Failed to upload file for book: {title}. Skipping.")
        return None

    analysis_result = analyze_book(book_metadata, file_id, prompts, max_workers)
    analysis_result["title"] = title
    analysis_result["authors"] = authors
    analysis_result["subjects"] = subjects

    analysis_result_filled = {
        col: analysis_result.get(col, pd.NA) for col in results_df_columns
    }
    logging.info(f"Analysis completed for book: {title}")
    return analysis_result_filled


def run_llm_analysis(
    max_books, max_workers, output_path="data/processed/analysis_results.csv"
):
    """Run the LLM analysis on each book in the books dataset and save the results.
    :param max_books: The maximum number of books to analyze.
    :param max_workers: The maximum number of workers to use for analysis.
    :param output_path: The path to save the analysis results.
    """
    dataset_path = os.path.join(PROCESSED_DATA_DIR, "books_metadata.csv")
    results_path = output_path

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found at {dataset_path}. Exiting.")
        return

    df = pd.read_csv(dataset_path)
    prompts = load_prompts(PROMPTS_FILE)

    initialize_analysis_results(prompts, results_path)

    if not prompts:
        logging.error("No prompts loaded. Exiting.")
        return

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=columns)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_book = {}
        for idx, row in enumerate(df.iterrows()):
            if idx >= max_books:
                break
            future = executor.submit(
                process_book,
                row[1],
                prompts,
                results_path,
                results_df.columns,
                max_workers,
            )
            future_to_book[future] = row[1]

        for future in as_completed(future_to_book):
            book_data = future_to_book[future]
            try:
                result = future.result()
                if result:
                    title = result["title"]
                    results_df = results_df[results_df["title"] != title]
                    results_df = pd.concat(
                        [results_df, pd.DataFrame([result])], ignore_index=True
                    )

                    results_df.to_csv(results_path, index=False, encoding="utf-8")
                    logging.info(f"Analysis results saved to {results_path}")
            except Exception as exc:
                logging.error(f"Book analysis generated an exception: {exc}")


if __name__ == "__main__":
    run_llm_analysis()
