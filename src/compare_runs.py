import os
import string
import spacy
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from src.logging_config import setup_logging
from src.utils import load_config

import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart(metrics_dict, run_names):
    """Plot a radar chart based on the provided metrics."""
    # Define the metrics
    metrics = list(metrics_dict.keys())
    num_metrics = len(metrics)
    num_runs = len(run_names)
    
    sns.set(style="ticks", context="talk")

    # Create values array with the correct shape
    values = np.zeros((num_metrics, num_runs))
    
    # Populate the values array with data from metrics_dict
    for i, metric in enumerate(metrics):
        metric_values = metrics_dict[metric]
        if isinstance(metric_values, dict):
            for j, run_name in enumerate(run_names):
                values[i, j] = metric_values.get(run_name, 0)  # Default to 0 if not available
        else:
            values[i, :] = metric_values

    # Remove the "Thematic Consistency" metric if it's not useful
    if "Thematic Consistency" in metrics:
        thematic_index = metrics.index("Thematic Consistency")
        values = np.delete(values, thematic_index, 0)
        metrics.remove("Thematic Consistency")
        num_metrics -= 1

    # Standardize the values to a range of 0 to 1 for comparability
    min_values = values.min(axis=1, keepdims=True)
    max_values = values.max(axis=1, keepdims=True)
    scaled_values = (values - min_values) / (max_values - min_values)

    # Print the shape for debugging
    print("Scaled Values array shape:", scaled_values.shape)

    # Check the shape of the values array
    if scaled_values.shape[1] != num_runs:
        raise ValueError("Mismatch in number of runs. Expected {}, got {}".format(num_runs, scaled_values.shape[1]))

    # Number of variables
    labels = metrics
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the circle
    angles += angles[:1]

    # Initialize the radar plot
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))

    # Plot each run's data on the radar chart
    for i in range(num_runs):
        run_values = scaled_values[:, i].tolist()
        run_values += run_values[:1]  # Complete the loop
        ax.plot(angles, run_values, linewidth=1, linestyle='solid', label=run_names[i])
        ax.fill(angles, run_values, alpha=0.25)

    # Add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Add padding to prevent label overlap
    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment('center')
        if angle in [0, np.pi]:
            label.set_y(label.get_position()[1] - 0.2)
        elif angle < np.pi:
            label.set_horizontalalignment('left')
            label.set_x(label.get_position()[0] + 0.1)
        else:
            label.set_horizontalalignment('right')
            label.set_x(label.get_position()[0] - 0.1)
    plt.title('Multi-Metric Comparison Across Runs')
    plt.savefig('reports/figures/multi_metric_comparison.png')


def plot_similarity_heatmap(average_similarities, run_names):
    data = []
    for run1 in run_names:
        row = []
        for run2 in run_names:
            row.append(average_similarities[run1][run2] if run1 != run2 else 1.0)  # 1.0 for identical runs
        data.append(row)

    df = pd.DataFrame(data, index=run_names, columns=run_names)

    # Plot the heatmap
    plt.figure(figsize=(14, 12))  # Increased figure size for better label spacing
    sns.set(style="ticks", context="talk")
    ax = sns.heatmap(df, annot=True, cmap="magma", fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})

    # Adjust the layout to ensure labels fit
    plt.title('Average Similarity Across All Long-Form Questions')
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x labels for better fit
    plt.yticks(rotation=0, fontsize=10)  # Keep y labels horizontal
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig('reports/figures/average_similarity_heatmap.png')

def plot_disagreements_per_question(question_disagreement_counts):
    questions = list(question_disagreement_counts.keys())
    num_disagreements = list(question_disagreement_counts.values())

    sns.set(style="ticks", context="talk")
    plt.figure(figsize=(12, 10))
    plt.barh(questions, num_disagreements, color='skyblue')
    plt.xlabel('Number of Disagreements')
    plt.title('Disagreements per Question')
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('reports/figures/disagreements_per_question.png')


def plot_disagreements_per_title(disagreement_report):
    titles = []
    num_disagreements = []
    
    for title, disagreements in disagreement_report.items():
        total_disagreements = sum(disagreements.values())
        titles.append(title)
        num_disagreements.append(total_disagreements)
    
    sns.set(style="ticks", context="talk")
    plt.figure(figsize=(12, 10))
    plt.barh(titles, num_disagreements, color='skyblue')
    plt.xlabel('Number of Disagreements')
    plt.title('Disagreements per Title')
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('reports/figures/disagreements_per_book.png')


# Function to calculate similarity
def calculate_similarity(text1, text2):
    try:
        nlp = spacy.load("en_core_web_md")
    except Exception as e:
        logging.error(f"Error loading Spacy model: {e}")
        return 0.0
    
    if not text1 or not text2:
        logging.error("One or both texts are empty or None.")
        return 0.0

    logging.info("text1: %s", text1)
    logging.info("text2: %s", text2)
    
    try:
        doc1 = nlp(text1)
        doc2 = nlp(text2)
    except Exception as e:
        logging.error(f"Error creating Spacy Doc objects: {e}")
        return 0.0

    if doc1.vector_norm == 0 or doc2.vector_norm == 0:
        logging.error("One or both Doc objects have no vectors.")
        return 0.0
    
    similarity = doc1.similarity(doc2)
    logging.info("Similarity: %s", similarity)
    return similarity

def calculate_metrics(normalized_dataframes, non_categorical_columns):
    """
    Calculate various metrics for each run based on the provided dataframes.

    :param normalized_dataframes: List of pandas DataFrames, one for each run.
    :param non_categorical_columns: List of non-categorical columns to analyze.
    :return: Dictionary containing metrics for each run.
    """
    num_runs = len(normalized_dataframes)
    run_names = [f'run_{i+1}' for i in range(num_runs)]
    metrics_dict = {
        'Sentiment Consistency': [],
        'Lexical Diversity': [],
        'Thematic Consistency': [],
        'Grammatical Consistency': []
    }
    
    for df in normalized_dataframes:
        # Initialize metrics for the current run
        sentiment_consistency = 0
        lexical_diversity = 0
        thematic_consistency = 0
        grammatical_consistency = 0
        num_entries = 0
        col_count = 0

        # Iterate over non-categorical columns for analysis
        for col in non_categorical_columns:
            record_count = 0
            if col in [
                'title',
                'authors',
                'subjects',
                'Using a list of keywords, what is the setting of the story?',
                'List up to 5 main characters in the story.',
                'Using a list, what symbols are used in the story, if any?',
                'Using a list, what motifs are used in the story, if any?',
                'Using a list of keywords what type of language is used? I.e Is the language simple or elegant? Short or long sentences? Straightforward or descriptive? Are words from other languages used frequently?',
                'Using a list of keywords, what sub-genres of science fiction does this story fall under?',
                ' a list of keywords, what sub-genres of fantasy does this story fall under?',
                'Using a list of keywords, what sub-genres of horror does this story fall under?'
            ]:
                pass
            else:
                col_count += 1
                logging.info(f"Column {col_count}: {col}")
                # if col_count > 3:
                #     break
                for response in df[col]:
                    if pd.notna(response):
                        record_count += 1
                        # if record_count > 5:
                        #     break
                        logging.info(f"Response {record_count}: {response}")
                        num_entries += 1
                        
                        # Sentiment Consistency (Dummy Calculation, replace with real sentiment analysis)
                        # Assuming a dummy function sentiment_analysis() returning a score between 0 and 1
                        sentiment_consistency += sentiment_analysis(response)

                        # Lexical Diversity (Type-Token Ratio)
                        words = response.split()
                        num_words = len(words)
                        unique_words = len(set(words))
                        lexical_diversity += unique_words / num_words if num_words > 0 else 0
                        
                        # Grammatical Consistency (Dummy Calculation, replace with real grammatical analysis)
                        # Assuming a dummy function grammatical_analysis() returning a score between 0 and 1
                        grammatical_consistency += grammatical_analysis(response)

        
        # Calculate averages
        if num_entries > 0:
            metrics_dict['Sentiment Consistency'].append(sentiment_consistency / num_entries)
            metrics_dict['Lexical Diversity'].append(lexical_diversity / num_entries)
            metrics_dict['Thematic Consistency'].append(thematic_consistency / num_entries)
            metrics_dict['Grammatical Consistency'].append(grammatical_consistency / num_entries)
        else:
            metrics_dict['Sentiment Consistency'].append(0)
            metrics_dict['Lexical Diversity'].append(0)
            metrics_dict['Thematic Consistency'].append(0)
            metrics_dict['Grammatical Consistency'].append(0)

    
    return metrics_dict, run_names

def sentiment_analysis(text):
    """Perform sentiment analysis using TextBlob."""
    analysis = TextBlob(text)
    logging.info("Sentiment Polarity: %s", analysis.sentiment.polarity)
    return analysis.sentiment.polarity  # Returns a value between -1.0 (negative) and 1.0 (positive)

def extract_unique_themes(df):
    """Extract unique themes from the column containing a list of themes."""
    all_themes = df["Using a list of keywords, what are the major themes of the story?"].dropna().str.split(',')
    unique_themes = set()
    for theme_list in all_themes:
        for theme in theme_list:
            cleaned_theme = theme.strip().lower()
            if cleaned_theme:
                unique_themes.add(cleaned_theme)
    return list(unique_themes)

def grammatical_analysis(text):
    """Analyze grammatical structure consistency."""
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    pos_count = Counter()
    
    for token in doc:
        pos_count[token.pos_] += 1
    
    num_nouns = pos_count['NOUN']
    num_verbs = pos_count['VERB']
    total_tokens = len(doc)
    
    noun_ratio = num_nouns / total_tokens if total_tokens > 0 else 0
    verb_ratio = num_verbs / total_tokens if total_tokens > 0 else 0
    logging.info("Noun / verb ration: %s", noun_ratio / (verb_ratio + 1e-9))
    return noun_ratio / (verb_ratio + 1e-9)  # Simple ratio as a consistency measure


def analyze_disagreements(dataframes, categorical_columns):
    disagreement_report = {}
    question_disagreement_counts = {col: 0 for col in categorical_columns}

    for title in dataframes[0]['title']:
        disagreement_report[title] = {}
        for col in categorical_columns:
            responses = []
            for df in dataframes:
                response = df[df['title'] == title][col].values[0]
                if pd.notna(response) and response not in responses:
                    responses.append(response)
            
            disagreement_count = len(responses) - 1 if len(responses) > 1 else 0
            disagreement_report[title][col] = disagreement_count
            question_disagreement_counts[col] += disagreement_count
    
    return disagreement_report, question_disagreement_counts


def aggregate_average_similarity(similarity_analysis, run_names):
    # Initialize dictionary to store sum and count of similarities
    similarity_sums = {run1: {run2: 0 for run2 in run_names} for run1 in run_names}
    count_sums = {run1: {run2: 0 for run2 in run_names} for run1 in run_names}

    # Aggregate similarity scores
    for col, similarities in similarity_analysis.items():
        for run1, run2, similarity in similarities:
            similarity_sums[run1][run2] += similarity
            count_sums[run1][run2] += 1
            similarity_sums[run2][run1] += similarity
            count_sums[run2][run1] += 1

    # Calculate average similarities
    average_similarities = {
        run1: {
            run2: (similarity_sums[run1][run2] / count_sums[run1][run2] if count_sums[run1][run2] != 0 else 0)
            for run2 in run_names
        }
        for run1 in run_names
    }

    return average_similarities

def analyze_text_similarity(run_names, dataframes, non_categorical_columns):
    similarity_analysis = {col: [] for col in non_categorical_columns}
    col_count = 0

    # Iterate over the columns to compare
    for col in non_categorical_columns:
        col_count += 1
        if col in [
            'title',
            'authors',
            'subjects',
            'Using a list of keywords, what is the setting of the story?',
            'List up to 5 main characters in the story.',
            'Using a list of keywords, what are the major themes of the story?',
            'Using a list, what symbols are used in the story, if any?',
            'Using a list, what motifs are used in the story, if any?',
            'Using a list of keywords what type of language is used? I.e Is the language simple or elegant? Short or long sentences? Straightforward or descriptive? Are words from other languages used frequently?',
            'Using a list of keywords, what sub-genres of science fiction does this story fall under?',
            ' a list of keywords, what sub-genres of fantasy does this story fall under?',
            'Using a list of keywords, what sub-genres of horror does this story fall under?'
        ]:
            pass
        else:
            # Compare each run with every other run
            for i in range(len(run_names)):
                record_count = 0
                for j in range(i + 1, len(run_names)):
                    record_count += 1
                    run1, run2 = run_names[i], run_names[j]
                    df1, df2 = dataframes[i], dataframes[j]

                    # Filter the dataframes to have only common titles
                    common_titles = set(df1['title']).intersection(df2['title'])
                    df1_filtered = df1[df1['title'].isin(common_titles)]
                    df2_filtered = df2[df2['title'].isin(common_titles)]

                    # Ensure the titles are in the same order for comparison
                    df1_filtered = df1_filtered.sort_values(by='title').reset_index(drop=True)
                    df2_filtered = df2_filtered.sort_values(by='title').reset_index(drop=True)

                    similarity_scores = []
                    # Iterate over the rows and calculate similarity
                    for text1, text2 in zip(df1_filtered[col], df2_filtered[col]):
                        if pd.notna(text1) and pd.notna(text2):
                            similarity = calculate_similarity(text1, text2)
                            similarity_scores.append(similarity)

                    # Calculate the average similarity for this pair of runs and this column
                    if similarity_scores:
                        average_similarity = sum(similarity_scores) / len(similarity_scores)
                        similarity_analysis[col].append((run1, run2, average_similarity))
        #             if record_count > 5:
        #                 break
        # if col_count > 3:
        #     break
    return similarity_analysis

def find_single_word_columns(df):
    """Find columns where all records contain only one word."""
    single_word_columns = []

    # Define a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)

    for col in df.columns:
        # Assume column qualifies until proven otherwise
        is_single_word = True
        for value in df[col].dropna():
            # Remove punctuation and extra spaces
            cleaned_value = str(value).translate(translator).strip()
            # Check if the cleaned value is a single word
            if len(cleaned_value.split()) != 1:
                is_single_word = False
                break
        
        if is_single_word:
            single_word_columns.append(col)

    return single_word_columns

def get_non_categorical_columns(df, categorical_columns):
    """Get all columns that are not specified as categorical."""
    # Convert to a set for faster operations
    categorical_set = set(categorical_columns)
    # Subtract categorical columns from the DataFrame's columns
    non_categorical_columns = [col for col in df.columns if col not in categorical_set]
    return non_categorical_columns

def filter_common_books(dataframes):
    """Filter out records that are not present in all dataframes."""
    # There were some books that weren't analyzed in every run due to rate limit issues or other errors.

    common_titles = set(dataframes[0]['title'])

    # Intersect with titles from all other dataframes
    for df in dataframes[1:]:
        common_titles.intersection_update(df['title'])

    # Filter each dataframe to only include common titles
    filtered_dataframes = [df[df['title'].isin(common_titles)] for df in dataframes]

    return filtered_dataframes

# Define a function to normalize text values
def normalize_text_for_all(text):
    ''' Normalize text for all columns, only removes periods and makes lowercase.'''
    if pd.isna(text):
        return text
    text = text.translate(str.maketrans('', '', '.'))
    text = text.lower()
    corrections = {
        "first-person": "first",
        "third-person": "third",
    }
    return corrections.get(text, text)

def compare_runs():
    # Load the configuration
    config = load_config()
    if not config:
        raise Exception("Configuration could not be loaded. Exiting.")

    # Set up logging
    setup_logging(config.get('LOG_FILE', 'project_log.log'), config.get('LOG_LEVEL', 'INFO'))

    # Define the directory containing the runs
    process_dir = 'data/processed'

    # Initialize lists to store dataframes and run names
    dataframes = []
    run_names = []

    # Iterate over each run directory
    for run_dir in os.listdir(process_dir):
        run_path = os.path.join(process_dir, run_dir, 'analysis_results.csv')
        if os.path.isfile(run_path):
            df = pd.read_csv(run_path)
            dataframes.append(df)
            run_names.append(run_dir)

    # Check if we have any data to process
    if not dataframes:
        logging.info("No analysis results found.")
        exit()

    filtered_dataframes = filter_common_books(dataframes)
    for i, df in enumerate(filtered_dataframes):
        filtered_dataframes[i] = df.sort_values(by='title').reset_index(drop=True)

    normalized_dataframes = []
    for df in filtered_dataframes:
        normalized_df = df.copy()  # Copy the dataframe to avoid modifying the original
        for col in normalized_df.columns:
            # Apply normalize_text to each column
            normalized_df[col] = normalized_df[col].apply(normalize_text_for_all)
        normalized_dataframes.append(normalized_df)

    # Concatenate all dataframes for comparison
    all_data = pd.concat(normalized_dataframes, keys=run_names, names=['Run', 'Index'])

    # Identify categorical columns (excluding non-categorical data like plot summaries)
    categorical_columns = find_single_word_columns(all_data)
    logging.debug("Categorical columns: %s", categorical_columns)

    # # Initialize dictionary to store categorical analysis
    categorical_analysis = {col: all_data[col].value_counts() for col in categorical_columns}
    # logging.info("Categorical analysis: %s", categorical_analysis)

    # Compute agreement across runs
    disagreement_report, question_disagreement_counts = analyze_disagreements(normalized_dataframes, categorical_columns)
    logging.info("Disagreement report: %s", disagreement_report)
    logging.info("Question disagreement counts: %s", question_disagreement_counts)

    # Visualize the disagreements per title and question
    plot_disagreements_per_title(disagreement_report)
    plot_disagreements_per_question(question_disagreement_counts)

    # Save categorical analysis
    categorical_df = pd.DataFrame(categorical_analysis)
    categorical_df.to_csv(os.path.join(process_dir, 'categorical_analysis.csv'), index=False)

    # # Define columns for similarity analysis (e.g., plot summaries)
    non_categorical_columns = get_non_categorical_columns(all_data, categorical_columns)
    logging.debug("Non-categorical columns: %s", non_categorical_columns)

    # Analyze similarity for non-categorical columns
    similarity_analysis = analyze_text_similarity(run_names, normalized_dataframes, non_categorical_columns)
    logging.info("Similarity analysis: %s", similarity_analysis)

    #Aggregate average similarity
    average_similarities = aggregate_average_similarity(similarity_analysis, run_names)
    logging.info("Aggregate similarity analysis: %s", average_similarities)

    # Plot similarity heatmap
    plot_similarity_heatmap(average_similarities, run_names)

    # Calculate metrics for each run
    metrics_dict, run_names = calculate_metrics(normalized_dataframes, non_categorical_columns)
    logging.info("Metrics: %s", metrics_dict)

    # Plot spider chart of sentiment, lexical, thematic, and grammatical consistency
    plot_radar_chart(metrics_dict, run_names)

if __name__ == "__main__":
    compare_runs()