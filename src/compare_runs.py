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


def plot_radar_chart(metrics_dict, run_names):
    """Plot a radar chart based on the provided metrics.
    :param metrics_dict (dict): A dictionary containing metrics for each run.
    :param run_names (list): A list of run names to use.
    """
    metrics = list(metrics_dict.keys())
    num_metrics = len(metrics)
    num_runs = len(run_names)

    sns.set(style="ticks", context="talk")

    values = np.zeros((num_metrics, num_runs))

    for i, metric in enumerate(metrics):
        metric_values = metrics_dict[metric]
        if isinstance(metric_values, dict):
            for j, run_name in enumerate(run_names):
                values[i, j] = metric_values.get(run_name, 0)
        else:
            values[i, :] = metric_values

    ### Thematic Consistency is not included in the radar chart this code may be removed
    if "Thematic Consistency" in metrics:
        thematic_index = metrics.index("Thematic Consistency")
        values = np.delete(values, thematic_index, 0)
        metrics.remove("Thematic Consistency")
        num_metrics -= 1

    min_values = values.min(axis=1, keepdims=True)
    max_values = values.max(axis=1, keepdims=True)
    scaled_values = (values - min_values) / (max_values - min_values)

    print("Scaled Values array shape:", scaled_values.shape)

    if scaled_values.shape[1] != num_runs:
        raise ValueError(
            "Mismatch in number of runs. Expected {}, got {}".format(
                num_runs, scaled_values.shape[1]
            )
        )

    labels = metrics
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))

    for i in range(num_runs):
        run_values = scaled_values[:, i].tolist()
        run_values += run_values[:1]  # Complete the loop
        ax.plot(angles, run_values, linewidth=1, linestyle="solid", label=run_names[i])
        ax.fill(angles, run_values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment("center")
        if angle in [0, np.pi]:
            label.set_y(label.get_position()[1] - 0.2)
        elif angle < np.pi:
            label.set_horizontalalignment("left")
            label.set_x(label.get_position()[0] + 0.1)
        else:
            label.set_horizontalalignment("right")
            label.set_x(label.get_position()[0] - 0.1)
    plt.title("Multi-Metric Comparison Across Runs")
    plt.savefig("reports/figures/multi_metric_comparison.png")


def plot_similarity_heatmap(average_similarities, run_names):
    """Plot a heatmap of average similarities between runs.
    :param average_similarities (dict): A dictionary containing average similarities between runs.
    :param run_names (list): A list of run names to use
    """
    data = []
    for run1 in run_names:
        row = []
        for run2 in run_names:
            row.append(
                average_similarities[run1][run2] if run1 != run2 else 1.0
            )  # 1.0 for identical runs
        data.append(row)

    df = pd.DataFrame(data, index=run_names, columns=run_names)

    plt.figure(figsize=(14, 12))
    sns.set(style="ticks", context="talk")
    ax = sns.heatmap(
        df,
        annot=True,
        cmap="magma",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Average Similarity Across All Long-Form Questions")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/figures/average_similarity_heatmap.png")


def plot_disagreements_per_question(question_disagreement_counts):
    """Plot a bar chart of disagreements per question.
    :param question_disagreement_counts (dict): A dictionary containing the number of disagreements per question.
    """
    questions = list(question_disagreement_counts.keys())
    num_disagreements = list(question_disagreement_counts.values())

    sns.set(style="ticks", context="talk")
    plt.figure(figsize=(12, 10))
    plt.barh(questions, num_disagreements, color="skyblue")
    plt.xlabel("Number of Disagreements")
    plt.title("Disagreements per Question")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/figures/disagreements_per_question.png")


def plot_disagreements_per_title(disagreement_report):
    """Plot a bar chart of disagreements per title.
    :param disagreement_report (dict): A dictionary containing the number of disagreements per title.
    """
    titles = []
    num_disagreements = []

    for title, disagreements in disagreement_report.items():
        total_disagreements = sum(disagreements.values())
        titles.append(title)
        num_disagreements.append(total_disagreements)

    sns.set(style="ticks", context="talk")
    plt.figure(figsize=(12, 10))
    plt.barh(titles, num_disagreements, color="skyblue")
    plt.xlabel("Number of Disagreements")
    plt.title("Disagreements per Title")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/figures/disagreements_per_book.png")


def calculate_similarity(text1, text2):
    """Calculate the similarity between two texts using Spacy's word vectors.
    :param text1 (str): The first text to compare.
    :param text2 (str): The second text to compare.
    :return float: The similarity score between 0 and 1.
    """
    try:
        nlp = spacy.load("en_core_web_md")
    except Exception as e:
        logging.error(f"Error loading Spacy model: {e}")
        return 0.0

    if not text1 or not text2:
        logging.error("One or both texts are empty or None.")
        return 0.0

    logging.debug("text1: %s", text1)
    logging.debug("text2: %s", text2)

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
    logging.debug("Similarity: %s", similarity)
    return similarity


def calculate_metrics(normalized_dataframes, non_categorical_columns):
    """
    Calculate various metrics for each run based on the provided dataframes.

    :param normalized_dataframes: List of pandas DataFrames, one for each run.
    :param non_categorical_columns: List of non-categorical columns to analyze.
    :return: Dictionary containing metrics for each run.
    """
    num_runs = len(normalized_dataframes)
    run_names = [f"run_{i+1}" for i in range(num_runs)]
    metrics_dict = {
        "Sentiment Consistency": [],
        "Lexical Diversity": [],
        "Thematic Consistency": [],
        "Grammatical Consistency": [],
    }

    for df in normalized_dataframes:
        sentiment_consistency = 0
        lexical_diversity = 0
        thematic_consistency = 0
        grammatical_consistency = 0
        num_entries = 0

        for col in non_categorical_columns:
            if col in [
                "title",
                "authors",
                "subjects",
                "Using a list of keywords, what is the setting of the story?",
                "List up to 5 main characters in the story.",
                "Using a list, what symbols are used in the story, if any?",
                "Using a list, what motifs are used in the story, if any?",
                "Using a list of keywords what type of language is used? I.e Is the language simple or elegant? Short or long sentences? Straightforward or descriptive? Are words from other languages used frequently?",
                "Using a list of keywords, what sub-genres of science fiction does this story fall under?",
                " a list of keywords, what sub-genres of fantasy does this story fall under?",
                "Using a list of keywords, what sub-genres of horror does this story fall under?",
            ]:
                pass
            else:
                for response in df[col]:
                    if pd.notna(response):
                        logging.debug(f"Response {record_count}: {response}")
                        num_entries += 1
                        sentiment_consistency += sentiment_analysis(response)
                        words = response.split()
                        num_words = len(words)
                        unique_words = len(set(words))
                        lexical_diversity += (
                            unique_words / num_words if num_words > 0 else 0
                        )
                        grammatical_consistency += grammatical_analysis(response)

        if num_entries > 0:
            metrics_dict["Sentiment Consistency"].append(
                sentiment_consistency / num_entries
            )
            metrics_dict["Lexical Diversity"].append(lexical_diversity / num_entries)
            metrics_dict["Thematic Consistency"].append(
                thematic_consistency / num_entries
            )
            metrics_dict["Grammatical Consistency"].append(
                grammatical_consistency / num_entries
            )
        else:
            metrics_dict["Sentiment Consistency"].append(0)
            metrics_dict["Lexical Diversity"].append(0)
            metrics_dict["Thematic Consistency"].append(0)
            metrics_dict["Grammatical Consistency"].append(0)

    return metrics_dict, run_names


def sentiment_analysis(text):
    """Perform sentiment analysis using TextBlob.
    :param: text (str): The text to analyze.
    return: float: The sentiment polarity score.
    """
    analysis = TextBlob(text)
    logging.debug("Sentiment Polarity: %s", analysis.sentiment.polarity)
    return analysis.sentiment.polarity


def extract_unique_themes(df):
    """Extract unique themes from the column containing a list of themes.
    :param df (pd.DataFrame): The DataFrame containing the data.
    :return list: A list of unique themes.
    """
    all_themes = (
        df["Using a list of keywords, what are the major themes of the story?"]
        .dropna()
        .str.split(",")
    )
    unique_themes = set()
    for theme_list in all_themes:
        for theme in theme_list:
            cleaned_theme = theme.strip().lower()
            if cleaned_theme:
                unique_themes.add(cleaned_theme)
    return list(unique_themes)


def grammatical_analysis(text):
    """Analyze grammatical structure consistency.
    :param text (str): The text to analyze.
    :return float: The noun / verb ratio.
    """
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    pos_count = Counter()

    for token in doc:
        pos_count[token.pos_] += 1

    num_nouns = pos_count["NOUN"]
    num_verbs = pos_count["VERB"]
    total_tokens = len(doc)

    noun_ratio = num_nouns / total_tokens if total_tokens > 0 else 0
    verb_ratio = num_verbs / total_tokens if total_tokens > 0 else 0
    logging.debug("Noun / verb ration: %s", noun_ratio / (verb_ratio + 1e-9))
    return noun_ratio / (verb_ratio + 1e-9)  # Simple ratio as a consistency measure


def analyze_disagreements(dataframes, categorical_columns):
    """
    Analyze disagreements between runs for categorical columns.
    :param dataframes: List of pandas DataFrames, one for each run.
    :param categorical_columns: List of categorical columns to analyze.
    :return: Tuple containing a disagreement report and question disagreement counts.
    """
    disagreement_report = {}
    question_disagreement_counts = {col: 0 for col in categorical_columns}

    for title in dataframes[0]["title"]:
        disagreement_report[title] = {}
        for col in categorical_columns:
            responses = []
            for df in dataframes:
                response = df[df["title"] == title][col].values[0]
                if pd.notna(response) and response not in responses:
                    responses.append(response)

            disagreement_count = len(responses) - 1 if len(responses) > 1 else 0
            disagreement_report[title][col] = disagreement_count
            question_disagreement_counts[col] += disagreement_count

    return disagreement_report, question_disagreement_counts


def aggregate_average_similarity(similarity_analysis, run_names):
    """
    Aggregate average similarity scores between runs.
    :param similarity_analysis: Dictionary containing similarity scores.
    :param run_names: List of run names.
    :return: Dictionary containing average similarity scores.
    """
    similarity_sums = {run1: {run2: 0 for run2 in run_names} for run1 in run_names}
    count_sums = {run1: {run2: 0 for run2 in run_names} for run1 in run_names}

    for col, similarities in similarity_analysis.items():
        for run1, run2, similarity in similarities:
            similarity_sums[run1][run2] += similarity
            count_sums[run1][run2] += 1
            similarity_sums[run2][run1] += similarity
            count_sums[run2][run1] += 1

    average_similarities = {
        run1: {
            run2: (
                similarity_sums[run1][run2] / count_sums[run1][run2]
                if count_sums[run1][run2] != 0
                else 0
            )
            for run2 in run_names
        }
        for run1 in run_names
    }

    return average_similarities


def analyze_text_similarity(run_names, dataframes, non_categorical_columns):
    """
    Analyze text similarity between runs for non-categorical columns.
    :param run_names: List of run names.
    :param dataframes: List of pandas DataFrames, one for each run.
    :param non_categorical_columns: List of non-categorical columns to analyze.
    :return: Dictionary containing similarity analysis results.
    """
    similarity_analysis = {col: [] for col in non_categorical_columns}

    for col in non_categorical_columns:
        if col in [
            "title",
            "authors",
            "subjects",
            "Using a list of keywords, what is the setting of the story?",
            "List up to 5 main characters in the story.",
            "Using a list of keywords, what are the major themes of the story?",
            "Using a list, what symbols are used in the story, if any?",
            "Using a list, what motifs are used in the story, if any?",
            "Using a list of keywords what type of language is used? I.e Is the language simple or elegant? Short or long sentences? Straightforward or descriptive? Are words from other languages used frequently?",
            "Using a list of keywords, what sub-genres of science fiction does this story fall under?",
            " a list of keywords, what sub-genres of fantasy does this story fall under?",
            "Using a list of keywords, what sub-genres of horror does this story fall under?",
        ]:
            pass
        else:
            for i in range(len(run_names)):
                for j in range(i + 1, len(run_names)):
                    run1, run2 = run_names[i], run_names[j]
                    df1, df2 = dataframes[i], dataframes[j]

                    common_titles = set(df1["title"]).intersection(df2["title"])
                    df1_filtered = df1[df1["title"].isin(common_titles)]
                    df2_filtered = df2[df2["title"].isin(common_titles)]

                    df1_filtered = df1_filtered.sort_values(by="title").reset_index(
                        drop=True
                    )
                    df2_filtered = df2_filtered.sort_values(by="title").reset_index(
                        drop=True
                    )

                    similarity_scores = []
                    for text1, text2 in zip(df1_filtered[col], df2_filtered[col]):
                        if pd.notna(text1) and pd.notna(text2):
                            similarity = calculate_similarity(text1, text2)
                            similarity_scores.append(similarity)

                    if similarity_scores:
                        average_similarity = sum(similarity_scores) / len(
                            similarity_scores
                        )
                        similarity_analysis[col].append(
                            (run1, run2, average_similarity)
                        )

    return similarity_analysis


def find_single_word_columns(df):
    """Find columns where all records contain only one word.
    :param df: The DataFrame to analyze.
    :return: List of column names where all records contain only one word.
    """
    single_word_columns = []

    translator = str.maketrans("", "", string.punctuation)

    for col in df.columns:
        is_single_word = True
        for value in df[col].dropna():
            cleaned_value = str(value).translate(translator).strip()
            if len(cleaned_value.split()) != 1:
                is_single_word = False
                break

        if is_single_word:
            single_word_columns.append(col)

    return single_word_columns


def get_non_categorical_columns(df, categorical_columns):
    """Get all columns that are not specified as categorical.
    :param df: The DataFrame to analyze.
    :param categorical_columns: List of categorical columns.
    :return: List of non-categorical columns.
    """
    categorical_set = set(categorical_columns)
    non_categorical_columns = [col for col in df.columns if col not in categorical_set]
    return non_categorical_columns


def filter_common_books(dataframes):
    """Filter out records that are not present in all dataframes.
    :param dataframes: List of pandas DataFrames.
    :return: List of filtered pandas DataFrames.
    """
    # There were some books that weren't analyzed in every run due to rate limit issues or other errors.

    common_titles = set(dataframes[0]["title"])

    for df in dataframes[1:]:
        common_titles.intersection_update(df["title"])

    filtered_dataframes = [df[df["title"].isin(common_titles)] for df in dataframes]

    return filtered_dataframes


def normalize_text_for_all(text):
    """Normalize text for all columns, only removes periods and makes lowercase.
    :param text: The text to normalize.
    :return: The normalized text.
    """
    if pd.isna(text):
        return text
    text = text.translate(str.maketrans("", "", "."))
    text = text.lower()
    corrections = {
        "first-person": "first",
        "third-person": "third",
    }
    return corrections.get(text, text)


def compare_runs():
    config = load_config()
    if not config:
        raise Exception("Configuration could not be loaded. Exiting.")

    setup_logging(
        config.get("LOG_FILE", "project_log.log"), config.get("LOG_LEVEL", "INFO")
    )

    process_dir = "data/processed"

    dataframes = []
    run_names = []

    for run_dir in os.listdir(process_dir):
        run_path = os.path.join(process_dir, run_dir, "analysis_results.csv")
        if os.path.isfile(run_path):
            df = pd.read_csv(run_path)
            dataframes.append(df)
            run_names.append(run_dir)

    if not dataframes:
        logging.debug("No analysis results found.")
        exit()

    filtered_dataframes = filter_common_books(dataframes)
    for i, df in enumerate(filtered_dataframes):
        filtered_dataframes[i] = df.sort_values(by="title").reset_index(drop=True)

    normalized_dataframes = []
    for df in filtered_dataframes:
        normalized_df = df.copy()
        for col in normalized_df.columns:
            normalized_df[col] = normalized_df[col].apply(normalize_text_for_all)
        normalized_dataframes.append(normalized_df)

    all_data = pd.concat(normalized_dataframes, keys=run_names, names=["Run", "Index"])

    categorical_columns = find_single_word_columns(all_data)
    logging.debug("Categorical columns: %s", categorical_columns)

    categorical_analysis = {
        col: all_data[col].value_counts() for col in categorical_columns
    }
    logging.debug("Categorical analysis: %s", categorical_analysis)

    disagreement_report, question_disagreement_counts = analyze_disagreements(
        normalized_dataframes, categorical_columns
    )
    logging.debug("Disagreement report: %s", disagreement_report)
    logging.debug("Question disagreement counts: %s", question_disagreement_counts)

    plot_disagreements_per_title(disagreement_report)
    plot_disagreements_per_question(question_disagreement_counts)

    categorical_df = pd.DataFrame(categorical_analysis)
    categorical_df.to_csv(
        os.path.join(process_dir, "categorical_analysis.csv"), index=False
    )

    non_categorical_columns = get_non_categorical_columns(all_data, categorical_columns)
    logging.debug("Non-categorical columns: %s", non_categorical_columns)

    similarity_analysis = analyze_text_similarity(
        run_names, normalized_dataframes, non_categorical_columns
    )
    logging.debug("Similarity analysis: %s", similarity_analysis)

    average_similarities = aggregate_average_similarity(similarity_analysis, run_names)
    logging.debug("Aggregate similarity analysis: %s", average_similarities)

    plot_similarity_heatmap(average_similarities, run_names)

    metrics_dict, run_names = calculate_metrics(
        normalized_dataframes, non_categorical_columns
    )
    logging.debug("Metrics: %s", metrics_dict)

    plot_radar_chart(metrics_dict, run_names)


if __name__ == "__main__":
    compare_runs()
