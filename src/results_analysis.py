import pandas as pd
import string
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import spacy

def load_data(file_path):
    """Load the CSV data into a pandas DataFrame."""
    return pd.read_csv(file_path)

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

def normalize_text(text):
    """Normalize text by removing punctuation, stripping whitespace, and converting to lowercase."""
    if pd.isna(text):
        return text
    text = text.translate(str.maketrans('', '', '.'))
    text = text.strip().lower()
    corrections = {
        "first-person": "first",
        "third-person": "third",
    }
    return corrections.get(text, text)

def apply_normalization(df, columns):
    """Apply normalization to specified columns in the DataFrame."""
    for column in columns:
        df[column] = df[column].apply(normalize_text)
    return df

# Sentiment analysis using TextBlob
def calculate_sentiment(text):
    """Calculate the sentiment polarity of the provided text."""
    if pd.isna(text):
        return None
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Plotting a word cloud
def plot_word_cloud(text_data, column, title):
    """Generate and display a word cloud from the provided text data."""
    wordcloud = WordCloud(width=800, height=400, colormap='magma', background_color='black').generate(' '.join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'reports/figures/word_cloud_{column}.png')

# Plotting a heatmap for theme distributions
def plot_theme_heatmap(df, top_n=10):
    """Plot a heatmap showing the distribution of the top N themes across genres."""
    expanded_rows = []
    for _, row in df.iterrows():
        for genre in row['Genres']:
            for theme in row['Themes']:
                expanded_rows.append({'Genre': genre, 'Theme': theme.strip()})
    
    expanded_df = pd.DataFrame(expanded_rows)
    theme_counts = expanded_df['Theme'].value_counts().nlargest(top_n).index
    top_themes_df = expanded_df[expanded_df['Theme'].isin(theme_counts)]

    heatmap_data = top_themes_df.pivot_table(index='Genre', columns='Theme', aggfunc='size', fill_value=0)

    sns.set(style="ticks", context="talk")
    plt.figure(figsize=(12, 8))

    sns.heatmap(heatmap_data, cmap="magma", annot=True, linewidths=.5)
    plt.title(f'Distribution of Top {top_n} Themes Across Genres')
    plt.xlabel('Themes')
    plt.ylabel('Genres')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig('reports/figures/theme_heatmap.png')

def plot_subgenre_distribution(subgenre_df, subgenre_counts):
    """
    Plots a horizontal bar chart of the number of books in each subgenre.
    
    Args:
    subgenre_df (DataFrame): A DataFrame with columns 'Main Genre', 'Subgenre', and 'Count'.
    subgenre_counts (dict): A dictionary with genre names as keys and corresponding subgenre counts as values
    """
    sns.set(style="ticks", context="talk")

    # Plotting the distribution of subgenres for each main genre
    for genre in subgenre_counts:
        plt.figure(figsize=(12, 6))
        genre_data = subgenre_df[(subgenre_df['Main Genre'] == genre) & (subgenre_df['Subgenre'] != 'speculative fiction')].head(10)
        bar_plot = sns.barplot(x='Count', y='Subgenre', data=genre_data, palette="magma", hue=genre_data.index)
        bar_plot.legend_.remove()

        for i, value in enumerate(genre_data['Count']):
            bar_plot.text(value + 0.3, i, str(value), color='black', ha='center', va='center')
        
        plt.title(f'Distribution of {genre} Subgenres', fontsize=16)
        plt.xlabel('Number of Books', fontsize=10)
        plt.ylabel('Subgenre', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=9)
        plt.savefig(f'reports/figures/{genre}_subgenre_distribution.png')

def plot_genre_counts(genre_df):
    """
    Plots a bar chart of the number of books in each genre.
    
    Args:
    genre_df (DataFrame): A DataFrame with columns 'Genre' and 'Count'.
    """
    sns.set(style="ticks", context="talk")

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Genre', y='Count', data=genre_df, palette="magma", hue=genre_df.index)
    bar_plot.legend_.remove()

    for i, value in enumerate(genre_df['Count']):
        bar_plot.text(i, value + 0.3, str(value), color='black', ha='center', va='center')

    plt.title('Total Books in Each Genre', fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel('Number of Books', fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('reports/figures/genre_counts.png')


def plot_summary_sentiment(sentiment_df):
    """
    Plots histograms of sentiment polarity and subjectivity for summaries.
    Args:
    sentiment_df (DataFrame): A DataFrame containing sentiment polarity and subjectivity scores for summaries
    """
    # Add sentiment analysis results to DataFrame
    sentiment_df[['theme_polarity', 'theme_subjectivity']] = sentiment_df['Theme'].apply(lambda x: pd.Series(get_sentiment(x) if pd.notna(x) else (0, 0)))
    sentiment_df[['summary_polarity', 'summary_subjectivity']] = sentiment_df['Summarize the plot of the story.'].apply(lambda x: pd.Series(get_sentiment(x) if pd.notna(x) else (0, 0)))

    # Set up the figure and axis for four subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    sns.set(style="ticks", context="talk")

    # Polarity plot for summaries
    sns.histplot(sentiment_df['summary_polarity'], bins=10, kde=True, color='blue', ax=axs[0, 0])
    axs[0, 0].set_title('Sentiment Polarity of Summaries')
    axs[0, 0].set_xlabel('Polarity')
    axs[0, 0].set_ylabel('Frequency')

    # Subjectivity plot for summaries
    sns.histplot(sentiment_df['summary_subjectivity'], bins=10, kde=True, color='green', ax=axs[0, 1])
    axs[0, 1].set_title('Sentiment Subjectivity of Summaries')
    axs[0, 1].set_xlabel('Subjectivity')
    axs[0, 1].set_ylabel('Frequency')

    # Polarity plot for themes
    sns.histplot(sentiment_df['theme_polarity'], bins=10, kde=True, color='purple', ax=axs[1, 0])
    axs[1, 0].set_title('Sentiment Polarity of Themes')
    axs[1, 0].set_xlabel('Polarity')
    axs[1, 0].set_ylabel('Frequency')

    # Subjectivity plot for themes
    sns.histplot(sentiment_df['theme_subjectivity'], bins=10, kde=True, color='orange', ax=axs[1, 1])
    axs[1, 1].set_title('Sentiment Subjectivity of Themes')
    axs[1, 1].set_xlabel('Subjectivity')
    axs[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('reports/figures/sentiment_analysis.png')

def plot_character_polarity(character_df):
    character_df[['protagonist_polarity', 'protagonist_subjectivity']] = character_df['Who is the protagonist of the story?'].apply(get_sentiment).apply(pd.Series)
    character_df[['antagonist_polarity', 'antagonist_subjectivity']] = character_df['Who is the antagonist?'].apply(get_sentiment).apply(pd.Series)

    sns.set(style="ticks", context="talk")

    # Visualize the sentiment analysis
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=character_df[['protagonist_polarity', 'antagonist_polarity']], palette='magma')
    plt.title('Sentiment Analysis of Protagonist and Antagonist Descriptions')
    plt.xlabel('Character Type')
    plt.ylabel('Polarity')
    plt.savefig('reports/figures/protagonist_polarity.png')

    # Additional comparison based on gender, orientation, class, and race
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='What is the gender of the protagonist?', y='protagonist_polarity', data=character_df, palette='magma')
    plt.title('Sentiment Analysis of Protagonist Descriptions by Gender')
    plt.xlabel('Gender of Protagonist')
    plt.ylabel('Polarity')
    plt.savefig('reports/figures/gender_polarity.png')

    # Additional comparison based on gender, orientation, class, and race
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='What is the economic status of the protagonist?', y='protagonist_polarity', data=character_df, palette='magma')
    plt.title('Sentiment Analysis of Protagonist Descriptions by Economic Status')
    plt.xlabel('Economic Status of Protagonist')
    plt.ylabel('Polarity')
    plt.savefig('reports/figures/economic_polarity.png')

def plot_marginalized_identities(data):
    """
    Plots the frequency of marginalized identities across the dataset.
    
    Args:
    data (DataFrame): The dataset containing boolean columns for marginalized identities.
    """
    # Count the frequency of each marginalized identity
    identity_counts = data[['LGBT', 'Racial Minority', 'Disability', 'Low Income', 'Women']].sum().reset_index()
    identity_counts.columns = ['Identity', 'Count']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="ticks", context="talk")
    # ax.set_facecolor('black')

    bar_plot = sns.barplot(x='Identity', y='Count', data=identity_counts, ax=ax, palette="magma", hue=identity_counts.index)

    ax.set_title('Representation of Marginalized Identities', fontsize=16)
    ax.set_xlabel('Identity', fontsize=14)
    ax.set_ylabel('Number of Titles', fontsize=10)
    
    bar_plot.legend_.remove()

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    for index, row in identity_counts.iterrows():
        ax.text(index, row['Count'] + 0.05, f'{row["Count"]}', color='black', ha="center")
    
    plt.tight_layout()
    plt.savefig('reports/figures/marginalized_identities.png')

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


def transform_identity_data(df):
    """
    Transforms the dataset to include boolean columns for marginalized identities.
    
    Args:
    df (DataFrame): The  dataset containing llm analysis results.
    
    Returns:
    DataFrame: The transformed dataset with additional boolean columns.
    """
    # Initialize the new columns with False
    df['LGBT'] = False
    df['Racial Minority'] = False
    df['Disability'] = False
    df['Low Income'] = False
    df['Women'] = False

    # Define keywords to identify marginalized identities
    lgbt_keywords = ['LGBT', 'gay', 'lesbian', 'bisexual', 'transgender', 'queer', 'asexual']
    racial_minority_keywords = ['yes']
    disability_keywords = ['yes']
    low_income_keywords = ['lowerclass']
    gender_keywords = ['female']

    for index, row in df.iterrows():
        # Check for LGBT representation
        if pd.notna(row['What is the sexual orientation of the protagonist?']):
            if any(keyword in row['What is the sexual orientation of the protagonist?'].lower() for keyword in lgbt_keywords):
                df.at[index, 'LGBT'] = True

        if pd.notna(row['What is the sexual orientation of the antagonist?']):
            if any(keyword in row['What is the sexual orientation of the antagonist?'].lower() for keyword in lgbt_keywords):
                df.at[index, 'LGBT'] = True

        # Check for Racial Minority representation
        if pd.notna(row['Could any of the characters be considered a racial minority?']):
            if any(keyword in row['Could any of the characters be considered a racial minority?'].lower() for keyword in racial_minority_keywords):
                df.at[index, 'Racial Minority'] = True

        # Check for Disability representation
        if pd.notna(row['Do any of the characters have a disability?']):
            if any(keyword in row['Do any of the characters have a disability?'].lower() for keyword in disability_keywords):
                df.at[index, 'Disability'] = True

        # Check for low-income representation
        if pd.notna(row['What is the economic status of the protagonist?']):
            if any(keyword in row['What is the economic status of the protagonist?'].lower() for keyword in low_income_keywords):
                df.at[index, 'Low Income'] = True

        if pd.notna(row['What is the economic status of the antagonist?']):
            if any(keyword in row['What is the economic status of the antagonist?'].lower() for keyword in low_income_keywords):
                df.at[index, 'Low Income'] = True
        
        # Check for women representation
        if pd.notna(row['What is the gender of the protagonist?']):
            if any(keyword in row['What is the gender of the protagonist?'].lower() for keyword in gender_keywords):
                df.at[index, 'Women'] = True

        if pd.notna(row['What is the gender of the antagonist?']):
            if any(keyword in row['What is the gender of the antagonist?'].lower() for keyword in gender_keywords):
                df.at[index, 'Women'] = True
    return df

# Main analysis function
def analyze_and_visualize(file_path):
    """Main function to load data, perform analysis, and generate visualizations."""
    df = load_data(file_path)
    
    # Normalization
    categorical_columns = find_single_word_columns(df)
    df = apply_normalization(df, categorical_columns)

    non_categorical_columns = get_non_categorical_columns(df, categorical_columns)
    
    # Plot Word Cloud for a specific column
    df = df.rename(
        columns={'Using a list of keywords, what are the major themes of the story?': 'Theme',
        "What is the tone of the story?": 'Tone', 
        "What is the mood of the story?": 'Mood', 
        "Using a list, what symbols are used in the story, if any?": 'Symbols', 
        "Using a list, what motifs are used in the story, if any?": 'Motifs'}
        )
    word_cloud_columns = ['Theme', 'Tone', 'Mood', 'Symbols', 'Motifs']
    for column in word_cloud_columns:
        plot_word_cloud(df[column].dropna(), column, f'Word Cloud of {column}')

    # Create a Genre column based on classification
    df['Genres'] = df.apply(lambda row: [genre for genre, value in {
        'Science Fiction': row['Is this work classified as science fiction?'],
        'Fantasy': row['Is this work classified as fantasy?'],
        'Horror': row['Is this work classified as horror?']
    }.items() if value.lower() == 'yes'], axis=1)

    # Extract themes and associate with genres
    df['Themes'] = df['Theme'].apply(lambda x: x.split(',') if pd.notna(x) else [])

    # Plot Theme Heatmap
    plot_theme_heatmap(df)

    # Sentiment Analysis
    plot_summary_sentiment(df)

    # Character analysis
    character_columns = [
        'Who is the protagonist of the story?',
        'Who is the antagonist?',
        'What is the gender of the protagonist?',
        'What is the gender of the antagonist?',
        'What is the sexual orientation of the protagonist?',
        'What is the sexual orientation of the antagonist?',
        'What is the economic status of the protagonist?',
        'What is the economic status of the antagonist?',
        'What is the religion of the protagonist?',
        'Could any of the characters be considered a racial minority?',
        'Do any of the characters have a disability?'
    ]
    character_df = df[character_columns]
    tdf = transform_identity_data(character_df)
    plot_marginalized_identities(tdf)
    plot_character_polarity(character_df)

    genre_counts = {
        'Science Fiction': df['Is this work classified as science fiction?'].str.lower().value_counts().get('yes', 0),
        'Fantasy': df['Is this work classified as fantasy?'].str.lower().value_counts().get('yes', 0),
        'Horror': df['Is this work classified as horror?'].str.lower().value_counts().get('yes', 0)
    }

    # Convert to DataFrame for easy plotting
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    logging.info(f"Genre counts: {genre_counts}")
    plot_genre_counts(genre_df)

    subgenre_columns = {
        'Science Fiction': 'Using a list of keywords, what sub-genres of science fiction does this story fall under?',
        'Fantasy': 'Using a list of keywords, what sub-genres of fantasy does this story fall under?',
        'Horror': 'Using a list of keywords, what sub-genres of horror does this story fall under?'
    }

    # Extract and count subgenres
    subgenre_counts = {}
    for genre, col in subgenre_columns.items():
        subgenres = df[col].dropna().apply(lambda x: [s.strip() for s in x.split(',')])
        subgenre_counts[genre] = subgenres.explode().value_counts()

    subgenre_df = pd.concat(subgenre_counts, names=['Main Genre', 'Subgenre']).reset_index(name='Count')
    plot_subgenre_distribution(subgenre_df, subgenre_counts)



if __name__ == "__main__":
    data_path = 'data/processed/analysis_results.csv'
    analyze_and_visualize(data_path)
