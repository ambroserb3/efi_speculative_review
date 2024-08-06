# Speculative Fiction Analysis with LLMs

This project explores the use of large language models (LLMs) for analyzing speculative fiction literature. It aims to automate and provide a foundation to scale literary analysis, providing consistent and reproducible insights into themes, character development, and socio-political issues.

### Project Overview

Traditional literary analysis is often subjective and time-consuming. This project addresses these challenges by leveraging LLMs to analyze large corpora of speculative fiction, offering a systematic and scalable approach. By identifying patterns, themes, and character dynamics, the project aims to uncover new interpretations and insights.

### Technologies Used

- **Python:** The primary programming language for data collection, analysis, and visualization.
- **Large Language Models (LLMs):** For natural language processing and text generation tasks.
- **Spacy:** For grammatical, similarity analysis and part-of-speech tagging.
- **Textblob**: For sentiment analysis
- **Matplotlib and Seaborn:** For data visualization.

### Key Features

1. **Automated Text Analysis:** Extracts plot summaries, character descriptions, thematic elements, and more.
2. **Sentiment and Consistency Analysis:** Evaluates the sentiment of descriptions and the consistency of LLM responses across different runs.
3. **Data Visualization:** Provides graphical representations of findings, including sentiment distribution, theme frequency, and more.
4. **Scalability:** Built with scalability in mind.

### Usage

The project primarily uses a command-line interface (CLI) for executing scripts and analyzing outputs.

**How to Use:**

1. **Data Collection:** Use `data_collection.py` to gather texts from sources like Project Gutenberg.
2. **Text Analysis:** Run `llm_analysis.py` to analyze texts using the LLM.
3. **Comparison and Visualization:** Use `compare_runs.py` and `results_analysis.py` to compare outputs and visualize results.

### Installation

1. **Install Python 3.8+:** Ensure you have Python 3.8 or later installed on your system.
2. **Install Dependencies:**

```bash
   pip install -r requirements.txt
```

**Set up LLM API Access:**

- Obtain an API key from OpenAI.
- Configure your API key in config.json

### Configuration

```json
{
  "SUBJECTS": ["Science Fiction", "Horror", "Fantasy"],
  "OUTPUT_DIR_TEXT": "data/full_text",
  "OUTPUT_DIR_METADATA": "data/metadata",
  "PROCESSED_DATA_DIR": "data/processed",
  "LOG_FILE": "data_collection.log",
  "LOG_LEVEL": "INFO",
  "OPENAI_API_KEY": "Your API key",
  "OPENAI_MODEL": "gpt-4o-mini",
  "iterations": 5,
  "max_books": 70,
  "max_workers": 10,
  "temperature": 0.4,
  "top_p": 0.8
}
```

### Usage Examples

```bash
# Data Collection
python -m src.data_collection

# Text Analysis
python -m src.llm_analysis.py

# Comparison and Visualization
python -m src.compare_runs
python -m src.results_analysis

```

alternatively:

```bash
python -m src.main
```

This command currently will run the data_collection and llm_analysis steps, and run llm_analysis for the number of iterations specified in your configuration.

### Future Work

The study of speculative fiction through large language models (LLMs) opens numerous avenues for future research. While this project has laid the groundwork for using LLMs in literary analysis, there remain several unexplored areas and opportunities for refinement.

**Leveraging Historical Context During Publication:** Future research could incorporate detailed historical context data, such as the socio-political climate and major events during the time of a work's publication. This additional layer could provide a richer understanding of the narratives, offering insights into how historical circumstances influenced the themes and characterizations in speculative fiction.

**Tracking the Effects of Temperature and Top_p on Analysis:** Investigating the impact of varying LLM parameters, such as Temperature and Top_p, on analysis outcomes could yield valuable information about the model's interpretative flexibility. By systematically adjusting these parameters, researchers can study how they influence the model's creative outputs and consistency, thereby identifying optimal settings for literary analysis.

**Performance with Different LLM Models:** Future studies could compare the performance of various LLM architectures, including but not limited to GPT, BERT, and newer models like T5 or Bloom. This comparison could reveal differences in interpretative accuracy, consistency, and bias, helping to identify the most suitable models for different types of literary analysis tasks.

**Creating Solid Consistency and Reliability Benchmarks:** Establishing benchmarks for consistency and reliability in LLM-generated analyses is crucial. Future work could focus on developing standardized metrics to evaluate the reproducibility of LLM outputs across multiple runs and diverse datasets. This would involve creating controlled experiments and datasets that can serve as benchmarks for future research.

**Cross-referencing Results with Other Sources of Literary Analysis:** To validate and enrich the findings, future research could cross-reference LLM-generated analyses with traditional literary criticism and other digital humanities methodologies. This triangulation could enhance the credibility of the results and provide a more comprehensive understanding of the texts.

**Prompt-Engineering Techniques for Refinement:** Continued experimentation with prompt-engineering techniques could refine the accuracy and depth of LLM analyses. By exploring different prompt structures, question formulations, and contextual clues, researchers can optimize the models for more nuanced and specific literary analyses. This could also involve developing dynamic prompts that adapt based on initial LLM responses, enhancing the depth and relevance of subsequent analyses.

These potential research directions underscore the immense possibilities that LLMs offer for the future of literary analysis. By continuing to innovate and explore these areas, researchers can push the boundaries of what is possible in the intersection of artificial intelligence and the humanities.

### Contributing

Contributions are welcome!

1. **Fork the repository:** Create a fork of the project on GitHub.
2. **Create a branch:** Create a new branch for your changes.
3. **Make your changes:** Implement your changes and ensure they follow the project's coding standards.
4. **Submit a pull request:** Submit a pull request with a clear description of your changes.

### License

This project is licensed under the [MIT License](vscode-webview://1dfbo1r3ig1tpkoru0cp1840vs7r71jms0v1p8f5vg8q3r59ct9c/LICENSE).

### Contact

For any questions or support, please contact s2599421@ed.ac.uk.
