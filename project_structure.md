# EFI Speculative Review Project Structure

## Data

This directory contains all data related to the project.

- `processed/`
  - run/`analysis_results.csv` - LLM responses
  - `books_metadata.csv` - Extracted book information
- `full_text/`
  - [Full text files of books]
- metadata/
  - metadata corresponding to each full-text

## Notebooks

This directory contains Jupyter Notebooks for data collection and analysis.

- `data_analysis.ipynb` - Jupyter Notebook for data analysis

## Source Code

This directory contains all source code files.

- `__init__.py`
- `data_collection.py` - Script for collecting data from Project Gutenberg API
- `llm_analysis.py` - Script for analyzing text using OpenAI's API
- compare_runs.py - Script for comparing results across runs
- results_analysis.py - Script for visualizing results from a single run
- `create_metadata.py` - Script for creating metadata from the collected data
- `logging_config.py` - Logging configuration for the project
- `utils.py` - Utility functions

## Reports

This directory contains reports, figures, and drafts related to the dissertation.

- `figures/` - Folder to store plots and figures

## Tests

This directory contains test scripts for the project.

- `test_data_collection.py` - Tests for data collection script
- `test_data_analysis.py` - Tests for data analysis script

## Logs

This directory contains log files.

- `data_collection.log` - Log file for data collection script

## Configuration

This directory contains configuration files.

- `config.json` - Configuration settings for the project
- `prompt.json` - Prompts for OpenAI API

## Root Directory Files

- `LICENSE` - License for the project
- `README.md` - Project overview and setup instructions
- `requirements.txt` - Required Python libraries
- `.gitignore` - Files and directories to be ignored by git
- `project_structure.md` - This file, describing the project structure
