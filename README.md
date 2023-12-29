# NLP Text Analyzer

This project is a Python-based Natural Language Processing (NLP) Text Analyzer that uses Streamlit for the user interface and leverages Hugging Face's `transformers` library to perform text summarization using the BART model, visualize word clouds, and display the most common words in a given text.

## Overview

The NLP Text Analyzer consists of the following functionalities:

- **Text Summarization**: Utilizes the BART model from Hugging Face's `transformers` library to generate a summary of the user-provided text.
- **Word Cloud Generation**: Generates a word cloud visualization based on the input text.
- **Most Common Words**: Displays the top 10 most common words and their frequencies in the input text.

## Libraries Used

- `streamlit`: Used for building the web-based user interface.
- `transformers` (from Hugging Face): Provides pre-trained models for NLP tasks. Specifically, the `BartForConditionalGeneration` and `BartTokenizer` are used for text summarization.
- `nltk`: Utilized for text processing tasks like tokenization and frequency analysis.
- `wordcloud`: Enables the creation of word cloud visualizations.
- `matplotlib`: Used for plotting word cloud and other visualizations.

## Usage

### Setup

1. Install the necessary Python dependencies listed in `requirements.txt`.
2. Run the Streamlit app locally using the command: `streamlit run your_script.py`.

### Functionality

1. **Text Input**: Enter your text in the provided text area.
2. **Summary**: Displays a summary of the input text using the BART model.
3. **Word Cloud**: Shows a visual representation of word frequency in the input text.
4. **Most Common Words**: Provides a table showing the top 10 most common words and their frequencies.

## Collab Notebook
Access the Colab notebook used for development [here](https://colab.research.google.com/drive/1Y2vv_pZ5nKXKLrXrmsSu6z8hz6ncjWOz#scrollTo=y5-24_9jLdT2).

## Acknowledgments
- The project utilizes the power of Hugging Face's `transformers` library for NLP tasks.
- The word cloud visualization is created using the `wordcloud` library.
