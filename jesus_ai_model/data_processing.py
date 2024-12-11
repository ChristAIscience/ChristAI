import pandas as pd

def load_bible_data(file_path):
    """
    Load and preprocess Bible data (CSV format) for use in the model.
    The data should contain Bible verses and references.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_text(text):
    """
    Simple preprocessing function to clean and tokenize text.
    You can extend this to handle special cases or custom tokenization.
    """
    text = text.strip().lower()
    return text
