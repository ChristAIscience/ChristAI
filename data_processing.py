import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

def load_bible_data(file_path):
    """
    Load Bible data from a CSV or text file. The file should contain the Bible verses with proper references.

    Parameters:
    - file_path (str): Path to the data file (CSV or text).
    
    Returns:
    - DataFrame: Pandas DataFrame containing the text and reference columns.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        # If the file is a text file, parse each line as a separate verse
        with open(file_path, 'r') as f:
            verses = f.readlines()
        
        data = pd.DataFrame(verses, columns=['text'])
        data['reference'] = 'Unknown'

    else:
        raise ValueError("Unsupported file format. Only CSV and TXT are supported.")
    
    return data

def preprocess_text(text):
    """
    Preprocesses the input text by stripping unnecessary whitespace and converting to lowercase.
    Optionally, can remove punctuation or perform other custom cleaning.

    Parameters:
    - text (str): The text to be preprocessed.

    Returns:
    - str: The cleaned text.
    """
    # Remove extra whitespace and convert to lowercase
    text = text.strip().lower()
    
    # Remove non-alphanumeric characters (optional)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    return text

def prepare_data_for_training(data, test_size=0.2):
    """
    Splits the dataset into training and validation sets.

    Parameters:
    - data (DataFrame): The dataset containing text data.
    - test_size (float): The proportion of the dataset to be used as the validation set.

    Returns:
    - tuple: Two DataFrames (train_data, val_data) split according to the test_size.
    """
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data

def save_data_to_csv(data, output_path):
    """
    Saves the processed data to a CSV file.

    Parameters:
    - data (DataFrame): The DataFrame containing the processed data.
    - output_path (str): Path to save the processed data.
    """
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
