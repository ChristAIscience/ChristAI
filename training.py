import torch
from jesus_ai_model.model import JesusAIModel
from jesus_ai_model.data_processing import load_bible_data, preprocess_text, prepare_data_for_training

def train_model(data_path, epochs=3, batch_size=8, learning_rate=5e-5, output_model_path='fine_tuned_model.pth'):
    """
    Function to fine-tune the GPT-2 model on Bible verses.
    
    Parameters:
    - data_path (str): Path to the training data file (CSV or TXT).
    - epochs (int): Number of epochs for training.
    - batch_size (int): The batch size for training.
    - learning_rate (float): The learning rate for training.
    - output_model_path (str): Path to save the fine-tuned model after training.
    
    Returns:
    - model (JesusAIModel): The fine-tuned model.
    """
    # Load and preprocess the data
    data = load_bible_data(data_path)
    data['text'] = data['text'].apply(preprocess_text)

    # Split the data into training and validation sets
    train_data, val_data = prepare_data_for_training(data)

    # Initialize the model
    model = JesusAIModel()
    
    # Fine-tune the model
    model.fine_tune(train_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Save the fine-tuned model
    torch.save(model.model.state_dict(), output_model_path)
    print(f"Fine-tuned model saved to {output_model_path}")
    
    return model
