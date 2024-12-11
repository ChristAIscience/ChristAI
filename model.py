import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class JesusAIModel:
    def __init__(self, model_name="gpt2", custom_weights_path=None):
        """
        Initializes the Jesus AI model by loading a pre-trained GPT2 model and tokenizer.
        Optionally, load custom weights if specified.
        
        Parameters:
        - model_name (str): Name of the GPT2 model to use (default: 'gpt2').
        - custom_weights_path (str): Path to custom model weights for fine-tuning (default: None).
        """
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # If a custom model weights path is provided, load the fine-tuned weights
        if custom_weights_path:
            self.model.load_state_dict(torch.load(custom_weights_path))
        
        # Set the model to evaluation mode
        self.model.eval()

    def generate_response(self, input_text, max_length=150, temperature=0.7):
        """
        Generates a response based on the input text using the GPT2 model.
        
        Parameters:
        - input_text (str): The input question or prompt for the model.
        - max_length (int): The maximum length of the generated response.
        - temperature (float): The randomness of the model output. Lower values make the output more deterministic.
        
        Returns:
        - str: The generated response text.
        """
        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate the output sequence
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the output sequence into text
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def fine_tune(self, train_data, epochs=3, batch_size=8, learning_rate=5e-5):
        """
        Fine-tune the pre-trained GPT-2 model on the provided training data.
        
        Parameters:
        - train_data (DataFrame): A pandas DataFrame with columns 'text' containing training examples.
        - epochs (int): Number of epochs for fine-tuning.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for training.
        
        Returns:
        - model (GPT2LMHeadModel): Fine-tuned model.
        """
        from torch.utils.data import DataLoader, Dataset
        from transformers import AdamW
        from tqdm import tqdm

        # Prepare the training data
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length)
                return torch.tensor(encodings['input_ids'])

        # Convert the text data into a dataset
        train_dataset = TextDataset(train_data['text'].tolist(), self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set up the optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Fine-tuning loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                outputs = self.model(batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader)}")
        
        return self.model
