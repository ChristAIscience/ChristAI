import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class JesusAIModel:
    def __init__(self):
        self.model_name = "gpt2"  # Replace with a more appropriate model if needed
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

    def generate_response(self, input_text):
        # Encode the input text and generate a response
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
