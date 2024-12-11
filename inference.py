from jesus_ai_model.model import JesusAIModel

def get_jesus_response(user_input, model=None):
    """
    Function to get the response from the Jesus AI model.
    
    Parameters:
    - user_input (str): The question or prompt for the model.
    - model (JesusAIModel, optional): An instance of the JesusAIModel. If not provided, a new instance is created.
    
    Returns:
    - str: The generated response from the model.
    """
    if not model:
        model = JesusAIModel()  # Load the default model if not passed
    
    # Get the response from the model
    response = model.generate_response(user_input)
    
    return response

if __name__ == "__main__":
    user_input = input("Ask Jesus: ")
    response = get_jesus_response(user_input)
    print("Response:", response)
