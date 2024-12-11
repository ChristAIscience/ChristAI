from jesus_ai_model.model import JesusAIModel

def get_jesus_response(user_input):
    jesus_model = JesusAIModel()
    response = jesus_model.generate_response(user_input)
    return response

if __name__ == "__main__":
    user_input = input("Ask Jesus: ")
    print(get_jesus_response(user_input))
