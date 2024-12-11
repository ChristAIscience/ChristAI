import pytest
from jesus_ai_model.model import JesusAIModel

def test_generate_response():
    model = JesusAIModel()
    response = model.generate_response("What is the meaning of love?")
    assert len(response) > 0, "Response should not be empty"
    assert "love" in response.lower(), "Response should mention love"
