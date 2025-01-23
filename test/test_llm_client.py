import os
import pytest
from src.utils.llm_client import LLMClient

def test_llm_client_singleton():
    # Set environment variables for testing
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_BASE_URL"] = "https://test_url"

    # Create two instances of LLMClient
    client1 = LLMClient().get_client()
    client2 = LLMClient().get_client()

    # Check that both instances are the same (singleton)
    assert client1 is client2

    # Check that the client is initialized with the correct API key and base URL
    assert client1.api_key == "test_api_key"
    assert client1.base_url == "https://test_url"

if __name__ == "__main__":
    pytest.main()