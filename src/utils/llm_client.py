import os

from openai import OpenAI


class LLMClient:
    _instance = None  # Private static variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        """
        Controls the creation of the single instance of LLMClient.
        Initializes the OpenAI client if it does not already exist.
        """
        if not cls._instance:
            cls._instance = super(LLMClient, cls).__new__(cls, *args, **kwargs)
            # Initialize the OpenAI client with API key and base URL from environment variables
            cls._instance.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "temp"),
                base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
            )
        return cls._instance

    def get_client(self):
        """
        Provides access to the single instance of the OpenAI client.

        Returns:
            OpenAI: The OpenAI client instance.
        """
        return self.client