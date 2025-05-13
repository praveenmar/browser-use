import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

class LLMClient:
    def __init__(self, model_name: str):
        # Load environment variables from .env
        load_dotenv()

        # Now fetch the API Key
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError("Google API Key not found. Please ensure .env is correctly set.")

        # Initialize the model with API Key
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key
        )

    async def plan_actions(self, task_description: str) -> list:
        response = self.model.invoke(task_description)
        actions = self._parse_response(response)
        return actions

    def _parse_response(self, response):
        return [
            {"type": "navigate", "url": "https://example.com"},
            {"type": "click", "selector": "#login"},
            {"type": "type", "selector": "#username", "text": "testuser"}
        ]
