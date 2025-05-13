import asyncio

from browser_use import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Disable telemetry to avoid connection errors
os.environ["BROWSER_USE_DISABLE_TELEMETRY"] = "1"

# Load environment variables
#load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = "sk-f514046c3e5f43e5825c7c5facff7757"
async def main():
    # Initialize the model
    llm = ChatOpenAI(base_url='https://api.deepseek.com/v1', model='deepseek-reasoner', api_key=SecretStr(api_key))

    # Create agent with the model
    agent = Agent(
        task="Navigate to google.com and search for 'latest AI news'",
        llm=llm,
        browser=None  # Add browser config if needed
    )

    try:
        await agent.run()
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())