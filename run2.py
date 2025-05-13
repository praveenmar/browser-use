import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
from dotenv import load_dotenv
import asyncio

# Read GOOGLE_API_KEY into env
#load_dotenv()

browser = Browser(
    config=BrowserConfig(
        browser_binary_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    )
)

# Initialize the model with explicit API key (either from env or hardcoded)
google_api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', google_api_key=google_api_key)

async def main():
    agent = Agent(
        task="search for orange hrm opensource and go to orangehrm website and enter Admin in username field and admin123 in password field nd click on login and validate Dashboards should display on home screen",
        llm=llm
    )
    result = await agent.run()
    print(result)
    await browser.close()

asyncio.run(main())