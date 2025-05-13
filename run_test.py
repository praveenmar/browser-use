# run_test.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import asyncio
import logging
from test_framework.agent.test_agent import TestAgent
from browser_use.controller.service import Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    controller = Controller()
    browser = Browser(config=BrowserConfig())
    browser_context = BrowserContext(browser=browser, config=browser.config.new_context_config)
    
    # Initialize LLM
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=google_api_key
    )
    
    # Create test agent with the task description
    agent = TestAgent(
        task="""
               @agent=keep this in mind that you are testing a website so, follow the instructions carefully do not go beyond the instructions
                step 1. Navigate to http://uitestingplayground.com/home
                step 2. assert the 'Wait for animation to stop before clicking a button' is a text on the page
                step 3. assert the link 'Shadow DOM' with href 'http://uitestingplayground.com/shadowdom' is present in the page
                step 4  assert the link Mouse Over with href 'http://uitestingplayground.com/mousehover' is present in the page
                step 5. click on the 'Shadow DOM' hyperlink
                step 6. assert the text 'Shadow DOM' is present in the page
                """,
        llm=llm,
        controller=controller,
        browser=browser,
        browser_context=browser_context
    )
    
    try:
        logger.info("Starting test execution...")
        result = await agent.run()
        logger.info("Test completed successfully!")
        return result
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())