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
import json
import sys

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
                @agent:
                    You are performing automated testing on a website. Your task is to strictly follow the provided instructions step by step. Do not take any actions beyond what is explicitly specified. Your goal is to validate that the website behaves as expected based on the instructions.
                step 1. Navigate to http://uitestingplayground.com/home
                step 2. Assert that the text 'Wait foredit field to become enabled' is displayed on the page.
                step 3. and Assert that the 'Entering text into an edit field may not have' is displayed on the page.
                step 4. Assert that the text 'Check that class attribute based XPath is well formed' is displayed on the page.
        """,
        llm=llm,
        controller=controller,
        browser=browser,    
        browser_context=browser_context,
        assertion_mode="soft"  # Use soft assertions to continue even if some verifications fail
    )
    
    try:
        logger.info("Starting test execution...")
        result = await agent.run()
        
        # Check if the test was successful
        if result.success:
            logger.info("✅ Test completed successfully!")
        else:
            # Extract error message from the result
            error_msg = None
            if hasattr(result, 'error') and result.error:
                error_msg = result.error
            elif hasattr(result, 'extracted_content') and result.extracted_content:
                error_msg = result.extracted_content
            else:
                error_msg = "Test failed without specific error message"
                
            logger.error(f"❌ Test failed: {error_msg}")
            if hasattr(result, 'metadata') and result.metadata:
                logger.error(f"Failure details: {json.dumps(result.metadata, indent=2)}")
            sys.exit(1)  # Exit with error code
            
        return result
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())