# run_test.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import asyncio
import logging
from test_framework.agent.test_agent import TestAgent
from browser_use.controller.service import Controller
from browser_use.browser.session import BrowserSession
from browser_use.browser import BrowserProfile
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sys
import json
from typing import List, Dict, Any

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
    browser_profile = BrowserProfile(
        headless=False,
        ignore_https_errors=True,  # Allow insecure connections
        bypass_csp=True,  # Bypass Content Security Policy
        java_script_enabled=True,  # Ensure JavaScript is enabled
        offline=False,  # Ensure we're not in offline mode
        ignore_default_args=['--disable-extensions']  # Don't disable extensions
    )
    browser_session = BrowserSession(browser_profile=browser_profile)
    await browser_session.start()
    
    # Initialize LLM
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=google_api_key
    )
    
    # Create test agent with the task description
    agent = TestAgent(
        browser_session=browser_session,
        llm_client=llm,
        settings={
            "task": """
               @agent=keep this in mind that you are testing a website so, follow the instructions carefully do not go beyond the instructions
                step 1. Navigate to http://uitestingplayground.com/
                step 2. assert that the text 'Wait for edit field to become enabled' is displaying on the page.
                step 3. Assert that the text 'Entering text into an edit field may not have' is displayed on the page.
                step 4. Assert that the text 'Check that class attribute based XPath is well formed' is displayed on the page.
            """,
            "assertion_mode": "hard"  # Use hard assertions to stop on first failure
        }
    )
    
    try:
        logger.info("Starting test execution...")
        
        # Define test data with proper step structure
        test_data = {
            "steps": [
                {
                    "action": "navigate",
                    "params": {
                        "url": "http://uitestingplayground.com/"
                    }
                }
            ],
            "requirements": [
                "assert that the text 'Wait for edit field to become enabled' is displaying on the page",
                "assert that the text 'This text will never be found on the page' is displayed on the page",
                "assert that the text 'Check that class attribute based XPath is well formed' is displayed on the page"
            ]
        }
        
        result = await agent.run_test(test_data)
        
        # Check if the test was successful
        if result['success']:
            logger.info("✅ Test completed successfully!")
        else:
            # Extract error message from the result
            error_msg = result.get('error', "Test failed without specific error message")
            logger.error(f"❌ Test failed: {error_msg}")
            if 'validation_results' in result:
                logger.error(f"Failure details: {json.dumps(result['validation_results'], indent=2)}")
            sys.exit(1)  # Exit with error code
            
        return result
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        raise
    finally:
        await browser_session.stop()

if __name__ == "__main__":
    asyncio.run(main())