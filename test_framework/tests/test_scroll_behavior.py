import asyncio
import logging
from browser_use.browser.session import BrowserSession
from browser_use.test_framework.agent.test_agent import TestAgent
from browser_use.agent.service import Agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scroll_behavior():
    # Initialize browser session
    browser_session = BrowserSession()
    await browser_session.start()

    # Create test agent
    agent = TestAgent(browser_session=browser_session)

    # Define the task
    task = """
    step 1. Navigate to https://www.amazon.in/
    step 2. click on hamberger menu with the text 'All'
    step 3. assert that the text 'Digital Content and Devices' is present.
    step 4. click on sub nav with the text 'Amazon Prime Music'
    step 5. Assert that the text 'Amazon Prime Msic' with aria-level=2 in sub nav
    step 6. and select the option with text 'Amazon Prime Music'
    """

    # Run the test
    result = await agent.run_test(task)

    # Log the result
    logger.info(f"Test result: {result}")

    # Clean up
    await browser_session.stop()

if __name__ == "__main__":
    asyncio.run(test_scroll_behavior()) 