# run_test.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from browser_use.browser.session import BrowserSession
from browser_use.browser import BrowserProfile
from browser_use.controller.service import Controller
from browser_use.agent.views import ActionResult
from test_framework.agent.test_agent import TestAgent
from test_framework.agent.step_hooks import StepHooks
from test_framework.output_processor import OutputProcessor
from test_framework.models.action_model import ActionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set specific loggers to appropriate levels
logging.getLogger('controller').setLevel(logging.INFO)  # Keep controller logs
logging.getLogger('test_framework.validation.assertions').setLevel(logging.INFO)  # Keep validation logs
logging.getLogger('browser').setLevel(logging.INFO)  # Keep browser logs
logging.getLogger('agent').setLevel(logging.INFO)  # Keep agent logs
logging.getLogger('test_framework.agent.step_hooks').setLevel(logging.INFO)

# Create a filter to exclude only extraction content
class ExtractionFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        # Only filter out extraction content messages
        if "Extracted from page" in message or "page_content" in message or "exact_text" in message:
            return False
        return True

# Apply the filter only to the controller logger
controller_logger = logging.getLogger('controller')
controller_logger.addFilter(ExtractionFilter())

logger = logging.getLogger(__name__)

class CustomStepHooks(StepHooks):
    """Custom step hooks with enhanced functionality."""
    
    async def before_step(self, step_number: int, action: ActionModel) -> None:
        """Enhanced before step hook with validation and setup."""
        await super().before_step(step_number, action)
        
        # Validate browser state
        if hasattr(self.agent, 'browser_session'):
            page = await self.agent.browser_session.get_current_page()
            if not page:
                logger.warning(f"Browser page not ready before step {step_number}")
                return
                
            # Check if page is loaded
            try:
                await page.wait_for_load_state('networkidle', timeout=5000)
            except Exception as e:
                logger.warning(f"Page not fully loaded before step {step_number}: {str(e)}")
        
        logger.info(f"Starting step {step_number}: {action}")
        
    async def after_step(self, step_number: int, action: ActionModel, result: ActionResult) -> None:
        """Enhanced after step hook with detailed verification."""
        await super().after_step(step_number, action, result)
        
        if not result.success:
            logger.warning(f"Step {step_number} failed, attempting recovery...")
            
            # Try to get more context about the failure
            if hasattr(self.agent, 'browser_session'):
                try:
                    page = await self.agent.browser_session.get_current_page()
                    # Take screenshot on failure
                    screenshot_path = f"failure_step_{step_number}.png"
                    await page.screenshot(path=screenshot_path)
                    logger.info(f"Failure screenshot saved to {screenshot_path}")
                    
                    # Get page URL for context
                    current_url = page.url
                    logger.info(f"Failure occurred at URL: {current_url}")
                except Exception as e:
                    logger.error(f"Error during failure analysis: {str(e)}")
        
        logger.info(f"Completed step {step_number} with result: {result.success}")
        
    async def on_error(self, step_number: int, action: ActionModel, error: Exception) -> None:
        """Enhanced error handling with recovery attempts."""
        await super().on_error(step_number, action, error)
        
        logger.error(f"Error in step {step_number}: {str(error)}")
        
        # Attempt recovery for specific error types
        if "TimeoutError" in str(error):
            logger.info("Timeout error detected, attempting to refresh page...")
            if hasattr(self.agent, 'browser_session'):
                try:
                    page = await self.agent.browser_session.get_current_page()
                    await page.reload()
                    logger.info("Page refreshed successfully")
                except Exception as e:
                    logger.error(f"Failed to refresh page: {str(e)}")
        
    async def on_complete(self, success: bool) -> None:
        """Enhanced completion hook with cleanup and reporting."""
        await super().on_complete(success)
        
        # Generate test report
        if hasattr(self.agent, 'result'):
            total_steps = len(self.agent.result)
            passed_steps = sum(1 for r in self.agent.result if r.success)
            success_rate = (passed_steps / total_steps) * 100 if total_steps > 0 else 0
            
            logger.info(f"\nTest Summary:")
            logger.info(f"Total Steps: {total_steps}")
            logger.info(f"Passed Steps: {passed_steps}")
            logger.info(f"Failed Steps: {total_steps - passed_steps}")
            logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Cleanup resources
        if hasattr(self.agent, 'browser_session'):
            try:
                await self.agent.browser_session.stop()
                logger.info("Browser session closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser session: {str(e)}")

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
        ignore_default_args=['--disable-extensions'],  # Don't disable extensions
        
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'  # Set a modern user agent
    )
    browser_session = BrowserSession(browser_profile=browser_profile)
    
    try:
        await browser_session.start()
        
        # Initialize LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1  # Lower temperature for more deterministic responses
        )
        
        # Create test agent with the task description
        agent = TestAgent(
            browser_session=browser_session,
            llm_client=llm,
            settings={
                "task": """
                   
                @agent=keep this in mind that you are testing a website so, follow the instructions carefully do not go beyond the instructions
                step 1. Navigate to http://uitestingplayground.com/dynamicid
                step 2. assert that the text 'Then execute your test to make sure that ID is not used for button identification.' is visible
                step 3. Finally give me a report of the test like what passed and what failed.
                    
                """,
                "assertion_mode": "soft"  # Use soft assertions to continue on failure
            }
        )
        
        # Initialize and attach custom step hooks
        step_hooks = CustomStepHooks(agent)
        agent.step_hooks = step_hooks
        
        logger.info("Starting test execution...")
        
        # Let the agent handle everything from the task description
        result = await agent.run_test({})
        
        # Process and clean up the output
        if 'validation_results' in result:
            for validation in result['validation_results']:
                if 'extracted_content' in validation:
                    validation['extracted_content'] = OutputProcessor.process_extraction_result(
                        validation['extracted_content']
                    )
        
        # Calculate success rate
        if 'validation_results' in result:
            total_steps = len(result['validation_results'])
            passed_steps = sum(1 for v in result['validation_results'] if v.get('success', False))
            failed_steps = total_steps - passed_steps
            success_rate = (passed_steps / total_steps) * 100 if total_steps > 0 else 0
            
            # Log to console
            logger.info(f"\nTest Summary:")
            logger.info(f"{passed_steps} out of {total_steps} steps passed successfully")
            logger.info(f"{failed_steps} out of {total_steps} steps failed")
            logger.info(f"Success Rate: {success_rate:.0f}%\n")
        
        # Write cleaned output to file with clear formatting
        with open('output.txt', 'w', encoding='utf-8') as f:
            # Write test status
            if result['success']:
                f.write("✅ Test completed successfully!\n\n")
            else:
                f.write("❌ Test failed!\n\n")
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n\n")
            
            # Write success rate summary
            if 'validation_results' in result:
                f.write(f"Test Summary:\n")
                f.write(f"{passed_steps} out of {total_steps} steps passed successfully\n")
                f.write(f"{failed_steps} out of {total_steps} steps failed\n")
                f.write(f"Success Rate: {success_rate:.0f}%\n\n")
            
            # Write validation results in a clear format
            if 'validation_results' in result:
                f.write("Validation Results:\n")
                f.write("=" * 80 + "\n")
                for validation in result['validation_results']:
                    # Step information
                    f.write(f"\nStep {validation.get('step', 'N/A')}\n")
                    f.write("-" * 40 + "\n")
                    
                    # Requirement being tested
                    f.write(f"Requirement: {validation.get('requirement', 'N/A')}\n")
                    
                    # Extracted content (cleaned up)
                    if 'extracted_content' in validation:
                        f.write(f"Content: {validation['extracted_content']}\n")
                    
                    # Pass/Fail status
                    if 'success' in validation:
                        status = "✅ Passed" if validation['success'] else "❌ Failed"
                        f.write(f"Status: {status}\n")
                    
                    # Any relevant messages
                    if 'message' in validation:
                        f.write(f"Message: {validation['message']}\n")
                    
                    f.write("=" * 80 + "\n")
        
        # Check if the test was successful
        if result['success']:
            logger.info("✅ Test completed successfully!")
        else:
            # Extract error message from the result
            error_msg = result.get('error', "Test failed without specific error message")
            logger.error(f"❌ Test failed: {error_msg}")
            if 'validation_results' in result:
                import json  # Move json import here where it's actually used
                logger.error(f"Failure details: {json.dumps(result['validation_results'], indent=2)}")
            sys.exit(1)  # Exit with error code
            
        return result
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        raise
    finally:
        # Proper cleanup
        try:
            await browser_session.stop()
        except Exception as e:
            logger.error(f"Error during browser cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        # Set up the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the main function
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
    finally:
        try:
            # Clean up the event loop
            loop.close()
        except Exception as e:
            logger.error(f"Error during event loop cleanup: {str(e)}")
        sys.exit(0)