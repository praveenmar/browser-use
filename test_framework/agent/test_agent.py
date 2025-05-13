# test_framework/agent/test_agent.py
import os
from pathlib import Path
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from test_framework.reporting.test_reporter import TestReporter
from dotenv import load_dotenv
import logging
from test_framework.validation.assertions import TestAssertions
from test_framework.agent.step_hooks import step_end_hook
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.controller.service import Controller
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext

# test_framework/agent/test_agent.py
from test_framework.validation.assertions import TestAssertions

logger = logging.getLogger(__name__)

class TestAgent:
    def __init__(self, task: str, llm: BaseChatModel, controller: Controller, browser: Browser, browser_context: BrowserContext):
        self.task = task
        self.llm = llm
        self.controller = controller
        self.browser = browser
        self.browser_context = browser_context
        
        # Register assertion actions
        TestAssertions.register_actions(controller)
        
        # Initialize the agent
        self.agent = Agent(
            task=task,
            llm=llm,
            controller=controller,
            browser=browser,
            browser_context=browser_context
        )
        
        load_dotenv()
        
        # Initialize the LLM using browser-use's supported format
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key
        )
        
        self.task_description = task
        self.reporter = TestReporter()
        self.assertions = None

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    async def run(self):
        try:
            logger.info("Starting test execution...")
            logger.info(f"Task: {self.task_description}")

            # Initialize assertions
            self.assertions = TestAssertions(self.agent, self.agent.state.history)
            
            # Use browser-use's run method directly
            # Pass the step_end_hook to agent.run
            result = await self.agent.run(on_step_end=step_end_hook)
            
            # Log successful completion
            self.reporter.log_action("Test completed", {
                "task": self.agent.task,
                "success": True,
                "steps": self.agent.state.n_steps,
                "verification_steps": self.assertions._extract_verification_steps(self.agent.task)
            })
            
            # Generate the test report
            self.reporter.generate_report()
            
            return result
            
        except Exception as e:
            # Log test failure with detailed error information
            self.reporter.log_action("Test failed", {
                "task": self.agent.task,
                "error": str(e),
                "steps": self.agent.state.n_steps if hasattr(self.agent.state, 'n_steps') else 0,
                "verification_steps": self.assertions._extract_verification_steps(self.agent.task) if self.assertions else []
            })
            
            # Generate the test report
            self.reporter.generate_report()
            
            # Re-raise the exception
            raise
        finally:
            # Ensure browser context is properly closed
            if hasattr(self, 'agent') and hasattr(self.agent, 'browser_context'):
                try:
                    await self.agent.browser_context.close()
                except Exception as e:
                    logger.warning(f"Error closing browser context: {e}")