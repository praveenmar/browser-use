# test_framework/agent/test_agent.py
import logging
from typing import Optional, Dict, Any
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.controller.service import Controller
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from test_framework.llm.llm_client import LLMClient
from test_framework.validation.assertions import TestAssertions
from patchright.async_api import Page

logger = logging.getLogger("test_framework.agent.test_agent")

class TestAgent:
    def __init__(
        self,
        task: str,
        llm: Optional[LLMClient] = None,
        controller: Optional[Controller] = None,
        browser: Optional[Browser] = None,
        browser_context: Optional[BrowserContext] = None,
    ):
        self.task = task
        self.llm = llm or LLMClient("gemini-2.0-flash-exp")
        self.controller = controller or Controller()
        self.browser = browser or Browser()
        self.browser_context = browser_context or BrowserContext()
        
        # Register assertion actions
        TestAssertions.register_actions(self.controller)
        
        # Store original requirements
        self.original_requirements = self._extract_requirements(task)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    def _extract_requirements(self, task: str) -> list:
        """Extract original requirements from task description"""
        requirements = []
        for line in task.split('\n'):
            if 'assert' in line.lower() or 'verify' in line.lower():
                requirements.append(line.strip())
        return requirements

    async def run(self) -> ActionResult:
        """Run the test"""
        try:
            logger.info("Starting test execution...")
            logger.info(f"Task:\n{self.task}")

            # Initialize agent
            self.agent = Agent(
                task=self.task,
                llm=self.llm.model,  # Pass the model from LLMClient
                controller=self.controller,
                browser=self.browser,
                browser_context=self.browser_context,
            )

            # Run the test
            result: AgentHistoryList = await self.agent.run()
            
            # Get the current page and ensure it's a valid Page instance
            page = await self.browser_context.get_current_page()
            if not isinstance(page, Page):
                raise ValueError(f"Expected Playwright Page object, got {type(page)}")
            
            # Create assertions instance with the page
            assertions = TestAssertions(self.agent, result)
            assertions.page = page
            
            # Verify all original requirements were met
            for requirement in self.original_requirements:
                await assertions._verify_condition(requirement)
            
            # Return success if all verifications passed
            return ActionResult(
                success=True,
                message="All requirements verified successfully",
                data=result.model_dump()
            )

        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Test execution failed: {str(e)}",
                data={}
            )
        finally:
            # Clean up
            if self.browser_context:
                await self.browser_context.close()