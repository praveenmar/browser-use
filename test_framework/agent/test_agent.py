# test_framework/agent/test_agent.py
import logging
from typing import Optional, Dict, Any, List
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.controller.service import Controller
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from test_framework.llm.llm_client import LLMClient
from test_framework.validation.assertions import TestAssertions, AssertionResult
from test_framework.validation.validator import TestValidator, ValidationMode
from playwright.async_api import Page
import re
import json

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
        logger.debug(f"Initializing TestAgent with task: {task}")
        self.task = task
        self.llm = llm or LLMClient("gemini-2.0-flash-exp")
        self.controller = controller or Controller()
        self.browser = browser or Browser()
        self.browser_context = browser_context or BrowserContext()
        
        # Store original requirements
        self.original_requirements = self._extract_requirements(task)
        logger.debug(f"Extracted requirements: {self.original_requirements}")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    def _extract_requirements(self, task: str) -> List[str]:
        """Extract original requirements from task description"""
        logger.debug(f"Extracting requirements from task: {task}")
        requirements = []
        for line in task.split('\n'):
            if 'assert' in line.lower() or 'verify' in line.lower():
                requirements.append(line.strip())
                logger.debug(f"Found requirement: {line.strip()}")
        return requirements

    def _extract_navigation_steps(self, task: str) -> List[str]:
        """Extract navigation steps from task description"""
        logger.debug(f"Extracting navigation steps from task: {task}")
        steps = []
        for line in task.split('\n'):
            if 'step' in line.lower() and ('navigate' in line.lower() or 'go to' in line.lower()):
                steps.append(line.strip())
                logger.debug(f"Found navigation step: {line.strip()}")
        return steps

    async def run(self) -> ActionResult:
        """Run the test"""
        try:
            logger.info("Starting test execution...")
            logger.info(f"Task:\n{self.task}")

            # Initialize agent for navigation
            logger.debug("Initializing agent for navigation")
            self.agent = Agent(
                task=self.task,
                llm=self.llm,
                controller=self.controller,
                browser=self.browser,
                browser_context=self.browser_context,
            )

            # Run navigation steps
            logger.debug("Running navigation steps")
            result: AgentHistoryList = await self.agent.run()
            logger.debug(f"Navigation result: {result}")
            
            # Get the current page and ensure it's a valid Page instance
            logger.debug("Getting current page")
            page = await self.browser_context.get_current_page()
            if not isinstance(page, Page):
                raise ValueError(f"Expected Playwright Page object, got {type(page)}")
            
            # Create assertions instance for verification
            logger.debug("Creating assertions instance")
            assertions = TestAssertions(self.agent, result)
            assertions.page = page
            
            # Verify all requirements
            logger.debug("Starting requirement verification")
            verification_results = []
            for requirement in self.original_requirements:
                logger.debug(f"Verifying requirement: {requirement}")
                result = await assertions._verify_condition(requirement)
                logger.debug(f"Verification result: {result}")
                verification_results.append(result)
                
                # Log verification result
                if result.success:
                    logger.info(f"✅ Verification passed: {requirement}")
                    logger.debug(f"Verification details: {result.metadata}")
                else:
                    logger.error(f"❌ Verification failed: {requirement}")
                    logger.error(f"Error: {result.message}")
                    logger.debug(f"Verification details: {result.metadata}")
                    return ActionResult(
                        success=False,
                        message=f"Verification failed: {result.message}",
                        data={
                            "requirement": requirement,
                            "result": {
                                "success": result.success,
                                "error_code": result.error_code,
                                "message": result.message,
                                "metadata": result.metadata
                            }
                        }
                    )
            
            # Return success if all verifications passed
            logger.info("All verifications passed successfully")
            return ActionResult(
                success=True,
                message="All requirements verified successfully",
                data={
                    "verification_results": [
                        {
                            "success": r.success,
                            "error_code": r.error_code,
                            "message": r.message,
                            "metadata": r.metadata
                        } for r in verification_results
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}", exc_info=True)
            return ActionResult(
                success=False,
                message=f"Test execution failed: {str(e)}",
                data={}
            )
        finally:
            # Clean up
            if self.browser_context:
                logger.debug("Cleaning up browser context")
                await self.browser_context.close()
