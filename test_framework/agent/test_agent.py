# test_framework/agent/test_agent.py
import logging
from typing import Optional, Dict, Any, List
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.controller.service import Controller
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from test_framework.llm.llm_client import LLMClient
from test_framework.validation.assertions import (
    VerificationAssertions,
    ExtractionAssertions,
    MatchingAssertions,
    AssertionResult
)
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
        assertion_mode: str = "hard"  # Default to hard assertions
    ):
        """Initialize test agent.
        
        Args:
            task: The task description
            llm: Language model for agent
            controller: Controller for actions
            browser: Browser instance
            browser_context: Browser context
            assertion_mode: Mode of assertions - "hard" or "soft"
        """
        logger.debug(f"Initializing TestAgent with task: {task}")
        self.task = task
        self.llm = llm or LLMClient("gemini-2.0-flash-exp")
        self.controller = controller or Controller()
        self.browser = browser or Browser()
        self.browser_context = browser_context or BrowserContext()
        self.assertion_mode = assertion_mode.lower()
        if self.assertion_mode not in ["hard", "soft"]:
            raise ValueError("assertion_mode must be either 'hard' or 'soft'")
        
        # Store original requirements
        self.original_requirements = self._extract_requirements(task)
        logger.debug(f"Extracted requirements: {self.original_requirements}")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize agent
        self.agent = Agent(
            task=task,
            llm=self.llm,
            controller=self.controller,
            browser=self.browser,
            browser_context=self.browser_context
        )
        
        # Initialize assertions using VerificationAssertions directly
        self.assertions = VerificationAssertions(self.agent, self.agent.state.history)
        
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
            
            # Run the agent
            result = await self.agent.run()
            
            # Extract requirements from task
            requirements = self._extract_requirements(self.task)
            
            # Verify each requirement
            verification_results = []
            for i, requirement in enumerate(requirements):
                try:
                    verification_result = await self.assertions.verify_requirement(requirement, i)
                    verification_results.append(verification_result)
                    
                    # Log verification result
                    if verification_result.success:
                        logger.info(f"✅ Verification passed: {requirement}")
                    else:
                        logger.error(f"❌ Verification failed: {requirement}")
                        logger.error(f"Error: {verification_result.message}")
                        
                        # If hard assertions, stop on first failure
                        if self.assertion_mode == "hard":
                            return ActionResult(
                                success=False,
                                error=f"Failed to verify requirement: {requirement}",
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
                    logger.error(f"Error verifying requirement {requirement}: {str(e)}")
                    if self.assertion_mode == "hard":
                        raise
                    
            # Check if all verifications passed
            all_passed = all(r.success for r in verification_results)
            if all_passed:
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
            else:
                failed_requirements = [r for r in verification_results if not r.success]
                return ActionResult(
                    success=False,
                    error=f"Some requirements failed verification: {[r.message for r in failed_requirements]}",
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
                error=f"Test execution failed: {str(e)}",
                data={}
            )
        finally:
            # Clean up
            if self.browser_context:
                logger.debug("Cleaning up browser context")
                await self.browser_context.close()
