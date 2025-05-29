# test_framework/agent/test_agent.py
import logging
from typing import Optional, Dict, Any, List
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentHistoryList, AgentStepInfo
from browser_use.controller.service import Controller
from browser_use.browser.session import BrowserSession
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
from langchain.schema import HumanMessage

logger = logging.getLogger("test_framework.agent.test_agent")

class TestAgent:
    """Agent for running browser tests."""
    
    def __init__(
        self,
        browser_session: Optional[BrowserSession] = None,
        llm_client: Optional[LLMClient] = None,
        settings: Optional[Dict[str, Any]] = None
    ):
        """Initialize test agent.
        
        Args:
            browser_session: Optional browser session for browser interactions
            llm_client: Optional LLM client for natural language processing
            settings: Optional settings for the agent
        """
        self.browser_session = browser_session or BrowserSession()
        self.llm_client = llm_client
        self.settings = settings or {}
        self.agent = Agent(
            task=self.settings.get('task', ''),
            browser_session=self.browser_session,
            llm=self.llm_client
        )
        self.validator = TestValidator()
        empty_history = AgentHistoryList(history=[])
        self.verification_assertions = VerificationAssertions(self.agent, empty_history)
        self.extraction_assertions = ExtractionAssertions(self.agent, empty_history)
        self.matching_assertions = MatchingAssertions(self.agent, empty_history)
        
    async def start(self):
        """Start the test agent."""
        await self.browser_session.start()
        
    async def stop(self):
        """Stop the test agent."""
        await self.browser_session.stop()
        
    async def run_test(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a test with the given data.
        
        Args:
            test_data: Test data containing steps and requirements
            
        Returns:
            Dict[str, Any]: Test results
        """
        try:
            # Start browser session if not already initialized
            if not self.browser_session.initialized:
                await self.start()
                
            # Run test steps
            results = []
            for step in test_data.get('steps', []):
                # Create step info for the agent
                step_info = AgentStepInfo(step_number=len(results), max_steps=len(test_data.get('steps', [])))
                
                # Handle navigation action directly
                if step.get('action') == 'navigate':
                    url = step.get('params', {}).get('url')
                    if url:
                        logger.info(f"Directly navigating to: {url}")
                        page = await self.browser_session.get_current_page()
                        await page.goto(url)
                        # Wait for navigation to complete
                        await page.wait_for_load_state('networkidle')
                        results.append([ActionResult(extracted_content=f"Navigated to {url}")])
                        continue
                
                # Add step data to agent's message manager for other actions
                if 'action' in step:
                    action_data = {
                        'action': step['action'],
                        'params': step.get('params', {})
                    }
                    self.agent._message_manager._add_message_with_tokens(
                        HumanMessage(content=f"Execute action: {json.dumps(action_data)}")
                    )
                
                # Execute the step
                await self.agent.step(step_info)
                
                # Get the result from the agent's history
                if self.agent.state.history.history:
                    results.append(self.agent.state.history.history[-1].result)
                
            # Validate results
            validation_results = []
            for requirement in test_data.get('requirements', []):
                # Get the last result for validation
                if not results:
                    validation_results.append({
                        'success': False,
                        'error': "No results available for validation"
                    })
                    if self.settings.get('assertion_mode') == 'hard':
                        return {
                            'success': False,
                            'error': "No results available for validation",
                            'results': results,
                            'validation_results': validation_results
                        }
                    continue
                    
                last_result = results[-1]
                
                # Use VerificationAssertions to verify the requirement
                verification_result = await self.verification_assertions.verify_requirement(
                    requirement=requirement,
                    step_number=len(results)
                )
                
                if verification_result.success:
                    logger.info(f"✅ Verification passed: {requirement}")
                else:
                    logger.error(f"❌ Verification failed: {requirement}")
                    logger.error(f"Error: {verification_result.message}")
                    if self.settings.get('assertion_mode') == 'hard':
                        validation_results.append({
                            'success': False,
                            'error': verification_result.message,
                            'metadata': verification_result.metadata
                        })
                        return {
                            'success': False,
                            'error': f"Hard assertion failed: {verification_result.message}",
                            'results': results,
                            'validation_results': validation_results
                        }
                
                validation_results.append({
                    'success': verification_result.success,
                    'error': verification_result.message if not verification_result.success else None,
                    'metadata': verification_result.metadata
                })
            
            # Log final result
            if all(r['success'] for r in validation_results):
                logger.info("All verifications passed successfully")
            else:
                logger.error("Some verifications failed")
                
            return {
                'success': all(r['success'] for r in validation_results),
                'results': results,
                'validation_results': validation_results
            }
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def verify_text(self, action_result: ActionResult, expected_text: str) -> AssertionResult:
        """Verify text content matches expected value."""
        return await self.verification_assertions.verify_text(action_result, expected_text)
        
    async def verify_link(self, action_result: ActionResult, expected_text: str, expected_href: Optional[str] = None) -> AssertionResult:
        """Verify link content and href match expected values."""
        return await self.verification_assertions.verify_link(action_result, expected_text, expected_href)
        
    async def verify_attribute(self, action_result: ActionResult, expected_value: str, attribute: str) -> AssertionResult:
        """Verify attribute value matches expected value."""
        return await self.verification_assertions.verify_attribute(action_result, expected_value, attribute)
        
    async def verify_list(self, action_result: ActionResult, expected_values: List[str]) -> AssertionResult:
        """Verify list of values matches expected values."""
        return await self.verification_assertions.verify_list(action_result, expected_values)
        
    async def verify_multi_value(self, action_result: ActionResult, expected_values: Dict[str, str]) -> AssertionResult:
        """Verify multiple values match expected values."""
        return await self.verification_assertions.verify_multi_value(action_result, expected_values)
