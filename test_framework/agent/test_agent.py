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
from test_framework.output_processor import OutputProcessor
from playwright.async_api import Page
import re
import json
from langchain.schema import HumanMessage
import asyncio

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
        
        # Log the task before agent initialization
        logger.info("Task:")
        logger.info(self.settings.get('task', ''))
        
        self.agent = Agent(
            task=self.settings.get('task', ''),
            browser_session=self.browser_session,
            llm=self.llm_client,
            enable_memory=True,  # Enable memory for better context
            use_vision=True,     # Enable vision for better page understanding
            max_failures=3,      # Allow up to 3 failures before giving up
            retry_delay=10,      # Wait 10 seconds between retries
            max_actions_per_step=10  # Allow up to 10 actions per step
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
            validation_results = []
            
            # Process the task steps
            task_steps = self.settings.get('task', '').split('\n')
            for step in task_steps:
                step = step.strip()
                if not step or step.startswith('@agent='):
                    continue
                    
                # Create step info for the agent
                step_info = AgentStepInfo(step_number=len(results), max_steps=len(task_steps))
                
                # Add step to agent's message manager
                self.agent._message_manager._add_message_with_tokens(
                    HumanMessage(content=f"Execute step: {step}")
                )
                
                # Execute the step
                await self.agent.step(step_info)
                
                # Get the result from the agent's history
                if self.agent.state.history.history:
                    result = self.agent.state.history.history[-1].result
                    
                    # Process the result through our output processor
                    if isinstance(result, dict) and 'extracted_content' in result:
                        result['extracted_content'] = OutputProcessor.process_extraction_result(
                            result['extracted_content']
                        )
                    elif isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict) and 'extracted_content' in item:
                                item['extracted_content'] = OutputProcessor.process_extraction_result(
                                    item['extracted_content']
                                )
                    
                    results.append(result)
                    
                    # If this was an assertion step, use our assertion module
                    if 'assert' in step.lower():
                        # Extract all expected texts from the assertion
                        expected_texts = re.findall(r"'([^']*)'", step)
                        if expected_texts:
                            # Use our verification assertions for each expected text
                            for expected_text in expected_texts:
                                max_retries = 3
                                retry_count = 0
                                verification_success = False
                                last_error = None
                                
                                while retry_count < max_retries:
                                    verification_result = await self.verification_assertions.verify_requirement(
                                        requirement=step,
                                        step_number=len(results)
                                    )
                                    
                                    if verification_result.success:
                                        logger.info(f"✅ Verification passed for text: {expected_text}")
                                        verification_success = True
                                        validation_results.append({
                                            'success': True,
                                            'error': None,
                                            'metadata': verification_result.metadata
                                        })
                                        break
                                        
                                    retry_count += 1
                                    last_error = verification_result.message
                                    
                                if not verification_success:
                                    logger.error(f"❌ Verification failed for text: {expected_text}")
                                    if self.settings.get('assertion_mode') == 'hard':
                                        return {
                                            'success': False,
                                            'error': f"Verification failed for text: {expected_text} - {last_error}",
                                            'results': results,
                                            'validation_results': validation_results
                                        }
                    # Clear agent's memory after each step to prevent getting stuck
                    if hasattr(self.agent, 'clear_memory'):
                        await self.agent.clear_memory()
                        
                    # Update agent's evaluation state to success for the current step
                    if hasattr(self.agent, 'state') and hasattr(self.agent.state, 'evaluation'):
                        self.agent.state.evaluation = {
                            'status': 'success',
                            'message': f'Successfully completed step {len(results)}'
                        }
            
            # Log final result
            if all(result.get('success', True) for result in validation_results):
                logger.info("All verifications passed successfully")
            else:
                logger.error("Some verifications failed")
                
            return {
                'success': all(result.get('success', True) for result in validation_results),
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
