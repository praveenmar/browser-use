# test_framework/agent/test_agent.py
import logging
from typing import Optional, Dict, Any, List
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentHistoryList, AgentStepInfo
from browser_use.controller.service import Controller
from browser_use.browser.session import BrowserSession
from browser_use.agent.prompts import SystemPrompt
from test_framework.validation.assertions import (
    VerificationAssertions,
    ExtractionAssertions,
    MatchingAssertions,
    AssertionResult
)
from test_framework.validation.validator import TestValidator, ValidationMode
from test_framework.output_processor import OutputProcessor
from test_framework.agent.prompts import TestAgentPrompt
from playwright.async_api import Page
import re
import json
from langchain.schema import HumanMessage
import asyncio
import time
from datetime import datetime

logger = logging.getLogger("test_framework.agent.test_agent")

class TestAgent:
    """Agent for running browser tests."""
    
    def __init__(
        self,
        browser_session: Optional[BrowserSession] = None,
        llm_client: Optional[Any] = None,
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
        
        # Initialize test agent prompt
        self.test_prompt = TestAgentPrompt()
        
        self.agent = Agent(
            task="",
            llm=self.llm_client,
            browser_session=self.browser_session,
            override_system_message=self.test_prompt.get_system_message().content,
            max_actions_per_step=10
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
        
    async def wait_for_dom_stability(self):
        """Wait for the DOM to be stable before running assertions."""
        try:
            if hasattr(self.agent, 'browser_session') and self.agent.browser_session.page:
                logger.debug("Waiting for DOM stability...")
                await self.agent.browser_session.page.wait_for_load_state('networkidle')
                logger.debug("DOM is stable")
        except Exception as e:
            logger.warning(f"Error waiting for DOM stability: {e}")
        
    async def run_test(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a test with the given data.
        
        Args:
            test_data: Test data containing steps and requirements
            
        Returns:
            Dict[str, Any]: Test results
        """
        start_time = time.time()
        start_datetime = datetime.now()
        
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
                
                # Check if this step requires validation
                if 'assert' in step.lower():
                    # Wait for DOM stability before validation
                    await self.wait_for_dom_stability()
                    
                    # Clear extraction cache before validation
                    if hasattr(self.extraction_assertions, '_extraction_cache'):
                        self.extraction_assertions._extraction_cache.clear()
                    
                    # Perform validation
                    verification_success = True
                    last_error = None
                    
                    # Extract expected text from assertion
                    expected_text = None
                    
                    # Try different patterns for assertion text
                    patterns = [
                        r"assert that the text '([^']*)'",  # Standard format
                        r"assert thst the text '([^']*)'",  # Common typo
                        r"assert that '([^']*)'",          # Short format
                        r"assert thst '([^']*)'",          # Short format with typo
                        r"under the text [^']* assert thst '([^']*)'",  # Under text format
                        r"assert that the ([^']*) is present",  # Is present format
                        r"assert thst the ([^']*) is present"   # Is present format with typo
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, step.lower())
                        if match:
                            expected_text = match.group(1)
                            break
                            
                    if expected_text:
                        # Verify the text
                        verification_success, last_error = await self.verification_assertions.verify_text(
                            expected_text,
                            step
                        )
                        
                        if verification_success:
                            logger.info(f"✅ Verification passed for text: {expected_text}")
                            validation_results.append({
                                'step': len(results),
                                'success': True,
                                'text': expected_text
                            })
                        else:
                            logger.error(f"❌ Verification failed for text: {expected_text}")
                            validation_results.append({
                                'step': len(results),
                                'success': False,
                                'text': expected_text,
                                'error': last_error
                            })
                            
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
                    
                # Add step result
                results.append({
                    'step': len(results),
                    'success': True
                })
            
            # Calculate duration
            end_time = time.time()
            end_datetime = datetime.now()
            duration_seconds = end_time - start_time
            
            # Log final result
            total_assertions = len(validation_results)
            passed_assertions = sum(1 for r in validation_results if r.get('success', False))
            
            if passed_assertions == total_assertions:
                logger.info(f"All verifications passed successfully ({passed_assertions}/{total_assertions})")
            else:
                logger.error(f"Some verifications failed ({passed_assertions}/{total_assertions})")
                
            logger.info(f"Test completed in {duration_seconds:.2f} seconds")
                
            return {
                'success': all(result.get('success', True) for result in validation_results),
                'results': results,
                'validation_results': validation_results,
                'total_assertions': total_assertions,
                'passed_assertions': passed_assertions,
                'duration': {
                    'seconds': duration_seconds,
                    'start_time': start_datetime.isoformat(),
                    'end_time': end_datetime.isoformat()
                }
            }
            
        except Exception as e:
            end_time = time.time()
            end_datetime = datetime.now()
            duration_seconds = end_time - start_time
            
            logger.error(f"Error running test: {str(e)}")
            logger.error(f"Test failed after {duration_seconds:.2f} seconds")
            
            return {
                'success': False,
                'error': str(e),
                'duration': {
                    'seconds': duration_seconds,
                    'start_time': start_datetime.isoformat(),
                    'end_time': end_datetime.isoformat()
                }
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
