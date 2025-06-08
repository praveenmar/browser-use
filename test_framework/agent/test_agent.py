# test_framework/agent/test_agent.py
import logging
from typing import Optional, Dict, Any, List, Tuple
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
from ..validation.utils.visibility_utils import is_element_visible_by_handle, is_element_in_viewport

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
        # Ensure all assertions are case-sensitive by default
        self.verification_assertions.set_case_sensitive(True)
        self.extraction_assertions.set_case_sensitive(True)
        self.matching_assertions.set_case_sensitive(True)
        
        # Track assertion state
        self._assertion_state = {
            'pending_assertions': [],
            'current_step': 0,
            'last_assertion_time': 0
        }
        
    async def start(self):
        """Start the test agent."""
        try:
            await self.browser_session.start()
        except Exception as e:
            logger.error(f"Error starting browser session: {e}")
            raise
        
    async def stop(self):
        """Stop the test agent."""
        try:
            if hasattr(self.agent, 'clear_memory'):
                await self.agent.clear_memory()
            await self.browser_session.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
    async def wait_for_dom_stability(self):
        """Wait for the DOM to be stable before running assertions."""
        try:
            if hasattr(self.agent, 'browser_session'):
                # Ensure browser session is initialized
                if not self.agent.browser_session.initialized:
                    await self.agent.browser_session.start()
                
                # Get the page
                page = await self.agent.browser_session.get_current_page()
                if page:
                    logger.debug("Waiting for DOM stability...")
                    await page.wait_for_load_state('networkidle')
                    logger.debug("DOM is stable")
        except Exception as e:
            logger.warning(f"Error waiting for DOM stability: {e}")
        
    async def _handle_assertion_step(self, step: str) -> Tuple[bool, Optional[str]]:
        """Handle an assertion step.
        
        Args:
            step: The assertion step to handle
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            # Wait for DOM stability
            await self.wait_for_dom_stability()
            
            # Clear extraction cache
            if hasattr(self.extraction_assertions, '_extraction_cache'):
                self.extraction_assertions._extraction_cache.clear()
                
            # Extract expected text
            expected_text = None
            patterns = [
                r"assert that the text '([^']*)' is present",
                r"assert thst the text '([^']*)' is present",
                r"assert that '([^']*)' is present",
                r"assert thst '([^']*)' is present",
                r"verify that the text '([^']*)' is present",
                r"verify thst the text '([^']*)' is present",
                r"verify that '([^']*)' is present",
                r"verify thst '([^']*)' is present",
                r"assert that the text '([^']*)'",
                r"assert thst the text '([^']*)'",
                r"assert that '([^']*)'",
                r"assert thst '([^']*)'",
                r"verify that the text '([^']*)'",
                r"verify thst the text '([^']*)'",
                r"verify that '([^']*)'",
                r"verify thst '([^']*)'"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, step, re.IGNORECASE)
                if match:
                    expected_text = match.group(1)
                    break
                    
            if not expected_text:
                return False, "No text found in assertion step"
                
            # Add to pending assertions
            self._assertion_state['pending_assertions'].append({
                'text': expected_text,
                'step': step,
                'timestamp': time.time()
            })
            
            logger.info(f"🔍 Starting verification for text: '{expected_text}'")
            
            # Get the page
            page = await self.agent.browser_session.get_current_page()
            if not page:
                logger.error("❌ No browser page available")
                return False, "No browser page available"
                
            # First check if text exists in DOM
            logger.debug("📄 Checking DOM content...")
            content = await page.content()
            if expected_text not in content:
                logger.error(f"❌ Text '{expected_text}' not found in DOM")
                return False, f"Text '{expected_text}' not found in DOM"
            logger.debug(f"✅ Text '{expected_text}' found in DOM")
            
            # Try to find element with text using browser session
            try:
                element = await self.browser_session.get_locate_element_by_text(expected_text, nth=0)
                if element:
                    logger.debug("✅ Found element with text")
                    
                    # Get element position
                    element_box = await element.bounding_box()
                    if element_box:
                        logger.debug(f"📍 Element position - X: {element_box['x']}, Y: {element_box['y']}")
                    
                    # Check visibility using browser session
                    logger.debug("👁️ Checking element visibility...")
                    is_visible = await self.browser_session.is_visible_by_handle(element)
                    logger.debug(f"👁️ Element visibility: {is_visible}")
                    
                    if is_visible:
                        # Double check if element is actually in viewport
                        is_in_viewport = await page.evaluate("""
                            (element) => {
                                const rect = element.getBoundingClientRect();
                                return (
                                    rect.top >= 0 &&
                                    rect.left >= 0 &&
                                    rect.bottom <= window.innerHeight &&
                                    rect.right <= window.innerWidth
                                );
                            }
                        """, element)
                        
                        if is_in_viewport:
                            logger.debug("✅ Text is visible and in viewport")
                            return True, None
                        else:
                            logger.warning(f"⚠️ Text '{expected_text}' found but not in viewport")
                            return False, f"Text '{expected_text}' found but not in viewport"
                    else:
                        logger.warning(f"⚠️ Text '{expected_text}' found but not visible")
                        return False, f"Text '{expected_text}' found but not visible"
                else:
                    logger.warning(f"⚠️ Could not find element with text '{expected_text}'")
                    return False, f"Could not find element with text '{expected_text}'"
            except Exception as e:
                logger.warning(f"⚠️ Error checking text visibility: {e}")
                return False, f"Error checking text visibility: {e}"
            
            # If text is not visible, try scrolling through the page
            logger.debug("🔄 Starting scroll search for text...")
            viewport_height = await page.evaluate("window.innerHeight")
            current_scroll = await page.evaluate("window.scrollY")
            max_scroll = await page.evaluate("document.body.scrollHeight")
            
            # First scroll to top
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)
            
            # Scroll in smaller increments for more thorough search
            scroll_increment = viewport_height // 2  # Half viewport height
            while current_scroll < max_scroll:
                try:
                    # Scroll down by increment
                    await page.evaluate(f"window.scrollBy(0, {scroll_increment})")
                    await asyncio.sleep(0.5)  # Wait for scroll to complete
                    current_scroll = await page.evaluate("window.scrollY")
                    
                    # Try to find element again after scroll
                    element = await self.browser_session.get_locate_element_by_text(expected_text, nth=0)
                    if element:
                        logger.debug("✅ Found element with text after scroll")
                        
                        # Get element position after scroll
                        element_box = await element.bounding_box()
                        if element_box:
                            logger.debug(f"📍 Element position after scroll - X: {element_box['x']}, Y: {element_box['y']}")
                        
                        # Check visibility using browser session
                        logger.debug("👁️ Checking element visibility after scroll...")
                        is_visible = await self.browser_session.is_visible_by_handle(element)
                        logger.debug(f"👁️ Element visibility after scroll: {is_visible}")
                        
                        if is_visible:
                            # Double check if element is actually in viewport after scroll
                            is_in_viewport = await page.evaluate("""
                                (element) => {
                                    const rect = element.getBoundingClientRect();
                                    return (
                                        rect.top >= 0 &&
                                        rect.left >= 0 &&
                                        rect.bottom <= window.innerHeight &&
                                        rect.right <= window.innerWidth
                                    );
                                }
                            """, element)
                            
                            if is_in_viewport:
                                logger.debug("✅ Text is visible and in viewport after scrolling")
                                return True, None
                            else:
                                logger.warning(f"⚠️ Text '{expected_text}' found but not in viewport after scroll")
                        else:
                            logger.warning(f"⚠️ Text '{expected_text}' found but not visible after scroll")
                    else:
                        logger.debug(f"ℹ️ Text '{expected_text}' not found in current viewport after scroll")
                        
                except Exception as e:
                    logger.error(f"❌ Error checking viewport: {e}")
                    
            # If we've scrolled through the entire page and haven't found the text
            logger.error(f"❌ Text '{expected_text}' found in DOM but not visible in any viewport")
            return False, f"Text '{expected_text}' found in DOM but not visible in any viewport"
            
        except Exception as e:
            logger.error(f"❌ Error in verification: {e}")
            return False, str(e)
            
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
            current_step = 0
            
            for step in task_steps:
                step = step.strip()
                if not step or step.startswith('@agent='):
                    continue
                    
                current_step += 1
                logger.info(f"Executing step {current_step}: {step}")
                
                # Create step info for the agent
                step_info = AgentStepInfo(step_number=current_step, max_steps=len(task_steps))
                
                # Add step to agent's message manager
                self.agent._message_manager._add_message_with_tokens(
                    HumanMessage(content=f"Execute step {current_step}: {step}")
                )
                
                # Execute the step
                step_result = await self.agent.step(step_info)
                
                # Wait for step to complete
                if step_result:
                    await asyncio.sleep(1)  # Give time for any animations or state changes
                
                # Check if this step requires validation
                if 'assert' in step.lower():
                    # Wait for DOM stability before validation
                    try:
                        await self.wait_for_dom_stability()
                    except Exception as e:
                        logger.warning(f"DOM stability timeout: {e}")
                        # Continue anyway since we have retry logic in verification
                    
                    # Perform validation
                    verification_success = False
                    last_error = None
                    
                    # Extract expected text from assertion
                    expected_text = None
                    
                    # Try different patterns for assertion text
                    patterns = [
                        r"assert that the text '([^']*)' is present",
                        r"assert thst the text '([^']*)' is present",
                        r"assert that '([^']*)' is present",
                        r"assert thst '([^']*)' is present",
                        r"verify that the text '([^']*)' is present",
                        r"verify thst the text '([^']*)' is present",
                        r"verify that '([^']*)' is present",
                        r"verify thst '([^']*)' is present"
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, step.lower())
                        if match:
                            expected_text = match.group(1)
                            break
                            
                    if expected_text:
                        # Try verification assertions first
                        verification_success, last_error = await self.verification_assertions.verify_text(
                            expected_text,
                            step
                        )
                        
                        if verification_success:
                            logger.info(f"✅ Verification passed for text: {expected_text}")
                            validation_results.append({
                                'step': current_step,
                                'success': True,
                                'text': expected_text
                            })
                        else:
                            logger.error(f"❌ Verification failed for text: {expected_text}")
                            validation_results.append({
                                'step': current_step,
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
                
                # Update agent's evaluation state to success for the current step
                if hasattr(self.agent, 'state') and hasattr(self.agent.state, 'evaluation'):
                    self.agent.state.evaluation = {
                        'status': 'success',
                        'message': f'Successfully completed step {current_step}'
                    }
                    
                # Add step result
                results.append({
                    'step': current_step,
                    'success': True
                })
                
                # Wait for any animations or state changes to complete
                await asyncio.sleep(1)
            
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
