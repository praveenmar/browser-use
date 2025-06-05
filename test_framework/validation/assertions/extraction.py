"""
Extraction module for test assertions.
Handles text extraction and processing functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from urllib.parse import urljoin, urlparse
import re
import time
import asyncio

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent
from browser_use.browser.session import BrowserSession

from .base import BaseAssertions, AssertionResult
from .matching import MatchingAssertions

logger = logging.getLogger("test_framework.validation.assertions")

class ExtractionAssertions(BaseAssertions):
    """Assertions for extracting content from pages."""
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize extraction assertions.
        
        Args:
            agent: The agent instance
            result: The result history list
        """
        super().__init__(agent, result)
        self._matching = MatchingAssertions(agent, result)
        self._extraction_cache = {}
        self._processed_extractions = set()
        self.browser_session = None
        self._last_extraction_time = 0
        self._extraction_timeout = 5  # seconds
        self._assertion_states = {}  # Track assertion states
        self._fuzzy_match_threshold = 0.8  # Configurable threshold
        self._max_scroll_attempts = 3  # Maximum number of scroll attempts
        self._page_state = {}  # Track page state
        self._validation_context = {}  # Track validation context
        self._step_types = {}  # Track step types
        self._case_sensitive = True  # Default to case-sensitive
        
        # Initialize browser session
        if hasattr(agent, 'browser_session'):
            self.browser_session = agent.browser_session
            logger.debug(f"Initialized browser session: {type(self.browser_session)}")
            
        logger.debug(f"Initialized ExtractionAssertions with agent type: {type(agent)}")
        
    def clear_cache(self):
        """Clear the extraction cache and processed extractions."""
        self._extraction_cache.clear()
        self._processed_extractions.clear()
        self._last_extraction_time = 0
        self._assertion_states.clear()  # Clear assertion states
        self._page_state.clear()  # Clear page state
        self._validation_context.clear()  # Clear validation context
        self._step_types.clear()  # Clear step types
        
    def _determine_step_type(self, step: str) -> str:
        """Determine the type of step.
        
        Args:
            step: The step string
            
        Returns:
            str: Step type ('action', 'assertion', or 'unknown')
        """
        step = step.lower()
        if 'assert' in step:
            return 'assertion'
        elif any(keyword in step for keyword in ['click', 'navigate', 'scroll', 'type', 'select']):
            return 'action'
        return 'unknown'
        
    def _log_step_type(self, step: str, step_number: int):
        """Log the step type.
        
        Args:
            step: The step string
            step_number: The step number
        """
        step_type = self._determine_step_type(step)
        self._step_types[step_number] = step_type
        logger.debug(f"Step {step_number} identified as type: {step_type}")
        
        # Add step type to agent's memory if available
        if hasattr(self.agent, 'state') and hasattr(self.agent.state, 'memory'):
            if not hasattr(self.agent.state.memory, 'step_types'):
                self.agent.state.memory.step_types = {}
            self.agent.state.memory.step_types[step_number] = step_type
            
    def _was_assertion_successful(self, text: str) -> bool:
        """Check if an assertion was previously successful.
        
        Args:
            text: The text that was asserted
            
        Returns:
            bool: True if assertion was successful
        """
        # Only use assertion states if explicitly enabled
        try:
            if hasattr(self.agent, 'settings'):
                # Check if settings has use_assertion_states attribute
                if hasattr(self.agent.settings, 'use_assertion_states'):
                    return self._assertion_states.get(text, False)
                # Fallback to checking if settings is a dict
                elif isinstance(self.agent.settings, dict):
                    return self._assertion_states.get(text, False)
        except Exception as e:
            logger.debug(f"Error checking assertion state: {e}")
        return False
        
    def _mark_assertion_successful(self, text: str):
        """Mark an assertion as successful.
        
        Args:
            text: The text that was successfully asserted
        """
        # Only track assertion states if explicitly enabled
        try:
            if hasattr(self.agent, 'settings'):
                # Check if settings has use_assertion_states attribute
                if hasattr(self.agent.settings, 'use_assertion_states'):
                    self._assertion_states[text] = True
                # Fallback to checking if settings is a dict
                elif isinstance(self.agent.settings, dict):
                    self._assertion_states[text] = True
        except Exception as e:
            logger.debug(f"Error marking assertion state: {e}")
            
    async def _get_browser_page(self):
        """Get the browser page safely.
        
        Returns:
            Optional[Page]: The browser page if available, None otherwise
        """
        try:
            # First try to get page from agent's browser session
            if (hasattr(self.agent, 'browser_session') and 
                self.agent.browser_session):
                try:
                    page = await self.agent.browser_session.get_current_page()
                    logger.debug("Got page from agent's browser session")
                    return page
                except Exception as e:
                    logger.warning(f"Error accessing agent's browser page: {e}")
                    
            # Then try local browser session
            if self.browser_session:
                try:
                    page = await self.browser_session.get_current_page()
                    logger.debug("Got page from local browser session")
                    return page
                except Exception as e:
                    logger.warning(f"Error accessing local browser page: {e}")
                    
            logger.debug("No browser page available")
            return None
        except Exception as e:
            logger.warning(f"Error getting browser page: {e}")
            return None
            
    async def extract_text(self, text: str, goal: str = None) -> Dict[str, Any]:
        """Extract text from the current page.
        
        Args:
            text: The text to extract
            goal: Optional goal for the extraction
            
        Returns:
            Dict[str, Any]: Extraction result
        """
        logger.debug(f"Starting text extraction for: {text}")
        
        # Check if we have a cached result
        cache_key = f"{text}:{goal}" if goal else text
        if cache_key in self._extraction_cache:
            logger.debug("Using cached extraction result")
            return self._extraction_cache[cache_key]
            
        # Wait for DOM stability before extraction
        try:
            page = await self._get_browser_page()
            if page:
                logger.debug("Waiting for DOM stability before extraction...")
                await page.wait_for_load_state('networkidle')
                logger.debug("DOM is stable, proceeding with extraction")
            else:
                logger.warning("No browser page available for DOM stability check")
        except Exception as e:
            logger.warning(f"Error waiting for DOM stability: {e}")
            
        # Perform extraction
        try:
            # First try to get extraction from agent's controller
            if hasattr(self.agent, 'controller'):
                logger.debug("Attempting controller-based extraction")
                try:
                    # Create extraction action
                    from browser_use.agent.views import ActionModel
                    class ExtractContentActionModel(ActionModel):
                        extract_content: dict | None = None
                        
                    action_data = {
                        "extract_content": {
                            "goal": f"Find the text: {text}",
                            "should_strip_link_urls": True,
                            "include_element_attributes": True,
                            "include_xpath": True,
                            "include_aria_labels": True,
                            "include_visible_text": True
                        }
                    }
                    
                    action = ExtractContentActionModel(**action_data)
                    
                    # Use controller to extract content
                    page = await self._get_browser_page()
                    if page:
                        logger.debug("Using controller to extract content")
                        result = await self.agent.controller.act(
                            action, 
                            self.browser_session,
                            page_extraction_llm=getattr(self.agent, 'settings', None)
                        )
                        
                        if result and hasattr(result, 'extracted_content'):
                            logger.debug("Got extraction result from controller")
                            content = result.extracted_content
                            if isinstance(content, dict):
                                # Check if content indicates presence
                                if content.get('present', False):
                                    logger.debug("Text found via controller")
                                    return {
                                        'extracted_content': text,
                                        'text': text,
                                        'goal': goal,
                                        'timestamp': time.time(),
                                        'present': True
                                    }
                                logger.debug("Text not found via controller")
                                return {
                                    'extracted_content': content.get('text', ''),
                                    'text': text,
                                    'goal': goal,
                                    'timestamp': time.time(),
                                    'present': False
                                }
                    else:
                        logger.warning("No browser page available for controller extraction")
                except Exception as e:
                    logger.debug(f"Controller extraction failed: {e}")
            
            # Try to get page content directly from browser session
            page = await self._get_browser_page()
            if page:
                logger.debug("Attempting direct page content extraction")
                # Get page content
                content = await page.content()
                
                # Create extraction result
                result = {
                    'extracted_content': content,
                    'text': text,
                    'goal': goal,
                    'timestamp': time.time()
                }
                
                if result:
                    logger.debug("Got page content successfully")
                    self._extraction_cache[cache_key] = result
                    self._last_extraction_time = time.time()
                return result
            else:
                logger.warning("No browser page available for direct extraction")
                
            # Fallback to agent's extract_text if available
            if hasattr(self.agent, 'extract_text'):
                logger.debug("Attempting agent's extract_text method")
                result = await self.agent.extract_text(text, goal)
                if result:
                    logger.debug("Got extraction result from agent")
                    self._extraction_cache[cache_key] = result
                    self._last_extraction_time = time.time()
                return result
                
            logger.error("No extraction method available")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return None
            
    async def _get_page_state(self) -> Dict[str, Any]:
        """Get current page state.
        
        Returns:
            Dict[str, Any]: Current page state
        """
        try:
            page = await self._get_browser_page()
            if page:
                return {
                    'url': page.url,
                    'title': await page.title(),
                    'viewport': page.viewport_size,
                    'timestamp': time.time()
                }
        except Exception as e:
            logger.warning(f"Error getting page state: {e}")
        return {}
        
    async def _is_page_state_changed(self) -> bool:
        """Check if page state has changed since last validation.
        
        Returns:
            bool: True if page state has changed
        """
        current_state = await self._get_page_state()
        if not current_state:
            return True
            
        last_state = self._page_state.get('last_state', {})
        if not last_state:
            return True
            
        # Check if URL or title has changed
        if (current_state.get('url') != last_state.get('url') or
            current_state.get('title') != last_state.get('title')):
            return True
            
        # Check if viewport has changed
        if current_state.get('viewport') != last_state.get('viewport'):
            return True
            
        return False
        
    async def _update_validation_context(self, text: str, success: bool, content: str):
        """Update validation context with latest result.
        
        Args:
            text: The text that was validated
            success: Whether validation was successful
            content: The content that was validated
        """
        self._validation_context[text] = {
            'success': success,
            'content': content,
            'timestamp': time.time(),
            'page_state': await self._get_page_state()
        }
        
    async def _should_skip_validation(self, text: str) -> bool:
        """Check if validation should be skipped based on context.
        
        Args:
            text: The text to validate
        
        Returns:
            bool: True if validation should be skipped
        """
        # Get last validation context
        context = self._validation_context.get(text)
        if not context:
            return False
            
        # Check if page state has changed
        if await self._is_page_state_changed():
            return False
            
        # Check if validation is still valid (within timeout)
        if time.time() - context['timestamp'] > self._extraction_timeout:
            return False
            
        # Skip if last validation was successful
        return context['success']
        
    async def verify_text(self, text: str, requirement: str) -> Tuple[bool, Optional[str]]:
        """Verify text presence on the page.
        
        Args:
            text: The text to verify
            requirement: The original requirement string
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        # Log step type if not already logged
        try:
            # Safely get current step number
            current_step = 0
            if hasattr(self.result, 'steps'):
                current_step = len(self.result.steps)
            elif isinstance(self.result, list):
                current_step = len(self.result)
                
            if current_step not in self._step_types:
                self._log_step_type(requirement, current_step)
        except Exception as e:
            logger.debug(f"Could not log step type: {e}")
            
        # Check if we should skip validation
        if await self._should_skip_validation(text):
            logger.debug(f"Skipping validation for '{text}' - already verified in current page state")
            return True, None
            
        # Check if this assertion was already successful
        if self._was_assertion_successful(text):
            logger.debug(f"Assertion for text '{text}' was previously successful")
            return True, None
            
        # Clear cache if it's too old
        if time.time() - self._last_extraction_time > self._extraction_timeout:
            self.clear_cache()
            
        # Try to extract the text
        result = await self.extract_text(text)
        if not result:
            await self._update_validation_context(text, False, "")
            return False, f"Failed to extract text: {text}"
            
        # Check if result indicates presence
        if isinstance(result, dict) and result.get('present', False):
            logger.debug(f"Text '{text}' found via presence flag")
            self._mark_assertion_successful(text)
            await self._update_validation_context(text, True, text)
            return True, None
            
        # Check if the text was found
        if isinstance(result, dict):
            content = result.get('extracted_content', '')
        else:
            content = str(result)
            
        # Try exact match first
        if text in content:
            logger.debug(f"Text '{text}' found via exact match")
            self._mark_assertion_successful(text)
            await self._update_validation_context(text, True, content)
            return True, None
            
        # Try fuzzy matching
        similarity = self._calculate_text_similarity(text, content)
        if similarity > self._fuzzy_match_threshold:
            logger.debug(f"Text '{text}' found via fuzzy match (similarity: {similarity})")
            self._mark_assertion_successful(text)
            await self._update_validation_context(text, True, content)
            return True, None
            
        # Only scroll if both exact and fuzzy matching failed
        scroll_attempts = 0
        while scroll_attempts < self._max_scroll_attempts:
            try:
                # Get browser page
                page = await self._get_browser_page()
                if not page:
                    logger.warning("No browser page available for scrolling")
                    break
                    
                logger.debug(f"Text not found in visible area, attempting to scroll (attempt {scroll_attempts + 1})...")
                await page.evaluate("window.scrollBy(0, 500)")
                await asyncio.sleep(0.5)  # Wait for scroll to complete
                
                # Try extraction again after scroll
                result = await self.extract_text(text)
                if result:
                    # Check if result indicates presence
                    if isinstance(result, dict) and result.get('present', False):
                        logger.debug(f"Text '{text}' found via presence flag after scroll")
                        self._mark_assertion_successful(text)
                        await self._update_validation_context(text, True, text)
                        return True, None
                        
                    if isinstance(result, dict):
                        content = result.get('extracted_content', '')
                    else:
                        content = str(result)
                        
                    if text in content:
                        logger.debug(f"Text '{text}' found via exact match after scroll")
                        self._mark_assertion_successful(text)
                        await self._update_validation_context(text, True, content)
                        return True, None
                        
                    # Try fuzzy matching again
                    similarity = self._calculate_text_similarity(text, content)
                    if similarity > self._fuzzy_match_threshold:
                        logger.debug(f"Text '{text}' found via fuzzy match after scroll (similarity: {similarity})")
                        self._mark_assertion_successful(text)
                        await self._update_validation_context(text, True, content)
                        return True, None
                        
                scroll_attempts += 1
            except Exception as e:
                logger.warning(f"Error during scroll attempt {scroll_attempts + 1}: {e}")
                break
                
        # Update validation context with failure
        await self._update_validation_context(text, False, content)
        return False, f"Text not found: {text}"
        
    def _process_pending_extractions(self, current_step: int, lookahead_window: int = 2) -> None:
        """Process all extractions in current and future steps within lookahead window.
        
        Args:
            current_step: Current step number
            lookahead_window: Number of future steps to look ahead
        """
        logger.debug(f"Processing extractions for step {current_step} with lookahead {lookahead_window}")
        
        # Process current and future steps
        for i, action in enumerate(self.result):
            if i < current_step or i > current_step + lookahead_window:
                continue
                
            if i in self._processed_extractions:
                continue
                
            if not hasattr(action, 'extract_content') or not action.extract_content:
                continue
                
            # Process extraction
            extraction_id = f"{i}_{action.extract_content.get('goal', '')}"
            if extraction_id in self._extraction_cache:
                continue
                
            # Normalize extraction content
            if isinstance(action.extract_content, dict):
                content = action.extract_content
            elif isinstance(action.extract_content, str):
                try:
                    import json
                    content = json.loads(action.extract_content)
                except json.JSONDecodeError:
                    content = {"text": action.extract_content}
            else:
                content = {"text": str(action.extract_content)}
                
            # Cache normalized extraction
            self._extraction_cache[extraction_id] = content
            self._processed_extractions.add(i)
            
            logger.debug(f"Processed extraction for step {i}: {content}")
            
    async def _get_last_extraction_result(self, requirement: str, current_step: int, xpath: Optional[str] = None) -> Optional[ActionResult]:
        """Get the most relevant extraction result for a requirement.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            xpath: Optional XPath to target specific elements
            
        Returns:
            Optional[ActionResult]: Most relevant extraction result or None
        """
        logger.debug(f"Getting extraction result for requirement: {requirement}")
        if xpath:
            logger.debug(f"Using XPath target: {xpath}")
        
        # Extract expected text from requirement
        match = re.search(r'["\'](.*?)["\']', requirement)
        if not match:
            logger.error(f"Could not extract text from requirement: {requirement}")
            return None
        expected_text = match.group(1)
        
        # Process pending extractions first
        self._process_pending_extractions(current_step)
        
        # Try to find matching extraction with retries
        max_retries = 2
        retry_delay = 1  # seconds
        best_match = None
        best_score = 0.0
        
        for attempt in range(max_retries):
            if attempt > 0:
                logger.debug(f"Retry attempt {attempt + 1} for extraction")
                await asyncio.sleep(retry_delay)
                
            # First try exact match in extracted content
            for step, action in enumerate(self.result):
                if not hasattr(action, 'extract_content') or not action.extract_content:
                    continue
                    
                # Get the extracted text
                content = action.extract_content
                if isinstance(content, dict):
                    content = content.copy()
                    
                    # Try known text keys first
                    for key in ["text", "exact_text", "extracted_text", "quote", "content", "page_content"]:
                        if key in content:
                            text = content[key]
                            if isinstance(text, str) and expected_text in text:
                                logger.info(f"Found match in {key}: {text}")
                                return self._process_extraction_result(content)
                    
                    # If XPath is provided, try to find matching element
                    if xpath and "elements" in content:
                        elements = content.get("elements", [])
                        matching_element = None
                        
                        # Look for element with matching XPath
                        for element in elements:
                            if isinstance(element, dict):
                                # Check XPath first
                                if element.get("xpath") == xpath:
                                    matching_element = element
                                    break
                                    
                                # Then check other element attributes
                                for attr in ["id", "class", "name", "aria-label"]:
                                    if attr in element and expected_text in str(element[attr]):
                                        matching_element = element
                                        break
                                        
                                # Finally check element text
                                if "text" in element and expected_text in element["text"]:
                                    matching_element = element
                                    break
                                    
                        if matching_element:
                            # Update content with the matching element's text
                            if "text" in matching_element:
                                content["text"] = matching_element["text"]
                            elif "value" in matching_element:
                                content["text"] = matching_element["value"]
                            logger.debug(f"Found element matching criteria: {matching_element}")
                            return self._process_extraction_result(content)
                            
                extracted_text = self._extract_text_from_content(content)
                
                # Check for exact match
                if expected_text == extracted_text:
                    logger.info(f"Found exact match in extraction: {extracted_text}")
                    return self._process_extraction_result(content)
                
                # Calculate similarity between expected text and extracted text
                score = self._calculate_text_similarity(expected_text, extracted_text)
                if score > best_score:
                    best_score = score
                    best_match = action
                    
            if best_match and best_score > 0.8:  # Higher threshold for fuzzy matching
                logger.info(f"Found matching extraction with score {best_score}")
                content = best_match.extract_content
                if isinstance(content, dict) and "quote" in content:
                    content = content.copy()
                    content["text"] = content["quote"]
                return self._process_extraction_result(content)
                
            # If no good match found, try to extract directly
            logger.info("No matching extraction found, attempting direct extraction")
            try:
                # Use the agent's controller to extract content
                if not hasattr(self.agent, 'controller'):
                    logger.error("Agent does not have controller attribute")
                    continue
                    
                # Check if page_extraction_llm is configured
                if not hasattr(self.agent, 'settings') or not self.agent.settings.page_extraction_llm:
                    logger.warning("Direct extraction requires page_extraction_llm but none is configured")
                    continue
                    
                # Create extraction action with enhanced options
                action_data = {
                    "extract_content": {
                        "goal": f"Find the exact text: {expected_text}",
                        "should_strip_link_urls": True,
                        "include_element_attributes": True,  # Include element attributes
                        "include_xpath": True,  # Include XPath information
                        "include_aria_labels": True,  # Include ARIA labels
                        "include_visible_text": True  # Include visible text
                    }
                }
                
                # Add XPath if provided
                if xpath:
                    action_data["extract_content"]["xpath"] = xpath
                    logger.debug(f"Added XPath to extraction action: {xpath}")
                
                # Create ActionModel instance
                from browser_use.agent.views import ActionModel
                class ExtractContentActionModel(ActionModel):
                    extract_content: dict | None = None
                    
                action = ExtractContentActionModel(**action_data)
                
                # Get browser session from agent
                if not hasattr(self.agent, 'browser_session'):
                    logger.error("Agent does not have browser_session attribute")
                    continue
                    
                # Use controller to act on the action with browser session and page_extraction_llm
                result = await self.agent.controller.act(
                    action, 
                    self.agent.browser_session,
                    page_extraction_llm=self.agent.settings.page_extraction_llm
                )
                
                if result and hasattr(result, 'extracted_content'):
                    content = result.extracted_content
                    if isinstance(content, dict) and "quote" in content:
                        content = content.copy()
                        content["text"] = content["quote"]
                    return self._process_extraction_result(content)
                    
            except Exception as e:
                logger.warning(f"Direct extraction failed on attempt {attempt + 1}: {str(e)}", exc_info=True)
                if attempt < max_retries - 1:
                    continue
                    
        return None
        
    def _normalize_url(self, url: str, base_url: Optional[str] = None) -> str:
        """Normalize a URL, handling relative paths and fragments.
        
        Args:
            url: URL to normalize
            base_url: Optional base URL for relative paths
            
        Returns:
            str: Normalized URL
        """
        if not url:
            return ""
            
        # Remove fragments
        url = url.split('#')[0]
        
        # Handle relative URLs
        if base_url and not urlparse(url).netloc:
            url = urljoin(base_url, url)
            
        # Normalize path
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'
            
        # Reconstruct URL
        return f"{parsed.scheme}://{parsed.netloc}{path}"
        
    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text content from various formats, including deeply nested structures.
        
        Args:
            content: Content to extract text from (can be string, dict, list, or any nested combination)
            
        Returns:
            str: Extracted text, with all string values from nested structures combined
        """
        def extract_strings(value: Any) -> List[str]:
            """Recursively extract all string values from nested structures."""
            strings = []
            
            if isinstance(value, str):
                strings.append(value)
            elif isinstance(value, dict):
                # First check for known text keys
                for key in ["exact_text", "text", "extracted_text", "quote", "content", "page_content"]:
                    if key in value:
                        text = value[key]
                        if isinstance(text, str):
                            strings.append(text)
                        else:
                            strings.extend(extract_strings(text))
                
                # Then recursively process all values
                for val in value.values():
                    strings.extend(extract_strings(val))
            elif isinstance(value, list):
                # Process each item in the list
                for item in value:
                    strings.extend(extract_strings(item))
            elif value is not None:
                # Convert any other type to string
                strings.append(str(value))
                
            return strings
            
        # Extract all strings and join them with spaces
        strings = extract_strings(content)
        if not strings:
            return str(content)  # Fallback to string conversion if no strings found
            
        # Join all strings with spaces and clean up whitespace
        text = " ".join(strings)
        text = " ".join(text.split())  # Normalize whitespace
        return text

    def _calculate_text_similarity(self, text1: str, text2: str | dict) -> float:
        """Calculate similarity between two texts.
        
        Args:
            text1: First text to compare
            text2: Second text to compare (can be string or dict)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Normalize input texts
        if not self._case_sensitive:
            text1 = ' '.join(text1.lower().split())
        else:
            text1 = ' '.join(text1.split())
            
        def normalize_text(text: str) -> str:
            """Normalize text by removing extra whitespace and respecting case sensitivity."""
            if not self._case_sensitive:
                return ' '.join(text.lower().split())
            return ' '.join(text.split())
            
        def check_dict_match(d: dict, search_text: str) -> float:
            """Recursively check dictionary for matches."""
            # First check keys for exact match
            for key in d.keys():
                key_norm = normalize_text(key)
                if search_text == key_norm:
                    return 1.0
                    
            # Then check keys for contains
            for key in d.keys():
                key_norm = normalize_text(key)
                if search_text in key_norm or key_norm in search_text:
                    return 1.0
                    
            # Then check values for exact match
            for value in d.values():
                if isinstance(value, str):
                    value_norm = normalize_text(value)
                    if search_text == value_norm:
                        return 1.0
                elif isinstance(value, dict):
                    # Recursively check nested dictionaries
                    score = check_dict_match(value, search_text)
                    if score == 1.0:
                        return 1.0
                        
            # Then check values for contains
            for value in d.values():
                if isinstance(value, str):
                    value_norm = normalize_text(value)
                    if search_text in value_norm or value_norm in search_text:
                        return 1.0
                        
            # If no exact or contains match found, calculate similarity for keys
            from difflib import SequenceMatcher
            best_similarity = 0.0
            for key in d.keys():
                key_norm = normalize_text(key)
                # Only calculate similarity if:
                # 1. The key is similar in length (within 2 characters)
                # 2. The key shares at least one word with the search text
                if (abs(len(search_text) - len(key_norm)) <= 2 and
                    any(word in key_norm for word in search_text.split())):
                    similarity = SequenceMatcher(None, search_text, key_norm).ratio()
                    best_similarity = max(best_similarity, similarity)
            return best_similarity
            
        # Handle dictionary input
        if isinstance(text2, dict):
            return check_dict_match(text2, text1)
            
        # Handle string input
        text2 = normalize_text(text2)
        
        # Check for exact match
        if text1 == text2:
            return 1.0
            
        # Check if one contains the other
        if text1 in text2 or text2 in text1:
            return 1.0
            
        # If no match found, return 0
        return 0.0

    def set_case_sensitive(self, case_sensitive: bool = True):
        """Set whether text matching should be case-sensitive.
        
        Args:
            case_sensitive: Whether to use case-sensitive matching (default: True)
        """
        self._case_sensitive = case_sensitive 