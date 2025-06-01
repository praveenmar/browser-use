"""
Extraction module for test assertions.
Handles text extraction and processing functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urljoin, urlparse
import re

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
        logger.debug(f"Initialized ExtractionAssertions with agent type: {type(agent)}")
        
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
        
        # Try to find matching extraction
        best_match = None
        best_score = 0.0
        
        # First try exact match in extracted content
        for step, action in enumerate(self.result):
            if not hasattr(action, 'extract_content') or not action.extract_content:
                continue
                
            # Get the extracted text
            content = action.extract_content
            if isinstance(content, dict):
                content = content.copy()
                if "quote" in content:
                    content["text"] = content["quote"]
                    
                # If XPath is provided, try to find matching element
                if xpath and "elements" in content:
                    elements = content.get("elements", [])
                    matching_element = None
                    
                    # Look for element with matching XPath
                    for element in elements:
                        if isinstance(element, dict) and element.get("xpath") == xpath:
                            matching_element = element
                            break
                            
                    if matching_element:
                        # Update content with the matching element's text
                        if "text" in matching_element:
                            content["text"] = matching_element["text"]
                        elif "value" in matching_element:
                            content["text"] = matching_element["value"]
                        logger.debug(f"Found element matching XPath: {xpath}")
                    else:
                        logger.debug(f"No element found matching XPath: {xpath}")
                        continue
                        
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
                return None
                
            # Check if page_extraction_llm is configured
            if not hasattr(self.agent, 'settings') or not self.agent.settings.page_extraction_llm:
                logger.warning("Direct extraction requires page_extraction_llm but none is configured")
                return None
                
            # Create extraction action
            action_data = {
                "extract_content": {
                    "goal": f"Find the exact text: {expected_text}",
                    "should_strip_link_urls": True
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
                return None
                
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
            logger.warning(f"Direct extraction failed: {str(e)}", exc_info=True)
            
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
        text1 = ' '.join(text1.lower().split())
        
        def normalize_text(text: str) -> str:
            """Normalize text by removing extra whitespace and converting to lowercase."""
            return ' '.join(text.lower().split())
            
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