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
            
    async def _get_last_extraction_result(self, requirement: str, current_step: int) -> Optional[ActionResult]:
        """Get the most relevant extraction result for a requirement.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            
        Returns:
            Optional[ActionResult]: Most relevant extraction result or None
        """
        logger.debug(f"Getting extraction result for requirement: {requirement}")
        
        # Extract the expected text from the requirement
        match = re.search(r'["\'](.*?)["\']', requirement)
        if not match:
            logger.error(f"Could not extract text from requirement: {requirement}")
            return None
        expected_text = match.group(1)
        logger.debug(f"Looking for text: {expected_text}")
        
        # Process pending extractions first
        self._process_pending_extractions(current_step)
        
        # Try to find matching extraction
        best_match = None
        best_score = 0.0
        
        # First try exact match in extracted content
        for step, action in enumerate(self.result):
            if not hasattr(action, 'extract_content') or not action.extract_content:
                continue
                
            # Get the extracted text and normalize the content
            content = action.extract_content
            if isinstance(content, dict):
                # Normalize content to use consistent keys
                normalized_content = content.copy()
                if "quote" in normalized_content:
                    normalized_content["text"] = normalized_content["quote"]
                content = normalized_content
            
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
                
        if best_match and best_score > 0.6:  # Lower threshold for fuzzy matching
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
                    "goal": f"Find the text: {expected_text}",
                    "should_strip_link_urls": True
                }
            }
            
            # Create ActionModel instance
            from browser_use.agent.views import ActionModel
            class ExtractContentActionModel(ActionModel):
                extract_content: dict | None = None
                
            action = ExtractContentActionModel(**action_data)
            
            # Get browser context from agent
            if not hasattr(self.agent, 'browser_context'):
                logger.error("Agent does not have browser_context attribute")
                return None
                
            # Use controller to act on the action with browser context and page_extraction_llm
            result = await self.agent.controller.act(
                action, 
                self.agent.browser_context,
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
        """Extract text content from various formats.
        
        Args:
            content: Content to extract text from
            
        Returns:
            str: Extracted text
        """
        if isinstance(content, str):
            return content
            
        if isinstance(content, dict):
            # Try all possible text keys
            for key in ["text", "extracted_text", "quote", "content", "page_content"]:
                if key in content:
                    text = content[key]
                    if isinstance(text, str):
                        # Clean up the text
                        text = text.strip()
                        # Remove markdown blockquote if present
                        text = re.sub(r'^>\s*', '', text)
                        text = re.sub(r'\s*$', '', text)
                        return text
                    elif isinstance(text, dict) and "text" in text:
                        return text["text"]
                        
            # If no text found in specific keys, try to find any string value
            for value in content.values():
                if isinstance(value, str):
                    # Clean up the text
                    value = value.strip()
                    # Remove markdown blockquote if present
                    value = re.sub(r'^>\s*', '', value)
                    value = re.sub(r'\s*$', '', value)
                    return value
                    
            # If still no text found, convert the whole dict to string
            return str(content)
                
        return str(content)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Remove common variations
        text1 = text1.replace(" and ", " ").replace(",", "")
        text2 = text2.replace(" and ", " ").replace(",", "")
        
        # Exact match
        if text1 == text2:
            return 1.0
            
        # Check if one text contains the other
        if text1 in text2 or text2 in text1:
            return 0.9
            
        # Split into words and compare
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union 