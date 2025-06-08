"""
Verification module for test assertions.
Handles verification of different types of assertions.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import re
from difflib import SequenceMatcher
import asyncio
import time

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent
from browser_use.browser.session import BrowserSession
from playwright.async_api import Page
from ..utils.visibility_utils import is_element_visible_by_handle, is_element_in_viewport

from .base import BaseAssertions, AssertionResult, AssertionType
from .extraction import ExtractionAssertions
from .matching import MatchingAssertions

logger = logging.getLogger("test_framework.validation.assertions")

# Constants for similarity thresholds
FUZZY_MATCH_THRESHOLD = 0.85
EXACT_MATCH_THRESHOLD = 1.0

class VerificationAssertions(ExtractionAssertions, MatchingAssertions):
    """Handles verification of different types of assertions.
    
    This class provides methods for verifying various types of assertions including:
    - Text content verification
    - List verification
    - Link verification
    - Attribute verification
    
    Each verification method follows a multi-tiered matching strategy:
    1. Exact match (case-sensitive)
    2. Contains match (case-sensitive)
    3. Fuzzy match (if enabled)
    """
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize verification assertions.
        
        Args:
            agent: The agent instance for browser interactions
            result: The history of agent actions
        """
        super().__init__(agent, result)
        self._matching = MatchingAssertions(agent, result)
        self.browser_session = None
        
        # Initialize browser session
        if hasattr(agent, 'browser_session'):
            self.browser_session = agent.browser_session
            logger.debug(f"Initialized browser session: {type(self.browser_session)}")
        
    async def verify_requirement(self, requirement: str, step_number: int) -> AssertionResult:
        """Verify a test requirement.
        
        Args:
            requirement: The requirement string to verify
            step_number: The step number in the test
            
        Returns:
            AssertionResult: Result of the verification with detailed metadata
        """
        logger.info(f"[Step {step_number}] Verifying requirement: {requirement}")
        return await self._verify_condition(requirement, step_number)
        
    async def _verify_condition(self, requirement: str, current_step: int, use_fuzzy_matching: bool = True) -> AssertionResult:
        """Verify a condition or requirement using a multi-tiered matching strategy.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            use_fuzzy_matching: Whether to use fuzzy matching (default: True)
            
        Returns:
            AssertionResult: Result of the verification with detailed metadata
        """
        logger.debug(f"[Step {current_step}] Starting verification for: {requirement}")
        
        # Extract all quoted texts from requirement
        quoted_texts = re.findall(r'["\'](.*?)["\']', requirement)
        if not quoted_texts:
            logger.error(f"[Step {current_step}] No quoted text found in requirement: {requirement}")
            return AssertionResult(
                success=False,
                message=f"Could not extract expected text from requirement: {requirement}",
                error_code="INVALID_REQUIREMENT",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "error_type": "missing_quoted_text"
                }
            )
            
        # Get extraction result from controller
        extraction_result = await self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            logger.error(f"[Step {current_step}] No extraction result found for: {requirement}")
            return AssertionResult(
                success=False,
                message=f"Failed to extract content for requirement: {requirement}",
                error_code="EXTRACTION_FAILED",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_texts": quoted_texts,
                    "error_type": "extraction_failed"
                }
            )
            
        # Extract text from controller's result
        extracted_text = self._extract_text_from_content(extraction_result.extracted_content)
        if not extracted_text:
            logger.error(f"[Step {current_step}] No text content found in extraction for: {requirement}")
            return AssertionResult(
                success=False,
                message=f"No text content found for requirement: {requirement}",
                error_code="NO_CONTENT",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_texts": quoted_texts,
                    "extraction_result": extraction_result,
                    "error_type": "no_content"
                }
            )
            
        # Track results for each quoted text
        results = []
        all_success = True
        
        for expected_text in quoted_texts:
            logger.debug(f"[Step {current_step}] Processing expected text: '{expected_text}'")
            
            # Use the new match_text method with case-sensitive matching
            match_result = self._matching.match_text(expected_text, extracted_text, use_fuzzy_matching)
            match_result["expected_text"] = expected_text
            results.append(match_result)
            
            if not match_result["success"]:
                all_success = False
            
        # Create final assertion result
        metadata = {
            "requirement": requirement,
            "step": current_step,
            "extracted_text": extracted_text,
            "match_results": results,
            "use_fuzzy_matching": use_fuzzy_matching,
            "verification_type": "text"
        }
        
        if all_success:
            logger.info(f"[Step {current_step}] Requirement verified successfully: {requirement}")
            return AssertionResult(
                success=True,
                message=f"Requirement verified successfully: {requirement}",
                metadata=metadata
            )
        else:
            logger.error(f"[Step {current_step}] Failed to verify requirement: {requirement}")
            return AssertionResult(
                success=False,
                message=f"Failed to verify requirement: {requirement}",
                error_code="VERIFICATION_FAILED",
                metadata=metadata
            )
            
    def _process_list_requirement(self, requirement: str, current_step: int) -> AssertionResult:
        """Process a list-based requirement using a multi-tiered matching strategy.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            
        Returns:
            AssertionResult: Result of the verification with detailed metadata
        """
        logger.info(f"[Step {current_step}] Processing list requirement: {requirement}")
        
        # Extract expected items
        expected_items = []
        match_all = True
        
        # Parse requirement
        if "verify all" in requirement.lower():
            match_all = True
            # Extract items after "verify all"
            items_text = requirement.lower().split("verify all", 1)[1].strip()
        elif "verify any" in requirement.lower():
            match_all = False
            # Extract items after "verify any"
            items_text = requirement.lower().split("verify any", 1)[1].strip()
        else:
            # Extract items after "verify list"
            items_text = requirement.lower().split("verify list", 1)[1].strip()
            
        # Split items
        if "and" in items_text:
            expected_items = [item.strip() for item in items_text.split("and")]
        elif "or" in items_text:
            expected_items = [item.strip() for item in items_text.split("or")]
        else:
            expected_items = [items_text]
            
        if not expected_items:
            logger.error(f"[Step {current_step}] No items found in list requirement: {requirement}")
            return AssertionResult(
                success=False,
                error_code="INVALID_REQUIREMENT",
                message="No items found in list requirement",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "error_type": "no_items"
                }
            )
            
        # Get extraction result
        extraction_result = self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            logger.error(f"[Step {current_step}] No extraction found for list requirement: {requirement}")
            return AssertionResult(
                success=False,
                error_code="NO_EXTRACTION",
                message="No extraction found for list requirement",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_items": expected_items,
                    "error_type": "no_extraction"
                }
            )
            
        # Parse content
        content_info = self._parse_extracted_content(extraction_result.extracted_content, "list")
        if not content_info["value"]:
            logger.error(f"[Step {current_step}] Failed to parse content for list requirement: {requirement}")
            return AssertionResult(
                success=False,
                error_code="PARSE_ERROR",
                message=f"Failed to parse content: {content_info['parsing_error']}",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "content_info": content_info,
                    "error_type": "parse_error"
                }
            )
            
        # Extract items
        actual_items = []
        if isinstance(content_info["value"], list):
            actual_items = [self._extract_text_from_content(item) for item in content_info["value"]]
        elif isinstance(content_info["value"], dict):
            if "items" in content_info["value"]:
                actual_items = [self._extract_text_from_content(item) for item in content_info["value"]["items"]]
            else:
                actual_items = [self._extract_text_from_content(content_info["value"])]
        else:
            actual_items = [self._extract_text_from_content(content_info["value"])]
            
        # Track results for each expected item
        results = []
        all_success = True
        
        for expected_item in expected_items:
            logger.debug(f"[Step {current_step}] Processing expected item: '{expected_item}'")
            result = {
                "expected_text": expected_item,
                "match_type": None,
                "similarity_score": None,
                "matched_snippet": None,
                "success": False
            }
            
            best_match = None
            best_score = 0.0
            best_match_type = None
            
            # Try to find best match for this item
            for actual_item in actual_items:
                # Use the new match_text method
                match_result = self._matching.match_text(expected_item, actual_item)
                if match_result["success"] and match_result["similarity_score"] > best_score:
                    best_match = actual_item
                    best_score = match_result["similarity_score"]
                    best_match_type = match_result["match_type"]
                    
            # Update result with best match found
            if best_match:
                result.update({
                    "match_type": best_match_type,
                    "similarity_score": best_score,
                    "matched_snippet": best_match,
                    "success": True
                })
                logger.info(f"[Step {current_step}] Found {best_match_type} match for '{expected_item}' with score {best_score:.2f}")
            else:
                logger.warning(f"[Step {current_step}] No match found for item: '{expected_item}'")
                result.update({
                    "match_type": "no_match",
                    "success": False
                })
                all_success = False
                
            results.append(result)
            
        # Create final assertion result
        metadata = {
            "requirement": requirement,
            "step": current_step,
            "match_all": match_all,
            "expected_items": expected_items,
            "actual_items": actual_items,
            "match_results": results,
            "verification_type": "list"
        }
        
        # Determine overall success based on match_all flag
        overall_success = all_success if match_all else any(r["success"] for r in results)
        
        if overall_success:
            logger.info(f"[Step {current_step}] List requirement verified successfully: {requirement}")
            return AssertionResult(
                success=True,
                message=f"List requirement verified successfully: {requirement}",
                metadata=metadata
            )
        else:
            logger.error(f"[Step {current_step}] Failed to verify list requirement: {requirement}")
            return AssertionResult(
                success=False,
                message=f"Failed to verify list requirement: {requirement}",
                error_code="VERIFICATION_FAILED",
                metadata=metadata
            )
        
    def _verify_link(self, requirement: str, current_step: int) -> AssertionResult:
        """Verify a link requirement.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            
        Returns:
            AssertionResult: Result of the verification with detailed metadata
        """
        logger.info(f"[Step {current_step}] Processing link requirement: {requirement}")
        
        # Extract link text
        link_text = requirement.lower().replace("verify link", "").strip()
        
        # Get extraction result
        extraction_result = self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            logger.error(f"[Step {current_step}] No extraction found for link requirement: {requirement}")
            return AssertionResult(
                success=False,
                error_code="NO_EXTRACTION",
                message="No extraction found for link requirement",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "link_text": link_text,
                    "error_type": "no_extraction"
                }
            )
            
        # Parse content
        content_info = self._parse_extracted_content(extraction_result.extracted_content, "link")
        if not content_info["value"]:
            logger.error(f"[Step {current_step}] Failed to parse content for link requirement: {requirement}")
            return AssertionResult(
                success=False,
                error_code="PARSE_ERROR",
                message=f"Failed to parse content: {content_info['parsing_error']}",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "content_info": content_info,
                    "error_type": "parse_error"
                }
            )
            
        # Extract link
        link = None
        if isinstance(content_info["value"], dict):
            if "href" in content_info["value"]:
                link = content_info["value"]["href"]
            elif "link" in content_info["value"]:
                link = content_info["value"]["link"]
            elif "url" in content_info["value"]:
                link = content_info["value"]["url"]
                
        if not link:
            logger.error(f"[Step {current_step}] No link found in extraction for: {requirement}")
            return AssertionResult(
                success=False,
                error_code="NO_LINK",
                message="No link found in extraction",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "content_info": content_info,
                    "error_type": "no_link"
                }
            )
            
        # Normalize link
        link = self._normalize_url(link)
        
        # Create metadata
        metadata = {
            "requirement": requirement,
            "step": current_step,
            "link_text": link_text,
            "link": link,
            "content_info": content_info,
            "verification_type": "link"
        }
        
        # Verify link
        logger.info(f"[Step {current_step}] Link requirement verified successfully: {requirement}")
        return AssertionResult(
            success=True,
            message=f"Found link: {link}",
            metadata=metadata
        )
        
    def _verify_attribute(self, requirement: str, current_step: int) -> AssertionResult:
        """Verify an attribute requirement.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            
        Returns:
            AssertionResult: Result of the verification with detailed metadata
        """
        logger.info(f"[Step {current_step}] Processing attribute requirement: {requirement}")
        
        # Extract attribute and value
        attr_match = re.search(r"verify attribute (\w+)(?:\s*=\s*['\"]([^'\"]+)['\"])?", requirement)
        if not attr_match:
            logger.error(f"[Step {current_step}] Invalid attribute requirement format: {requirement}")
            return AssertionResult(
                success=False,
                error_code="INVALID_REQUIREMENT",
                message="Invalid attribute requirement format",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "error_type": "invalid_format"
                }
            )
            
        attr_name = attr_match.group(1)
        expected_value = attr_match.group(2)
        
        # Get extraction result
        extraction_result = self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            logger.error(f"[Step {current_step}] No extraction found for attribute requirement: {requirement}")
            return AssertionResult(
                success=False,
                error_code="NO_EXTRACTION",
                message="No extraction found for attribute requirement",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "attribute": attr_name,
                    "error_type": "no_extraction"
                }
            )
            
        # Parse content
        content_info = self._parse_extracted_content(extraction_result.extracted_content, "attribute")
        if not content_info["value"]:
            logger.error(f"[Step {current_step}] Failed to parse content for attribute requirement: {requirement}")
            return AssertionResult(
                success=False,
                error_code="PARSE_ERROR",
                message=f"Failed to parse content: {content_info['parsing_error']}",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "content_info": content_info,
                    "error_type": "parse_error"
                }
            )
            
        # Extract attribute value
        actual_value = None
        if isinstance(content_info["value"], dict):
            if attr_name in content_info["value"]:
                actual_value = content_info["value"][attr_name]
            elif "attributes" in content_info["value"] and attr_name in content_info["value"]["attributes"]:
                actual_value = content_info["value"]["attributes"][attr_name]
                
        if actual_value is None:
            logger.error(f"[Step {current_step}] Attribute {attr_name} not found in: {requirement}")
            return AssertionResult(
                success=False,
                error_code="NO_ATTRIBUTE",
                message=f"Attribute {attr_name} not found",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "content_info": content_info,
                    "error_type": "no_attribute"
                }
            )
            
        # Create metadata
        metadata = {
            "requirement": requirement,
            "step": current_step,
            "attribute": attr_name,
            "expected_value": expected_value,
            "actual_value": actual_value,
            "content_info": content_info,
            "verification_type": "attribute"
        }
        
        # If no expected value, just verify attribute exists
        if not expected_value:
            logger.info(f"[Step {current_step}] Attribute requirement verified successfully: {requirement}")
            return AssertionResult(
                success=True,
                message=f"Found attribute {attr_name}",
                metadata=metadata
            )
            
        # Verify attribute value using the new match_text method
        match_result = self._matching.match_text(expected_value, str(actual_value))
        metadata["match_result"] = match_result
        
        if match_result["success"]:
            logger.info(f"[Step {current_step}] Attribute requirement verified successfully: {requirement}")
            return AssertionResult(
                success=True,
                message=f"Attribute {attr_name} matches expected value",
                metadata=metadata
            )
        else:
            logger.error(f"[Step {current_step}] Failed to verify attribute requirement: {requirement}")
            return AssertionResult(
                success=False,
                message=f"Attribute {attr_name} does not match expected value",
                error_code="VERIFICATION_FAILED",
                metadata=metadata
            )

    async def _get_browser_page(self) -> Optional[Page]:
        """Get the browser page safely.
        
        Returns:
            Optional[Page]: The browser page if available, None otherwise
        """
        try:
            if self.browser_session:
                return await self.browser_session.get_current_page()
            return None
        except Exception as e:
            logger.warning(f"Error getting browser page: {e}")
            return None
            
    async def _wait_for_dom_stability(self, page: Page, timeout: float = 30.0) -> bool:
        """Wait for the DOM to be stable before running assertions.
        
        Args:
            page: The browser page
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if DOM is stable, False if timed out
        """
        try:
            logger.debug("Waiting for DOM stability...")
            start_time = time.time()
            
            # Wait for network idle state
            try:
                await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
                logger.debug("Network is idle")
            except Exception as e:
                logger.warning(f"Network idle timeout: {e}")
                
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                logger.warning(f"DOM stability timeout after {timeout}s")
                return False
                
            logger.debug("DOM is stable")
            return True
            
        except Exception as e:
            logger.warning(f"Error waiting for DOM stability: {e}")
            return False
            
    async def verify_text(self, action_result: ActionResult, expected_text: str) -> Tuple[bool, Optional[str]]:
        """Verify text content matches expected value.
        
        Args:
            action_result: The action result to verify
            expected_text: The expected text to find
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            logger.debug(f"🔍 Starting verification for text: '{expected_text}'")
            
            page = await self._get_browser_page()
            if not page:
                logger.error("❌ No browser page available")
                return False, "No browser page available"
                
            # Wait for initial DOM stability
            logger.debug("⏳ Waiting for initial DOM stability...")
            try:
                await page.wait_for_load_state('networkidle', timeout=5000)
                logger.debug("✅ Initial DOM stability achieved")
            except Exception as e:
                logger.warning(f"⚠️ Initial DOM stability timeout: {e}")
                
            # First check if text exists in DOM
            logger.debug("📄 Checking DOM content...")
            content = await page.content()
            if expected_text not in content:
                # Check if this is a continuation of a previous search
                if hasattr(self, '_previous_search_text') and self._previous_search_text == expected_text:
                    logger.debug("🔄 Continuing previous text search...")
                    # Don't fail immediately, let the agent continue searching
                    return False, f"Text '{expected_text}' not found in DOM, continuing search"
                else:
                    logger.error(f"❌ Text '{expected_text}' not found in DOM")
                    return False, f"Text '{expected_text}' not found in DOM"
            logger.debug(f"✅ Text '{expected_text}' found in DOM")
            
            # Store the current search text for continuation checks
            self._previous_search_text = expected_text
            
            # Wait for any ongoing scrolling to complete
            logger.debug("⏳ Waiting for any ongoing scrolling to complete...")
            try:
                await page.wait_for_load_state('networkidle', timeout=2000)
                await asyncio.sleep(0.5)  # Additional small wait for any animations
            except Exception:
                pass  # Ignore timeout, continue with verification
            
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
                            # Clear the previous search text since we found it
                            self._previous_search_text = None
                            return True, None
                        else:
                            logger.warning(f"⚠️ Text '{expected_text}' found but not in viewport")
                            # If element is not in viewport, scroll to it
                            try:
                                await element.scroll_into_view_if_needed()
                                await asyncio.sleep(0.5)  # Wait for scroll to complete
                                # Check visibility again after scrolling
                                is_visible = await self.browser_session.is_visible_by_handle(element)
                                if is_visible:
                                    logger.debug("✅ Text is visible after scrolling to it")
                                    # Clear the previous search text since we found it
                                    self._previous_search_text = None
                                    return True, None
                            except Exception as e:
                                logger.warning(f"⚠️ Error scrolling to element: {e}")
                            # Don't fail immediately, let the agent continue searching
                            return False, f"Text '{expected_text}' found but not in viewport, continuing search"
                    else:
                        logger.warning(f"⚠️ Text '{expected_text}' found but not visible")
                        # Don't fail immediately, let the agent continue searching
                        return False, f"Text '{expected_text}' found but not visible, continuing search"
                else:
                    logger.warning(f"⚠️ Could not find element with text '{expected_text}'")
                    # Don't fail immediately, let the agent continue searching
                    return False, f"Could not find element with text '{expected_text}', continuing search"
            except Exception as e:
                logger.warning(f"⚠️ Error checking text visibility: {e}")
                return False, f"Error checking text visibility: {e}, continuing search"
            
            # If text is not visible, try scrolling through the page
            logger.debug("🔄 Starting scroll search for text...")
            viewport_height = await page.evaluate("window.innerHeight")
            current_scroll = await page.evaluate("window.scrollY")
            max_scroll = await page.evaluate("document.body.scrollHeight")
            
            # First scroll to top
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)
            
            # Scroll in smaller increments for more thorough search
            scroll_increment = viewport_height // 8  # Eighth of viewport height for more precise scrolling
            while current_scroll < max_scroll:
                try:
                    # Scroll down by increment
                    await page.evaluate(f"window.scrollBy(0, {scroll_increment})")
                    await asyncio.sleep(0.3)  # Shorter wait time between smaller scrolls
                    
                    # Wait for any dynamic content to load
                    try:
                        await page.wait_for_load_state('networkidle', timeout=1000)
                    except Exception:
                        pass  # Ignore timeout, continue with verification
                        
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
                                # Clear the previous search text since we found it
                                self._previous_search_text = None
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

    async def verify_link(self, action_result: ActionResult, expected_text: str, expected_href: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Verify link content and href match expected values.
        
        Args:
            action_result: The action result to verify
            expected_text: The expected link text
            expected_href: Optional expected href value
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            page = await self._get_browser_page()
            if not page:
                return False, "No browser page available"
                
            # Wait for DOM stability with a shorter timeout
            await self._wait_for_dom_stability(page, timeout=10.0)
                
            # Try to find the link
            element = await page.get_by_role("link", name=expected_text).first
            if not element:
                return False, f"Link with text '{expected_text}' not found"
                
            # Scroll element into view first
            await element.scroll_into_view_if_needed()
            await asyncio.sleep(0.5)  # Wait for scroll to complete
            
            element_handle = await element.element_handle()
            if element_handle:
                # Check visibility using browser-use's utility
                is_visible = await is_element_visible_by_handle(element_handle, self.browser_session)
                if is_visible:
                    # Check if element is in viewport
                    is_in_viewport = await is_element_in_viewport(element_handle, page)
                    if is_in_viewport:
                        # Check href if provided
                        if expected_href:
                            href = await element.get_attribute("href")
                            if href != expected_href:
                                return False, f"Link href '{href}' does not match expected '{expected_href}'"
                        logger.info(f"✅ Link '{expected_text}' is visible in viewport")
                        return True, None
                    else:
                        logger.warning(f"Link '{expected_text}' found but not in viewport")
                        return False, f"Link '{expected_text}' found but not visible in viewport"
                else:
                    logger.warning(f"Link '{expected_text}' found but not visible")
                    return False, f"Link '{expected_text}' found but not visible"
                    
            return False, f"Link with text '{expected_text}' not found"
            
        except Exception as e:
            logger.error(f"Error verifying link: {e}")
            return False, str(e) 