"""
Verification module for test assertions.
Handles verification of different types of assertions.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import re
from difflib import SequenceMatcher

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

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

    async def verify_text(self, expected_text: str, requirement: str, case_sensitive: bool = True) -> Tuple[bool, Optional[str]]:
        """Verify text content with configurable case sensitivity.
        
        Args:
            expected_text: The expected text to verify
            requirement: The requirement string
            case_sensitive: Whether to perform case-sensitive matching (default: True)
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        logger.debug(f"Verifying text: expected='{expected_text}', case_sensitive={case_sensitive}")
        
        # Set case sensitivity for matching
        self._matching.set_case_sensitive(case_sensitive)
        
        # Create a mock action result with the requirement
        mock_result = ActionResult(
            success=True,
            extracted_content=requirement,
            error=None
        )
        
        # Verify the text
        result = await self._verify_condition(requirement, 0, use_fuzzy_matching=False)
        
        if result.success:
            return True, None
        else:
            return False, result.message 