"""
Verification module for test assertions.
Handles verification of different types of assertions.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import re
from difflib import SequenceMatcher

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

from .base import BaseAssertions, AssertionResult, AssertionType
from .extraction import ExtractionAssertions
from .matching import MatchingAssertions

logger = logging.getLogger("test_framework.validation.assertions")

class VerificationAssertions(ExtractionAssertions, MatchingAssertions):
    """Handles verification of different types of assertions"""
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize verification assertions"""
        super().__init__(agent, result)
        self._matching = MatchingAssertions(agent, result)
        
    async def verify_requirement(self, requirement: str, step_number: int) -> AssertionResult:
        """Verify a test requirement.
        
        Args:
            requirement: The requirement string to verify
            step_number: The step number in the test
            
        Returns:
            AssertionResult: Result of the verification
        """
        return await self._verify_condition(requirement, step_number)
        
    async def _verify_condition(self, requirement: str, current_step: int) -> AssertionResult:
        """Verify a condition or requirement.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            
        Returns:
            AssertionResult: Result of the verification
        """
        logger.debug(f"Verifying condition: {requirement}")
            
        # Extract expected text from requirement
        match = re.search(r'["\'](.*?)["\']', requirement)
        if not match:
            return AssertionResult(
                success=False,
                message=f"Could not extract expected text from requirement: {requirement}",
                error_code="INVALID_REQUIREMENT",
                metadata={
                    "requirement": requirement,
                    "step": current_step
                }
            )
        expected_text = match.group(1)
        
        # Get extraction result from controller
        extraction_result = await self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            return AssertionResult(
                success=False,
                message=f"Failed to extract content for requirement: {requirement}",
                error_code="EXTRACTION_FAILED",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_text": expected_text
                }
            )
            
        # Extract text from controller's result
        extracted_text = self._extract_text_from_content(extraction_result.extracted_content)
        if not extracted_text:
            return AssertionResult(
                success=False,
                message=f"No text content found for requirement: {requirement}",
                error_code="NO_CONTENT",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_text": expected_text,
                    "extraction_result": extraction_result
                }
            )
            
        # Try exact match first
        logger.debug(f"Attempting exact match - Expected: '{expected_text}', Actual: '{extracted_text}'")
        if expected_text == extracted_text:
            logger.info("Exact match successful")
            return AssertionResult(
                success=True,
                message=f"Requirement verified successfully with exact match: {requirement}",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_text": expected_text,
                    "extracted_text": extracted_text,
                    "mode": "exact"
                }
            )
            
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, expected_text, extracted_text).ratio()
        logger.debug(f"Text similarity ratio: {similarity}")
        
        # If similarity is too low, fail the verification
        if similarity < 0.95:  # 95% similarity threshold
            logger.warning(f"Text similarity too low ({similarity:.2f}) for requirement: {requirement}")
            return AssertionResult(
                success=False,
                message=f"Text similarity too low ({similarity:.2f}) for requirement: {requirement}",
                error_code="SIMILARITY_TOO_LOW",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_text": expected_text,
                    "extracted_text": extracted_text,
                    "similarity": similarity,
                    "mode": "similarity_check"
                }
            )
            
        # If similarity is high enough, try contains match
        logger.debug("Exact match failed but similarity is high, attempting contains match")
        if expected_text in extracted_text:
            logger.info("Contains match successful")
            return AssertionResult(
                success=True,
                message=f"Requirement verified successfully with contains match: {requirement}",
                metadata={
                    "requirement": requirement,
                    "step": current_step,
                    "expected_text": expected_text,
                    "extracted_text": extracted_text,
                    "similarity": similarity,
                    "mode": "contains"
                }
            )
            
        # If both exact and contains matching fail
        logger.warning(f"All verification attempts failed for requirement: {requirement}")
        return AssertionResult(
            success=False,
            message=f"Failed to verify requirement: {requirement}",
            error_code="VERIFICATION_FAILED",
            metadata={
                "requirement": requirement,
                "step": current_step,
                "expected_text": expected_text,
                "extracted_text": extracted_text,
                "similarity": similarity,
                "mode": "failed"
            }
        )
        
    def _process_list_requirement(self, requirement: str, current_step: int) -> AssertionResult:
        """Process a list-based requirement.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            
        Returns:
            AssertionResult: Result of the verification
        """
        logger.debug(f"Processing list requirement: {requirement}")
        
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
            return AssertionResult(
                success=False,
                error_code="INVALID_REQUIREMENT",
                message="No items found in list requirement",
                metadata={"requirement": requirement}
            )
            
        # Get extraction result
        extraction_result = self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            return AssertionResult(
                success=False,
                error_code="NO_EXTRACTION",
                message="No extraction found for list requirement",
                metadata={"requirement": requirement, "expected_items": expected_items}
            )
            
        # Parse content
        content_info = self._parse_extracted_content(extraction_result.extracted_content, "list")
        if not content_info["value"]:
            return AssertionResult(
                success=False,
                error_code="PARSE_ERROR",
                message=f"Failed to parse content: {content_info['parsing_error']}",
                metadata={"requirement": requirement, "content_info": content_info}
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
            
        # Create metadata
        metadata = self._create_metadata(requirement, expected_items, content_info)
        metadata.update({
            "match_all": match_all,
            "expected_items": expected_items,
            "actual_items": actual_items
        })
        
        # Verify list match
        return self._verify_list_match(expected_items, actual_items, match_all)
        
    def _verify_link(self, requirement: str, current_step: int) -> AssertionResult:
        """Verify a link requirement.
        
        Args:
            requirement: The requirement string
            current_step: Current step number
            
        Returns:
            AssertionResult: Result of the verification
        """
        logger.debug(f"Verifying link requirement: {requirement}")
        
        # Extract link text
        link_text = requirement.lower().replace("verify link", "").strip()
        
        # Get extraction result
        extraction_result = self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            return AssertionResult(
                success=False,
                error_code="NO_EXTRACTION",
                message="No extraction found for link requirement",
                metadata={"requirement": requirement, "link_text": link_text}
            )
            
        # Parse content
        content_info = self._parse_extracted_content(extraction_result.extracted_content, "link")
        if not content_info["value"]:
            return AssertionResult(
                success=False,
                error_code="PARSE_ERROR",
                message=f"Failed to parse content: {content_info['parsing_error']}",
                metadata={"requirement": requirement, "content_info": content_info}
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
            return AssertionResult(
                success=False,
                error_code="NO_LINK",
                message="No link found in extraction",
                metadata={"requirement": requirement, "content_info": content_info}
            )
            
        # Normalize link
        link = self._normalize_url(link)
        
        # Create metadata
        metadata = self._create_metadata(requirement, link_text, content_info)
        metadata.update({
            "link_text": link_text,
            "link": link
        })
        
        # Verify link
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
            AssertionResult: Result of the verification
        """
        logger.debug(f"Verifying attribute requirement: {requirement}")
        
        # Extract attribute and value
        attr_match = re.search(r"verify attribute (\w+)(?:\s*=\s*['\"]([^'\"]+)['\"])?", requirement)
        if not attr_match:
            return AssertionResult(
                success=False,
                error_code="INVALID_REQUIREMENT",
                message="Invalid attribute requirement format",
                metadata={"requirement": requirement}
            )
            
        attr_name = attr_match.group(1)
        expected_value = attr_match.group(2)
        
        # Get extraction result
        extraction_result = self._get_last_extraction_result(requirement, current_step)
        if not extraction_result:
            return AssertionResult(
                success=False,
                error_code="NO_EXTRACTION",
                message="No extraction found for attribute requirement",
                metadata={"requirement": requirement, "attribute": attr_name}
            )
            
        # Parse content
        content_info = self._parse_extracted_content(extraction_result.extracted_content, "attribute")
        if not content_info["value"]:
            return AssertionResult(
                success=False,
                error_code="PARSE_ERROR",
                message=f"Failed to parse content: {content_info['parsing_error']}",
                metadata={"requirement": requirement, "content_info": content_info}
            )
            
        # Extract attribute value
        actual_value = None
        if isinstance(content_info["value"], dict):
            if attr_name in content_info["value"]:
                actual_value = content_info["value"][attr_name]
            elif "attributes" in content_info["value"] and attr_name in content_info["value"]["attributes"]:
                actual_value = content_info["value"]["attributes"][attr_name]
                
        if actual_value is None:
            return AssertionResult(
                success=False,
                error_code="NO_ATTRIBUTE",
                message=f"Attribute {attr_name} not found",
                metadata={"requirement": requirement, "content_info": content_info}
            )
            
        # Create metadata
        metadata = self._create_metadata(requirement, expected_value, content_info)
        metadata.update({
            "attribute": attr_name,
            "expected_value": expected_value,
            "actual_value": actual_value
        })
        
        # If no expected value, just verify attribute exists
        if not expected_value:
            return AssertionResult(
                success=True,
                message=f"Found attribute {attr_name}",
                metadata=metadata
            )
            
        # Verify attribute value
        return self._verify_text_match(expected_value, str(actual_value)) 