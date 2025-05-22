"""
Assertions module for validating test results.
Provides high-level interface for test assertions.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
from urllib.parse import urljoin
import json
import re
import logging

from playwright.async_api import Page
from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

from .models import LinkExtractionResult
from .validator import TestValidator, ValidationMode, ValidationResult

logger = logging.getLogger("test_framework.validation.assertions")

class AssertionType(Enum):
    """Types of assertions supported"""
    TEXT = "text"
    LINK = "link"
    ATTRIBUTE = "attribute"

@dataclass
class AssertionResult:
    """Result of an assertion check"""
    success: bool
    error_code: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TestAssertions:
    """High-level interface for test assertions"""
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize assertions with agent and result history"""
        logger.debug(f"Initializing TestAssertions with agent and result history")
        self.agent = agent
        self.result = result
        self.page = None
    
    @classmethod
    def register_actions(cls, controller):
        """Register assertion actions with the controller"""
        logger.debug("Registering assertion actions with controller")
        # Register text verification action
        controller.registry.action(
            description="Verify text content matches expected value",
            param_model=None
        )(cls.verify_text)
        
        # Register link verification action
        controller.registry.action(
            description="Verify link text and href match expected values",
            param_model=None
        )(cls.verify_link)
        
        # Register attribute verification action
        controller.registry.action(
            description="Verify attribute value matches expected value",
            param_model=None
        )(cls.verify_attribute)
    
    def _get_last_extraction_result(self, action_type: str) -> Optional[ActionResult]:
        """Get the last successful extraction result of the specified type from history.
        
        Args:
            action_type: Type of extraction action to look for (e.g., "extract_text", "extract_link")
            
        Returns:
            Optional[ActionResult]: The last successful extraction result or None if not found
        """
        logger.debug(f"Looking for last {action_type} extraction in history")
        logger.debug(f"History type: {type(self.result)}")
        
        # Use a stack-like approach with a single pass
        last_extraction = None
        
        # Single pass through history - O(n) time complexity
        for history_item in self.result.history:
            logger.debug(f"Processing history item: {history_item}")
            
            # Check if this history item has a model output
            if not history_item.model_output:
                continue
                
            # Check if this history item has actions
            if not history_item.model_output.action:
                continue
                
            # Check if this history item has results
            if not history_item.result:
                continue
                
            # Get the last action and result
            action = history_item.model_output.action[-1]
            result = history_item.result[-1]
            
            logger.debug(f"Processing action: {action}, result: {result}")
            
            # Check if this is an extraction action
            if hasattr(action, 'extract_content'):
                logger.debug(f"Found extraction action: {action}")
                # Store the result and continue to get the last one
                last_extraction = result
                logger.debug(f"Stored extraction result: {result}")
                
        if last_extraction:
            logger.debug(f"Processing last extraction: {last_extraction}")
            # If we have a result, ensure it has the expected structure
            if not hasattr(last_extraction, 'extracted_content'):
                logger.debug(f"Converting result to ActionResult: {last_extraction}")
                # Create a new ActionResult with the content
                content = last_extraction
                if isinstance(content, dict):
                    logger.debug(f"Processing dictionary content: {content}")
                    # Handle element-specific extraction
                    if any(key.startswith('element_') for key in content.keys()):
                        # Get the first element's text
                        element_text = next(iter(content.values()))
                        content = {"text": element_text}
                        logger.debug(f"Converted element text: {content}")
                    # Handle page content extraction
                    elif "page_content" in content:
                        content = {"text": content["page_content"]}
                        logger.debug(f"Converted page content: {content}")
                    # Handle extracted_text format
                    elif "extracted_text" in content:
                        content = {"text": content["extracted_text"]}
                        logger.debug(f"Converted extracted text: {content}")
                else:
                    # Handle raw text content (like the entire page content)
                    content = {"text": str(content)}
                    logger.debug(f"Converted raw content: {content}")
                    
                last_extraction = ActionResult(
                    success=True,
                    extracted_content=content,
                    data=content
                )
                logger.debug(f"Created ActionResult: {last_extraction}")
                
            logger.debug(f"Returning extraction result: {last_extraction}")
            return last_extraction
            
        logger.warning(f"No successful {action_type} extraction found in history")
        return None
        
    def _parse_extracted_content(self, content: Any, expected_type: str) -> Dict[str, Any]:
        """Parse extracted content with proper error handling and logging.
        
        Args:
            content: The content to parse
            expected_type: Type of content being parsed (for logging)
            
        Returns:
            Dict[str, Any]: Parsed content info with status and value
        """
        logger.debug(f"Parsing extracted content for {expected_type}: {content}")
        result = {
            "parsing_status": "unknown",
            "parsing_error": None,
            "content_type": type(content).__name__,
            "value": None
        }
        
        if content is None:
            logger.warning(f"Extracted content is None for {expected_type}")
            return result
            
        if isinstance(content, dict):
            logger.info(f"Extracted content is a dictionary for {expected_type}")
            result["parsing_status"] = "dict"
            # Handle different dictionary formats
            if "extracted_text" in content:
                result["value"] = {"text": content["extracted_text"]}
            elif "text" in content:
                result["value"] = content
            else:
                result["value"] = content
            return result
            
        try:
            data = json.loads(content)
            logger.info(f"Successfully parsed JSON content for {expected_type}")
            result["parsing_status"] = "json"
            # Handle different JSON formats
            if isinstance(data, dict):
                if "extracted_text" in data:
                    result["value"] = {"text": data["extracted_text"]}
                elif "text" in data:
                    result["value"] = data
                else:
                    result["value"] = data
            else:
                result["value"] = {"text": str(data)}
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON content for {expected_type}: {str(e)}")
            result["parsing_status"] = "raw"
            result["parsing_error"] = str(e)
            result["value"] = {"text": str(content)}
            return result
            
    def _create_metadata(self, requirement: str, expected_value: Any, content_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized metadata for verification results.
        
        Args:
            requirement: The original requirement string
            expected_value: The expected value to verify
            content_info: Parsed content information
            
        Returns:
            Dict[str, Any]: Standardized metadata dictionary
        """
        return {
            "requirement": requirement,
            "expected_value": expected_value,
            "normalization": {
                "is_relative": False,
                "base_url": None,
                "normalized": False
            },
            "parsing": content_info
        }
        
    async def _verify_condition(self, requirement: str) -> AssertionResult:
        """Verify a single requirement condition using history.
        
        Args:
            requirement: The requirement string to verify
            
        Returns:
            AssertionResult: Result of the verification
        """
        logger.debug(f"Verifying condition: {requirement}")
        
        # Extract the condition type and parameters from the requirement
        if "assert" in requirement.lower() and "text" in requirement.lower():
            logger.debug("Processing text assertion")
            # Extract text to verify
            match = re.search(r'["\'](.*?)["\']', requirement)
            if not match:
                logger.error(f"Could not extract text from requirement: {requirement}")
                return AssertionResult(
                    success=False,
                    error_code="INVALID_REQUIREMENT",
                    message="Could not extract text to verify",
                    metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
                )
            expected_text = match.group(1)
            logger.debug(f"Extracted expected text: {expected_text}")
            
            # Get the last content extraction
            logger.debug("Getting last content extraction")
            last_extraction = self._get_last_extraction_result("extract_text")
            if not last_extraction:
                logger.error("No content extraction found in history")
                return AssertionResult(
                    success=False,
                    error_code="NO_EXTRACTION",
                    message="No content extraction found in history",
                    metadata=self._create_metadata(requirement, expected_text, {"parsing_status": "error"})
                )
            
            # Parse extracted content
            logger.debug(f"Parsing extracted content: {last_extraction}")
            content_info = self._parse_extracted_content(last_extraction.extracted_content, "text")
            logger.debug(f"Content info: {content_info}")
            
            # Create metadata
            metadata = self._create_metadata(requirement, expected_text, content_info)
            
            # Add actual text based on parsing result
            if content_info["parsing_status"] in ["dict", "json"]:
                logger.debug("Using dictionary/JSON content")
                metadata["actual_text"] = content_info["value"].get('text', str(content_info["value"]))
            else:
                logger.debug("Using raw content")
                metadata["actual_text"] = content_info["value"]
            
            # Verify the text
            logger.debug(f"Verifying text with metadata: {metadata}")
            return self.verify_text(last_extraction, expected_text, exact=False, metadata=metadata)
            
        elif "verify link" in requirement.lower():
            logger.debug("Processing link verification")
            # Extract link text and href
            text_match = re.search(r'verify link ["\'](.*?)["\']', requirement.lower())
            href_match = re.search(r'href=["\'](.*?)["\']', requirement.lower())
            
            if not text_match:
                logger.error(f"Could not extract link text from requirement: {requirement}")
                return AssertionResult(
                    success=False,
                    error_code="INVALID_REQUIREMENT",
                    message="Could not extract link text to verify",
                    metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
                )
                
            expected_text = text_match.group(1)
            expected_href = href_match.group(1) if href_match else None
            logger.debug(f"Extracted expected link text: {expected_text}, href: {expected_href}")
            
            # Get the last content extraction
            last_extraction = self._get_last_extraction_result("extract_link")
            if not last_extraction:
                logger.error("No content extraction found in history")
                return AssertionResult(
                    success=False,
                    error_code="NO_EXTRACTION",
                    message="No content extraction found in history",
                    metadata=self._create_metadata(requirement, expected_text, {"parsing_status": "error"})
                )
            
            # Parse extracted content
            content_info = self._parse_extracted_content(last_extraction.extracted_content, "link")
            logger.debug(f"Content info: {content_info}")
            
            # Create metadata
            metadata = self._create_metadata(requirement, expected_text, content_info)
            metadata["expected_href"] = expected_href
            
            # Add actual values based on parsing result
            if content_info["parsing_status"] in ["dict", "json"]:
                logger.debug("Using dictionary/JSON content")
                metadata["actual_text"] = content_info["value"].get('text')
                metadata["actual_href"] = content_info["value"].get('href')
            else:
                logger.debug("Using raw content")
                metadata["actual_text"] = content_info["value"]
                metadata["actual_href"] = None
            
            return self.verify_link(last_extraction, expected_text, expected_href, metadata=metadata)
            
        elif "verify attribute" in requirement.lower():
            logger.debug("Processing attribute verification")
            # Extract attribute value
            match = re.search(r'verify attribute ["\'](.*?)["\']', requirement.lower())
            if not match:
                logger.error(f"Could not extract attribute value from requirement: {requirement}")
                return AssertionResult(
                    success=False,
                    error_code="INVALID_REQUIREMENT",
                    message="Could not extract attribute value to verify",
                    metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
                )
            expected_value = match.group(1)
            logger.debug(f"Extracted expected attribute value: {expected_value}")
            
            # Get the last content extraction
            last_extraction = self._get_last_extraction_result("extract_attribute")
            if not last_extraction:
                logger.error("No content extraction found in history")
                return AssertionResult(
                    success=False,
                    error_code="NO_EXTRACTION",
                    message="No content extraction found in history",
                    metadata=self._create_metadata(requirement, expected_value, {"parsing_status": "error"})
                )
            
            # Parse extracted content
            content_info = self._parse_extracted_content(last_extraction.extracted_content, "attribute")
            logger.debug(f"Content info: {content_info}")
            
            # Create metadata
            metadata = self._create_metadata(requirement, expected_value, content_info)
            
            # Add actual value based on parsing result
            if content_info["parsing_status"] in ["dict", "json"]:
                logger.debug("Using dictionary/JSON content")
                metadata["actual_value"] = content_info["value"].get('value', str(content_info["value"]))
            else:
                logger.debug("Using raw content")
                metadata["actual_value"] = content_info["value"]
            
            return self.verify_attribute(last_extraction, expected_value, metadata=metadata)
            
        logger.error(f"Unknown requirement type: {requirement}")
        return AssertionResult(
            success=False,
            error_code="UNKNOWN_REQUIREMENT",
            message=f"Unknown requirement type: {requirement}",
            metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
        )
    
    @staticmethod
    def verify_text(
        action_result: ActionResult,
        expected_text: str,
        exact: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssertionResult:
        """Verify text content"""
        logger.debug(f"Verifying text: expected='{expected_text}', exact={exact}")
        
        if not action_result.success:
            logger.error(f"Text verification failed: {action_result.error}")
            return AssertionResult(
                success=False,
                error_code="ACTION_FAILED",
                message=f"Action failed: {action_result.error}",
                metadata=metadata
            )
            
        # Handle extracted content safely
        actual_text = None
        if action_result.extracted_content:
            logger.debug(f"Processing extracted content: {action_result.extracted_content}")
            if isinstance(action_result.extracted_content, dict):
                logger.debug("Content is a dictionary")
                actual_text = action_result.extracted_content.get('text', str(action_result.extracted_content))
            else:
                try:
                    data = json.loads(action_result.extracted_content)
                    logger.debug("Successfully parsed JSON content")
                    actual_text = data.get('text', action_result.extracted_content)
                except json.JSONDecodeError:
                    logger.debug("Using raw string content")
                    actual_text = action_result.extracted_content

        if not actual_text:
            logger.error("No text content found")
            return AssertionResult(
                success=False,
                error_code="TEXT_EMPTY",
                message="No text content found",
                metadata=metadata
            )
            
        # Use the validator to check the text
        mode = ValidationMode.EXACT if exact else ValidationMode.CONTAINS
        logger.debug(f"Validating text with mode: {mode}")
        result = TestValidator.validate_text(
            expected=expected_text,
            actual=actual_text,
            mode=mode,
            original_requirement=metadata.get("requirement", "") if metadata else ""
        )
        
        if not result.success:
            logger.error(f"Text validation failed: {result.reason}")
        else:
            logger.info("Text validation succeeded")
        
        # Update metadata with actual values
        if metadata is None:
            metadata = {}
        metadata.update({
            "expected_text": expected_text,
            "actual_text": actual_text,
            "validation_mode": mode.value,
            "validation_result": {
                "success": result.success,
                "error_code": result.error_code,
                "reason": result.reason
            }
        })
        
        return AssertionResult(
            success=result.success,
            error_code=result.error_code,
            message=result.reason,
            metadata=metadata
        )
    
    @staticmethod
    def verify_link(
        action_result: ActionResult,
        expected_text: str,
        expected_href: str,
        exact: bool = True,
        is_relative: bool = False,
        base_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssertionResult:
        """Verify link text and href"""
        logger.debug(f"Verifying link: text='{expected_text}', href='{expected_href}', exact={exact}")
        
        if not action_result.success:
            logger.error(f"Link verification failed: {action_result.error}")
            return AssertionResult(
                success=False,
                error_code="ACTION_FAILED",
                message=f"Action failed: {action_result.error}",
                metadata=metadata
            )
            
        # Handle extracted content safely
        actual_text = None
        actual_href = None
        
        if action_result.extracted_content:
            logger.debug(f"Processing extracted content: {action_result.extracted_content}")
            if isinstance(action_result.extracted_content, dict):
                logger.debug("Content is a dictionary")
                actual_text = action_result.extracted_content.get('text')
                actual_href = action_result.extracted_content.get('href')
            else:
                try:
                    data = json.loads(action_result.extracted_content)
                    logger.debug("Successfully parsed JSON content")
                    actual_text = data.get('text')
                    actual_href = data.get('href')
                except json.JSONDecodeError:
                    logger.debug("Using raw string content")
                    actual_text = action_result.extracted_content
                    actual_href = None
            
        if not actual_text:
            logger.error("Link element not found")
            return AssertionResult(
                success=False,
                error_code="LINK_NOT_FOUND",
                message="Link element not found",
                metadata=metadata
            )
            
        # Use the validator to check the link
        mode = ValidationMode.EXACT if exact else ValidationMode.CONTAINS
        logger.debug(f"Validating link with mode: {mode}")
        result = TestValidator.validate_link(
            expected_text=expected_text,
            expected_href=expected_href,
            actual_text=actual_text,
            actual_href=actual_href,
            base_url=base_url,
            is_relative=is_relative,
            mode=mode,
            original_requirement=metadata.get("requirement", "") if metadata else ""
        )
        
        if not result.success:
            logger.error(f"Link validation failed: {result.reason}")
        else:
            logger.info("Link validation succeeded")
        
        # Update metadata with actual values
        if metadata is None:
            metadata = {}
        metadata.update({
            "expected_text": expected_text,
            "expected_href": expected_href,
            "actual_text": actual_text,
            "actual_href": actual_href,
            "validation_mode": mode.value,
            "validation_result": {
                "success": result.success,
                "error_code": result.error_code,
                "reason": result.reason
            }
        })
        
        return AssertionResult(
            success=result.success,
            error_code=result.error_code,
            message=result.reason,
            metadata=metadata
        )
    
    @staticmethod
    def verify_attribute(
        action_result: ActionResult,
        expected_value: str,
        relaxed: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssertionResult:
        """Verify attribute value"""
        logger.debug(f"Verifying attribute: expected='{expected_value}', relaxed={relaxed}")
        
        if not action_result.success:
            logger.error(f"Attribute verification failed: {action_result.error}")
            return AssertionResult(
                success=False,
                error_code="ACTION_FAILED",
                message=f"Action failed: {action_result.error}",
                metadata=metadata
            )
            
        # Handle extracted content safely
        actual_value = None
        if action_result.extracted_content:
            logger.debug(f"Processing extracted content: {action_result.extracted_content}")
            if isinstance(action_result.extracted_content, dict):
                logger.debug("Content is a dictionary")
                actual_value = action_result.extracted_content.get('value', str(action_result.extracted_content))
            else:
                try:
                    data = json.loads(action_result.extracted_content)
                    logger.debug("Successfully parsed JSON content")
                    actual_value = data.get('value', action_result.extracted_content)
                except json.JSONDecodeError:
                    logger.debug("Using raw string content")
                    actual_value = action_result.extracted_content

        if not actual_value:
            logger.error("Attribute not found")
            return AssertionResult(
                success=False,
                error_code="ATTRIBUTE_NOT_FOUND",
                message="Attribute not found",
                metadata=metadata
            )
            
        # Use the validator to check the attribute
        mode = ValidationMode.RELAXED if relaxed else ValidationMode.EXACT
        logger.debug(f"Validating attribute with mode: {mode}")
        result = TestValidator.validate_text(
            expected=expected_value,
            actual=actual_value,
            mode=mode,
            original_requirement=metadata.get("requirement", "") if metadata else ""
        )
        
        if not result.success:
            logger.error(f"Attribute validation failed: {result.reason}")
        else:
            logger.info("Attribute validation succeeded")
        
        # Update metadata with actual values
        if metadata is None:
            metadata = {}
        metadata.update({
            "expected_value": expected_value,
            "actual_value": actual_value,
            "validation_mode": mode.value,
            "validation_result": {
                "success": result.success,
                "error_code": result.error_code,
                "reason": result.reason
            }
        })
        
        return AssertionResult(
            success=result.success,
            error_code=result.error_code,
            message=result.reason,
            metadata=metadata
        )