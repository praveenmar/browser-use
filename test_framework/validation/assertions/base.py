"""
Base module for test assertions.
Contains core functionality and types used across assertion modules.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import logging

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

logger = logging.getLogger("test_framework.validation.assertions")

class AssertionType(Enum):
    """Types of assertions supported"""
    TEXT = "text"
    LINK = "link"
    ATTRIBUTE = "attribute"
    LIST = "list"
    MULTI_VALUE = "multi_value"

@dataclass
class AssertionResult:
    """Result of an assertion check"""
    success: bool
    error_code: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ListAssertionResult(AssertionResult):
    """Result of a list-based assertion check"""
    matched_items: List[Dict[str, Any]] = None
    unmatched_items: List[Dict[str, Any]] = None
    total_items: int = 0
    matched_count: int = 0

class BaseAssertions:
    """Base class for test assertions with common functionality"""
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize assertions with agent and result history"""
        logger.debug(f"Initializing BaseAssertions with agent and result history")
        self.agent = agent
        self.result = result
        self.page = None
        
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
        
    def _process_extraction_result(self, result: Any) -> ActionResult:
        """Process an extraction result into a standardized ActionResult format.
        
        Args:
            result: The raw extraction result
            
        Returns:
            ActionResult: Processed result in standard format
        """
        logger.debug(f"Processing extraction result: {result}")
        
        # If already an ActionResult with extracted_content, return as is
        if hasattr(result, 'extracted_content'):
            return result
            
        # Create a new ActionResult with the content
        content = result
        if isinstance(content, dict):
            logger.debug(f"Processing dictionary content: {content}")
            # Handle element-specific extraction
            if any(key.startswith('element_') for key in content.keys()):
                # Get the first element's text
                element_text = next(iter(content.values()))
                content = str(element_text)
                logger.debug(f"Converted element text: {content}")
            # Handle page content extraction
            elif "page_content" in content:
                content = str(content["page_content"])
                logger.debug(f"Converted page content: {content}")
            # Handle extracted_text format
            elif "extracted_text" in content:
                content = str(content["extracted_text"])
                logger.debug(f"Converted extracted text: {content}")
            # Handle text format
            elif "text" in content:
                content = str(content["text"])
                logger.debug(f"Converted text content: {content}")
            else:
                # Handle any other dictionary format
                content = str(content)
                logger.debug(f"Converted dictionary content: {content}")
        elif isinstance(content, str):
            # Try to parse as JSON if it's a string
            try:
                import json
                data = json.loads(content)
                if isinstance(data, dict):
                    if "text" in data:
                        content = str(data["text"])
                    elif "extracted_text" in data:
                        content = str(data["extracted_text"])
                    elif "page_content" in data:
                        content = str(data["page_content"])
                    else:
                        content = str(data)
                else:
                    content = str(data)
                logger.debug(f"Parsed JSON content: {content}")
            except json.JSONDecodeError:
                # If not valid JSON, use the string as is
                content = str(content)
                logger.debug(f"Using raw string content: {content}")
        else:
            # Handle raw text content
            content = str(content)
            logger.debug(f"Converted raw content: {content}")
            
        return ActionResult(
            success=True,
            extracted_content=content
        )
        
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
            
        # Try to parse as JSON if it's a string
        if isinstance(content, str):
            try:
                import json
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
                result["value"] = {"text": content}
                return result

        # Handle any other type
        logger.info(f"Content is raw type {type(content)} for {expected_type}")
        result["parsing_status"] = "raw"
        result["value"] = {"text": str(content)}
        return result 