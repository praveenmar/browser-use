"""
Helper functions for the validation framework.
"""

import logging
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from browser_use.agent.views import ActionResult

logger = logging.getLogger("test_framework.validation.helpers")

@dataclass
class ExtractionResult:
    """Base class for extraction results"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class TextExtractionResult(ExtractionResult):
    """Result of text extraction"""
    text: Optional[str] = None

@dataclass
class LinkExtractionResult(ExtractionResult):
    """Result of link extraction"""
    text: Optional[str] = None
    href: Optional[str] = None
    is_relative: Optional[bool] = None
    base_url: Optional[str] = None
    message: Optional[str] = None
    original_requirement: Optional[str] = None

@dataclass
class AttributeExtractionResult(ExtractionResult):
    """Result of attribute extraction"""
    value: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    element_info: Optional[Dict[str, Any]] = None

@dataclass
class TextBySelectorExtractionResult(ExtractionResult):
    """Result of text extraction by selector with metadata"""
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    element_info: Optional[Dict[str, Any]] = None

def _parse_content(content: Any) -> Dict[str, Any]:
    """Parse content that could be a string or dictionary"""
    if content is None:
        return {}
        
    if isinstance(content, dict):
        return content
        
    if not isinstance(content, str):
        return {"text": str(content)}
        
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"text": content}

async def extract_text(controller, text: str, exact: bool = False, timeout: int = 5000, original_requirement: str = "") -> TextExtractionResult:
    """
    Extract text from the page using controller-based extraction.
    This is a pure function that only extracts data, no validation.
    
    Args:
        controller: The controller instance
        text: Text to search for
        exact: Whether to match text exactly
        timeout: Timeout in milliseconds
        original_requirement: Original user requirement for traceability
        
    Returns:
        TextExtractionResult with extracted text or None if not found
    """
    try:
        action_result = await controller.act({
            'extract_text': {
                'text': text,
                'exact': exact,
                'timeout': timeout,
                'original_requirement': original_requirement
            }
        })
        
        if action_result.success:
            data = _parse_content(action_result.extracted_content)
            return TextExtractionResult(
                success=True,
                data=data,
                text=data.get('text', action_result.extracted_content)
            )
        else:
            return TextExtractionResult(
                success=False,
                error=action_result.error
            )
            
    except Exception as e:
        logger.error(f"Error extracting text '{text}': {str(e)}")
        return TextExtractionResult(
            success=False,
            error=str(e)
        )

async def extract_link(controller, text: str, exact: bool = False, timeout: int = 5000, original_requirement: str = "") -> LinkExtractionResult:
    """
    Extract link information from the page using controller-based extraction.
    This is a pure function that only extracts data, no validation.
    
    Args:
        controller: The controller instance
        text: Link text to search for
        exact: Whether to match text exactly
        timeout: Timeout in milliseconds
        original_requirement: Original user requirement for traceability
        
    Returns:
        LinkExtractionResult with extracted link info or None if not found
    """
    try:
        action_result = await controller.act({
            'extract_link': {
                'text': text,
                'exact': exact,
                'timeout': timeout,
                'original_requirement': original_requirement
            }
        })
        
        if action_result.success and action_result.extracted_content:
            data = _parse_content(action_result.extracted_content)
            return LinkExtractionResult(
                success=True,
                text=data.get('text'),
                href=data.get('href'),
                is_relative=data.get('is_relative'),
                base_url=data.get('base_url'),
                message=None,
                original_requirement=original_requirement
            )
        else:
            return LinkExtractionResult(
                success=False,
                text=None,
                href=None,
                is_relative=False,
                base_url=None,
                message=action_result.error,
                original_requirement=original_requirement
            )
            
    except Exception as e:
        logger.error(f"Error extracting link '{text}': {str(e)}")
        return LinkExtractionResult(
            success=False,
            text=None,
            href=None,
            is_relative=False,
            base_url=None,
            message=str(e),
            original_requirement=original_requirement
        )

async def extract_element_text(controller, selector: str, timeout: int = 5000, original_requirement: str = "") -> TextBySelectorExtractionResult:
    """
    Extract text from an element using its selector.
    This is a pure function that only extracts data, no validation.
    
    Args:
        controller: The controller instance
        selector: CSS selector for the element
        timeout: Timeout in milliseconds
        original_requirement: Original user requirement for traceability
        
    Returns:
        TextBySelectorExtractionResult with extracted text and metadata
    """
    try:
        action_result = await controller.act({
            'extract_text_by_selector': {
                'selector': selector,
                'timeout': timeout,
                'original_requirement': original_requirement
            }
        })
        
        if action_result.success:
            data = _parse_content(action_result.extracted_content)
            return TextBySelectorExtractionResult(
                success=True,
                data=data,
                text=data.get('text', action_result.extracted_content),
                metadata=data.get('metadata'),
                element_info=data.get('element_info')
            )
        else:
            return TextBySelectorExtractionResult(
                success=False,
                error=action_result.error
            )
            
    except Exception as e:
        logger.error(f"Error extracting text for selector '{selector}': {str(e)}")
        return TextBySelectorExtractionResult(
            success=False,
            error=str(e)
        )

async def extract_element_attribute(
    controller,
    selector: str,
    attribute: str,
    include_metadata: bool = False,
    timeout: int = 5000,
    original_requirement: str = ""
) -> AttributeExtractionResult:
    """
    Extract an attribute value from an element using its selector.
    This is a pure function that only extracts data, no validation.
    
    Args:
        controller: The controller instance
        selector: CSS selector for the element
        attribute: Name of the attribute to extract
        include_metadata: Whether to include element metadata in result
        timeout: Timeout in milliseconds
        original_requirement: Original user requirement for traceability
        
    Returns:
        AttributeExtractionResult with extracted attribute value and optional metadata
    """
    try:
        action_result = await controller.act({
            'extract_attribute': {
                'selector': selector,
                'attribute': attribute,
                'include_metadata': include_metadata,
                'timeout': timeout,
                'original_requirement': original_requirement
            }
        })
        
        if action_result.success:
            data = _parse_content(action_result.extracted_content)
            return AttributeExtractionResult(
                success=True,
                data=data,
                value=data.get('value', action_result.extracted_content),
                metadata=data.get('metadata'),
                element_info=data.get('element_info')
            )
        else:
            return AttributeExtractionResult(
                success=False,
                error=action_result.error
            )
            
    except Exception as e:
        logger.error(f"Error extracting attribute '{attribute}' for selector '{selector}': {str(e)}")
        return AttributeExtractionResult(
            success=False,
            error=str(e)
        )