"""
Test runner that orchestrates extraction and validation.
This layer coordinates between the extractor and validator layers.
"""

from typing import Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum
import json

from browser_use.agent.views import ActionResult
from .extractor import BrowserExtractor, ExtractedTextResult, ExtractedLinkResult, ExtractedAttributeResult
from .validator import TestValidator, ValidationMode, ValidationResult

class AssertionType(str, Enum):
    """Types of assertions supported by the assertion runner"""
    TEXT = "text"
    LINK = "link"
    ATTRIBUTE = "attribute"

@dataclass
class AssertionRequest:
    """Request for running an assertion"""
    type: AssertionType
    expected_value: Any
    selector: Optional[str] = None
    mode: ValidationMode = ValidationMode.EXACT
    metadata: Optional[Dict[str, Any]] = None
    original_requirement: str = ""

class AssertionRunner:
    """Orchestrates extraction and validation of test assertions"""
    
    def __init__(self, extractor: BrowserExtractor):
        """Initialize with an extractor instance"""
        self.extractor = extractor
    
    def _map_action_result_to_extraction_result(self, action_result: ActionResult, assertion_type: AssertionType, original_requirement: str) -> Any:
        """Map ActionResult to appropriate extraction result type"""
        if not action_result.success:
            # Create appropriate failure result based on type
            if assertion_type == AssertionType.TEXT:
                return ExtractedTextResult(
                    success=False,
                    text=None,
                    message=action_result.error,
                    original_requirement=original_requirement
                )
            elif assertion_type == AssertionType.LINK:
                return ExtractedLinkResult(
                    success=False,
                    text=None,
                    href=None,
                    is_relative=None,
                    base_url=None,
                    message=action_result.error,
                    original_requirement=original_requirement
                )
            elif assertion_type == AssertionType.ATTRIBUTE:
                return ExtractedAttributeResult(
                    success=False,
                    value=None,
                    message=action_result.error,
                    original_requirement=original_requirement
                )
        
        # Handle successful results
        if assertion_type == AssertionType.TEXT:
            try:
                # Try to parse as JSON first
                data = json.loads(action_result.extracted_content) if action_result.extracted_content else {}
                return ExtractedTextResult(
                    success=True,
                    text=data.get('text'),
                    message=None,
                    original_requirement=original_requirement,
                    metadata=data.get('metadata'),
                    element_info=data.get('element_info')
                )
            except json.JSONDecodeError:
                # If not JSON, treat as simple text
                return ExtractedTextResult(
                    success=True,
                    text=action_result.extracted_content,
                    message=None,
                    original_requirement=original_requirement
                )
        elif assertion_type == AssertionType.LINK:
            try:
                # Try to parse as JSON first
                data = json.loads(action_result.extracted_content) if action_result.extracted_content else {}
                return ExtractedLinkResult(
                    success=True,
                    text=data.get('text'),
                    href=data.get('href'),
                    is_relative=data.get('is_relative'),
                    base_url=data.get('base_url'),
                    message=None,
                    original_requirement=original_requirement,
                    metadata=data.get('metadata'),
                    element_info=data.get('element_info')
                )
            except json.JSONDecodeError:
                # If not JSON, treat as simple text
                return ExtractedLinkResult(
                    success=True,
                    text=action_result.extracted_content,
                    href=None,
                    is_relative=None,
                    base_url=None,
                    message=None,
                    original_requirement=original_requirement
                )
        elif assertion_type == AssertionType.ATTRIBUTE:
            try:
                # Try to parse as JSON first
                data = json.loads(action_result.extracted_content) if action_result.extracted_content else {}
                return ExtractedAttributeResult(
                    success=True,
                    value=data.get('value', action_result.extracted_content),
                    message=None,
                    original_requirement=original_requirement,
                    metadata=data.get('metadata'),
                    element_info=data.get('element_info')
                )
            except json.JSONDecodeError:
                # If not JSON, treat as simple value
                return ExtractedAttributeResult(
                    success=True,
                    value=action_result.extracted_content,
                    message=None,
                    original_requirement=original_requirement
                )
    
    async def run_assertion(self, request: AssertionRequest) -> ValidationResult:
        """Run a single assertion by coordinating extraction and validation"""
        
        if request.type == AssertionType.TEXT:
            # Extract text
            extraction_result = await self.extractor.extract_text(
                selector=request.selector,
                original_requirement=request.original_requirement
            )
            
            if not extraction_result.success:
                return ValidationResult(
                    success=False,
                    expected=request.expected_value,
                    actual=None,
                    mode=request.mode,
                    error_code="ELEMENT_NOT_FOUND",
                    reason=extraction_result.message,
                    original_requirement=request.original_requirement
                )
            
            # Validate text
            return TestValidator.validate_text(
                expected=request.expected_value,
                actual=extraction_result.text if extraction_result.text is not None else "",
                mode=request.mode,
                original_requirement=request.original_requirement
            )
            
        elif request.type == AssertionType.LINK:
            # Extract link
            extraction_result = await self.extractor.extract_link(
                selector=request.selector,
                original_requirement=request.original_requirement
            )
            
            if not extraction_result.success:
                return ValidationResult(
                    success=False,
                    expected=request.expected_value,
                    actual=None,
                    mode=request.mode,
                    error_code="ELEMENT_NOT_FOUND",
                    reason=extraction_result.message,
                    original_requirement=request.original_requirement
                )
            
            # Validate link
            return TestValidator.validate_link(
                expected_text=request.expected_value.get("text", ""),
                expected_href=request.expected_value.get("href", ""),
                actual_text=extraction_result.text if extraction_result.text is not None else "",
                actual_href=extraction_result.href if extraction_result.href is not None else "",
                base_url=request.metadata.get("base_url") if request.metadata else None,
                is_relative=request.metadata.get("is_relative") if request.metadata else None,
                mode=request.mode,
                original_requirement=request.original_requirement
            )
            
        elif request.type == AssertionType.ATTRIBUTE:
            # Extract attribute
            extraction_result = await self.extractor.extract_attribute(
                selector=request.selector,
                attribute=request.metadata.get("attribute") if request.metadata else None,
                original_requirement=request.original_requirement
            )
            
            if not extraction_result.success:
                return ValidationResult(
                    success=False,
                    expected=request.expected_value,
                    actual=None,
                    mode=request.mode,
                    error_code="ELEMENT_NOT_FOUND",
                    reason=extraction_result.message,
                    original_requirement=request.original_requirement
                )
            
            # Validate attribute
            return TestValidator.validate_attribute(
                expected_value=request.expected_value,
                actual_value=extraction_result.value if extraction_result.value is not None else "",
                original_requirement=request.original_requirement
            )
            
        else:
            return ValidationResult(
                success=False,
                expected=request.expected_value,
                actual=None,
                mode=request.mode,
                error_code="INVALID_ASSERTION_TYPE",
                reason=f"Unknown assertion type: {request.type}",
                original_requirement=request.original_requirement
            ) 