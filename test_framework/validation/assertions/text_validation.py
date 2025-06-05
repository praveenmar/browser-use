"""
Text validation module for test framework.
Provides functionality for validating text content with configurable case sensitivity.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import logging

from browser_use.agent.views import ActionResult
from ..validator import TestValidator, ValidationMode, ValidationResult

logger = logging.getLogger("test_framework.validation.assertions.text_validation")

@dataclass
class AssertionResult:
    """Result of an assertion check"""
    success: bool
    error_code: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TextValidation:
    """Text validation functionality with configurable case sensitivity"""
    
    @staticmethod
    def verify_text(
        action_result: ActionResult,
        expected_text: str,
        exact: bool = True,
        case_sensitive: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssertionResult:
        """Verify text content with configurable case sensitivity.
        
        Args:
            action_result: The action result containing the text to verify
            expected_text: The expected text to match against
            exact: Whether to require an exact match
            case_sensitive: Whether to perform case-sensitive matching
            metadata: Additional metadata for the verification
            
        Returns:
            AssertionResult: Result of the verification
        """
        logger.debug(f"Verifying text: expected='{expected_text}', exact={exact}, case_sensitive={case_sensitive}")
        
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
            
        # Use the validator to check the text with configurable case sensitivity
        mode = ValidationMode.EXACT if exact else ValidationMode.CONTAINS
        logger.debug(f"Validating text with mode: {mode}, case_sensitive: {case_sensitive}")
        result = TestValidator.validate_text(
            expected=expected_text,
            actual=actual_text,
            mode=mode,
            case_sensitive=case_sensitive,
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
            "case_sensitive": case_sensitive,
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