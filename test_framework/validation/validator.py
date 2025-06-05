"""
Validator module for validating test results.
Provides low-level validation functions.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict
from enum import Enum
from pydantic import BaseModel
import urllib.parse
import logging
import re

logger = logging.getLogger("test_framework.validation.validator")

class ValidationMode(Enum):
    """Validation modes for text matching"""
    EXACT = "exact"
    CONTAINS = "contains"
    RELAXED = "relaxed"

class ValidationErrorCode(str, Enum):
    """Standardized error codes for validation failures"""
    # Element presence errors
    ELEMENT_NOT_FOUND = "ELEMENT_NOT_FOUND"
    ELEMENT_ATTRIBUTE_MISSING = "ELEMENT_ATTRIBUTE_MISSING"
    
    # Text validation errors
    TEXT_EMPTY = "TEXT_EMPTY"
    TEXT_EXACT_MISMATCH = "TEXT_EXACT_MISMATCH" 
    TEXT_CONTAINS_FAILURE = "TEXT_CONTAINS_FAILURE"
    TEXT_RELAXED_MISMATCH = "TEXT_RELAXED_MISMATCH"
    
    # Link validation errors
    LINK_TEXT_MISMATCH = "LINK_TEXT_MISMATCH"
    LINK_HREF_MISMATCH = "LINK_HREF_MISMATCH"
    
    # Config/mode errors
    INVALID_VALIDATION_MODE = "INVALID_VALIDATION_MODE"
    INVALID_INPUT = "INVALID_INPUT"
    
    # Other errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    success: bool
    error_code: Optional[str] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TestValidator:
    """Low-level validation functions"""
    
    @staticmethod
    def normalize_text(text: str, mode: ValidationMode, case_sensitive: bool = True) -> str:
        """Normalize text based on validation mode and case sensitivity.
        
        Args:
            text: The text to normalize
            mode: The validation mode
            case_sensitive: Whether to preserve case
            
        Returns:
            str: The normalized text
        """
        if not text:
            return ""
            
        # Apply mode-specific normalization first
        if mode == ValidationMode.RELAXED:
            # Remove extra whitespace and normalize spaces
            text = re.sub(r'\s+', ' ', text.strip())
        else:
            text = text.strip()
            
        # Handle case sensitivity after other normalizations
        if not case_sensitive:
            text = text.lower()
            
        return text

    @staticmethod
    def validate_text(
        expected: str,
        actual: str,
        mode: ValidationMode = ValidationMode.EXACT,
        case_sensitive: bool = True,
        original_requirement: str = ""
    ) -> ValidationResult:
        """Validate text content with configurable case sensitivity.
        
        Args:
            expected: The expected text
            actual: The actual text
            mode: The validation mode
            case_sensitive: Whether to perform case-sensitive matching
            original_requirement: The original requirement text
            
        Returns:
            ValidationResult: Result of the validation
        """
        logger.debug(f"Validating text: mode={mode}, case_sensitive={case_sensitive}")
        logger.debug(f"Expected: '{expected}'")
        logger.debug(f"Actual: '{actual}'")
        
        if not expected or not actual:
            return ValidationResult(
                success=False,
                error_code="EMPTY_TEXT",
                reason="Expected or actual text is empty",
                metadata={
                    "expected": expected,
                    "actual": actual,
                    "mode": mode.value,
                    "case_sensitive": case_sensitive
                }
            )
            
        # Normalize texts based on mode and case sensitivity
        expected_norm = TestValidator.normalize_text(expected, mode, case_sensitive)
        actual_norm = TestValidator.normalize_text(actual, mode, case_sensitive)
        
        logger.debug(f"Normalized expected: '{expected_norm}'")
        logger.debug(f"Normalized actual: '{actual_norm}'")
        
        # Perform validation based on mode
        if mode == ValidationMode.EXACT:
            success = expected_norm == actual_norm
            reason = "Exact match required" if not success else "Exact match found"
        elif mode == ValidationMode.CONTAINS:
            success = expected_norm in actual_norm
            reason = "Expected text not found in actual text" if not success else "Expected text found in actual text"
        else:  # RELAXED
            # Split into words and check if all expected words are present
            expected_words = set(expected_norm.split())
            actual_words = set(actual_norm.split())
            missing_words = expected_words - actual_words
            success = not missing_words
            reason = f"Missing words: {', '.join(missing_words)}" if not success else "All expected words found"
            
        return ValidationResult(
            success=success,
            error_code=None if success else "TEXT_MISMATCH",
            reason=reason,
            metadata={
                "expected": expected,
                "actual": actual,
                "expected_normalized": expected_norm,
                "actual_normalized": actual_norm,
                "mode": mode.value,
                "case_sensitive": case_sensitive,
                "original_requirement": original_requirement
            }
        )

    @staticmethod
    def validate_link(
        expected_text: str,
        expected_href: Optional[str],
        actual_text: str,
        actual_href: Optional[str],
        base_url: Optional[str] = None,
        is_relative: bool = False,
        mode: ValidationMode = ValidationMode.EXACT,
        original_requirement: Optional[str] = None
    ) -> ValidationResult:
        """Validate link text and href"""
        logger.debug(f"Validating link: text='{expected_text}', href='{expected_href}', mode={mode}")
        
        if not actual_text:
            logger.error("Link text is empty")
            return ValidationResult(
                success=False,
                error_code="LINK_TEXT_EMPTY",
                reason="Link text is empty",
                metadata={
                    "expected_text": expected_text,
                    "expected_href": expected_href,
                    "actual_text": actual_text,
                    "actual_href": actual_href,
                    "mode": mode.value,
                    "is_relative": is_relative,
                    "base_url": base_url
                }
            )
            
        # Validate link text
        logger.debug("Validating link text")
        text_result = TestValidator.validate_text(
            expected=expected_text,
            actual=actual_text,
            mode=mode,
            original_requirement=original_requirement
        )
        
        if not text_result.success:
            logger.error(f"Link text validation failed: {text_result.reason}")
            return text_result
            
        # If no href validation needed, return success
        if not expected_href:
            logger.info("No href validation needed")
            return ValidationResult(
                success=True,
                metadata={
                    "expected_text": expected_text,
                    "actual_text": actual_text,
                    "mode": mode.value,
                    "is_relative": is_relative,
                    "base_url": base_url
                }
            )
            
        # Validate href
        logger.debug("Validating link href")
        if not actual_href:
            logger.error("Link href is empty")
            return ValidationResult(
                success=False,
                error_code="LINK_HREF_EMPTY",
                reason="Link href is empty",
                metadata={
                    "expected_href": expected_href,
                    "actual_href": actual_href,
                    "mode": mode.value,
                    "is_relative": is_relative,
                    "base_url": base_url
                }
            )
            
        # Normalize URLs if needed
        if is_relative and base_url:
            logger.debug(f"Normalizing relative URL with base: {base_url}")
            actual_href = urllib.parse.urljoin(base_url, actual_href)
            
        # Validate href
        href_result = TestValidator.validate_text(
            expected=expected_href,
            actual=actual_href,
            mode=mode,
            original_requirement=original_requirement
        )
        
        if not href_result.success:
            logger.error(f"Link href validation failed: {href_result.reason}")
            return href_result
            
        logger.info("Link validation succeeded")
        return ValidationResult(
            success=True,
            metadata={
                "expected_text": expected_text,
                "actual_text": actual_text,
                "mode": mode.value,
                "is_relative": is_relative,
                "base_url": base_url
            }
        )

    @staticmethod
    def validate_attribute(
        expected_value: str,
        actual_value: str,
        mode: ValidationMode = ValidationMode.EXACT,
        original_requirement: Optional[str] = None
    ) -> ValidationResult:
        """Validate attribute value"""
        logger.debug(f"Validating attribute: expected='{expected_value}', actual='{actual_value}', mode={mode}")
        
        if not actual_value:
            logger.error("Attribute value is empty")
            return ValidationResult(
                success=False,
                error_code="ATTRIBUTE_EMPTY",
                reason="Attribute value is empty",
                metadata={
                    "expected_value": expected_value,
                    "actual_value": actual_value,
                    "mode": mode.value,
                    "original_requirement": original_requirement
                }
            )
            
        # Use text validation for attribute values
        logger.debug("Using text validation for attribute")
        return TestValidator.validate_text(
            expected=expected_value,
            actual=actual_value,
            mode=mode,
            original_requirement=original_requirement
        ) 