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

logger = logging.getLogger("test_framework.validation.validator")

class ValidationMode(str, Enum):
    """Validation modes for different types of checks"""
    EXACT = "exact"  # Exact string matching
    CONTAINS = "contains"  # Substring matching
    RELAXED = "relaxed"  # Case-insensitive, whitespace-normalized matching

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

class ValidationResult(BaseModel):
    """Result of a validation check"""
    success: bool
    expected: Any
    actual: Any
    mode: str
    error_code: Optional[str] = None
    reason: Optional[str] = None
    original_requirement: str

class TestValidator:
    """Low-level validation functions"""
    
    @staticmethod
    def normalize_text(text: str, mode: ValidationMode) -> str:
        """Normalize text based on validation mode"""
        if mode == ValidationMode.RELAXED:
            return " ".join(text.lower().split())
        return text

    @staticmethod
    def validate_text(
        expected: str,
        actual: str,
        mode: ValidationMode = ValidationMode.EXACT,
        original_requirement: Optional[str] = None
    ) -> ValidationResult:
        """Validate text content"""
        logger.debug(f"Validating text: expected='{expected}', actual='{actual}', mode={mode}")
        
        if not actual:
            logger.error("Actual text is empty")
            return ValidationResult(
                success=False,
                expected=expected,
                actual=actual,
                mode=mode.value,
                error_code="TEXT_EMPTY",
                reason="Actual text is empty",
                original_requirement=original_requirement or ""
            )
            
        if mode == ValidationMode.EXACT:
            logger.debug("Using exact matching")
            if expected == actual:
                logger.info("Exact match found")
                return ValidationResult(
                    success=True,
                    expected=expected,
                    actual=actual,
                    mode=mode.value,
                    original_requirement=original_requirement or ""
                )
            else:
                logger.error(f"Exact match failed: expected='{expected}', actual='{actual}'")
                return ValidationResult(
                    success=False,
                    expected=expected,
                    actual=actual,
                    mode=mode.value,
                    error_code="TEXT_MISMATCH",
                    reason=f"Expected exact match of '{expected}', got '{actual}'",
                    original_requirement=original_requirement or ""
                )
                
        elif mode == ValidationMode.CONTAINS:
            logger.debug("Using contains matching")
            if expected in actual:
                logger.info("Contains match found")
                return ValidationResult(
                    success=True,
                    expected=expected,
                    actual=actual,
                    mode=mode.value,
                    original_requirement=original_requirement or ""
                )
            else:
                logger.error(f"Contains match failed: expected='{expected}', actual='{actual}'")
                return ValidationResult(
                    success=False,
                    expected=expected,
                    actual=actual,
                    mode=mode.value,
                    error_code="TEXT_NOT_FOUND",
                    reason=f"Expected to find '{expected}' in '{actual}'",
                    original_requirement=original_requirement or ""
                )
                
        elif mode == ValidationMode.RELAXED:
            logger.debug("Using relaxed matching")
            # Convert to lowercase and remove extra whitespace
            expected_clean = ' '.join(expected.lower().split())
            actual_clean = ' '.join(actual.lower().split())
            
            if expected_clean == actual_clean:
                logger.info("Relaxed match found")
                return ValidationResult(
                    success=True,
                    expected=expected,
                    actual=actual,
                    mode=mode.value,
                    original_requirement=original_requirement or ""
                )
            else:
                logger.error(f"Relaxed match failed: expected='{expected_clean}', actual='{actual_clean}'")
                return ValidationResult(
                    success=False,
                    expected=expected,
                    actual=actual,
                    mode=mode.value,
                    error_code="TEXT_MISMATCH",
                    reason=f"Expected relaxed match of '{expected}', got '{actual}'",
                    original_requirement=original_requirement or ""
                )
                
        logger.error(f"Unknown validation mode: {mode}")
        return ValidationResult(
            success=False,
            expected=expected,
            actual=actual,
            mode=mode.value,
            error_code="INVALID_MODE",
            reason=f"Unknown validation mode: {mode}",
            original_requirement=original_requirement or ""
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
                original_requirement=original_requirement
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
                original_requirement=original_requirement
            )
            
        # Validate href
        logger.debug("Validating link href")
        if not actual_href:
            logger.error("Link href is empty")
            return ValidationResult(
                success=False,
                error_code="LINK_HREF_EMPTY",
                reason="Link href is empty",
                original_requirement=original_requirement
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
            original_requirement=original_requirement
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
                original_requirement=original_requirement
            )
            
        # Use text validation for attribute values
        logger.debug("Using text validation for attribute")
        return TestValidator.validate_text(
            expected=expected_value,
            actual=actual_value,
            mode=mode,
            original_requirement=original_requirement
        ) 