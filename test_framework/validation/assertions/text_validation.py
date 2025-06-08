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
    async def verify_text(
        self,
        expected_text: str,
        case_sensitive: bool = True,
        mode: ValidationMode = ValidationMode.EXACT,
        original_requirement: str = ""
    ) -> AssertionResult:
        """Verify text content with configurable case sensitivity.
        
        Args:
            expected_text: The expected text to verify
            case_sensitive: Whether to perform case-sensitive matching
            mode: The validation mode (EXACT, CONTAINS, or RELAXED)
            original_requirement: The original requirement text
            
        Returns:
            AssertionResult: Result of the verification
        """
        logger.debug(f"🔍 Starting text verification: mode={mode}, case_sensitive={case_sensitive}")
        logger.debug(f"📝 Expected text: '{expected_text}'")
        
        try:
            # Get the current page
            page = self.browser.page
            if not page:
                logger.error("❌ No browser page available")
                return AssertionResult(
                    success=False,
                    error_code="NO_PAGE",
                    message="No browser page available for verification",
                    metadata={
                        "expected_text": expected_text,
                        "mode": mode.value,
                        "case_sensitive": case_sensitive
                    }
                )
            
            # Wait for initial DOM stability
            logger.debug("⏳ Waiting for initial DOM stability...")
            await self.browser.wait_for_dom_stability()
            
            # Get the current page content
            content = await page.content()
            if not content:
                logger.error("❌ No page content available")
                return AssertionResult(
                    success=False,
                    error_code="NO_CONTENT",
                    message="No page content available for verification",
                    metadata={
                        "expected_text": expected_text,
                        "mode": mode.value,
                        "case_sensitive": case_sensitive
                    }
                )
            
            # Use validator to check text
            logger.debug("🔍 Validating text content...")
            validation_result = await self.validator.validate_text(
                expected=expected_text,
                actual=content,
                mode=mode,
                case_sensitive=case_sensitive,
                original_requirement=original_requirement,
                page=page,  # Pass the page for basic checks
                browser_session=self.browser  # Pass the browser session for enhanced checks
            )
            
            # Update metadata with actual values
            metadata = {
                "expected_text": expected_text,
                "actual_text": content,
                "validation_mode": mode.value,
                "case_sensitive": case_sensitive,
                "validation_result": validation_result.metadata
            }
            
            if validation_result.success:
                logger.info(f"✅ Text verification passed: {expected_text}")
                return AssertionResult(
                    success=True,
                    message=f"Text verification passed: {expected_text}",
                    metadata=metadata
                )
            else:
                logger.warning(f"⚠️ Text verification failed: {validation_result.reason}")
                return AssertionResult(
                    success=False,
                    error_code=validation_result.error_code,
                    message=validation_result.reason,
                    metadata=metadata
                )
                
        except Exception as e:
            logger.error(f"❌ Error during text verification: {str(e)}")
            return AssertionResult(
                success=False,
                error_code="VERIFICATION_ERROR",
                message=f"Error during text verification: {str(e)}",
                metadata={
                    "expected_text": expected_text,
                    "mode": mode.value,
                    "case_sensitive": case_sensitive,
                    "error": str(e)
                }
            ) 