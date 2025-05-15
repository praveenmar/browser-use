import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, TypedDict, Union
from patchright.async_api import expect, Page, Locator, TimeoutError as PatchwrightTimeoutError
from browser_use.agent.views import ActionResult
from enum import Enum, auto
import urllib.parse
from pydantic import BaseModel, Field

logger = logging.getLogger("test_framework.validation.assertions")

@dataclass
class VerifyTextAction:
    text: str
    exact: bool = False
    timeout: int = 5000
    original_requirement: str = ""

@dataclass
class VerifyLinkAction:
    text: str
    href: str
    exact: bool = False
    timeout: int = 5000
    original_requirement: str = ""

# Define standard return type schemas
class ExtractedTextResult(TypedDict):
    success: bool
    text: Optional[str]
    message: Optional[str]
    original_requirement: str

class ExtractedLinkResult(TypedDict):
    success: bool
    text: Optional[str]
    href: Optional[str]
    is_relative: Optional[bool]
    base_url: Optional[str]
    message: Optional[str]
    original_requirement: str

# Define validation modes
class ValidationMode(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    RELAXED = "relaxed"

# Define validation error codes grouped by category
class ValidationErrorCode(str, Enum):
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

# Define Pydantic model for validation results
class ValidationResult(BaseModel):
    success: bool
    expected: Any
    actual: Any
    mode: str
    error_code: Optional[str] = None
    reason: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "success": False,
                    "expected": "Login",
                    "actual": "Log In",
                    "mode": "exact",
                    "error_code": "TEXT_EXACT_MISMATCH",
                    "reason": "Expected exact 'Login', got 'Log In'"
                }
            ]
        }

# Pure validation functions
def validate_text(expected: str, actual: str, mode: ValidationMode = ValidationMode.EXACT) -> ValidationResult:
    """
    Validates text content against expected value using specified mode.
    
    Args:
        expected: The expected text
        actual: The actual text found in the DOM
        mode: ValidationMode (exact, contains, relaxed)
        
    Returns:
        ValidationResult with success flag, error code, and details
    """
    if actual is None:
        return ValidationResult(
            success=False,
            expected=expected,
            actual=None,
            mode=mode,
            error_code=ValidationErrorCode.ELEMENT_NOT_FOUND,
            reason="Actual text is None (element likely not found)"
        )
    
    if mode == ValidationMode.EXACT:
        success = expected == actual
        if not success:
            return ValidationResult(
                success=False,
                expected=expected,
                actual=actual,
                mode=mode,
                error_code=ValidationErrorCode.TEXT_EXACT_MISMATCH,
                reason=f"Expected exact '{expected}', got '{actual}'"
            )
    elif mode == ValidationMode.CONTAINS:
        success = expected in actual
        if not success:
            return ValidationResult(
                success=False,
                expected=expected,
                actual=actual,
                mode=mode,
                error_code=ValidationErrorCode.TEXT_CONTAINS_FAILURE,
                reason=f"Expected '{expected}' to be contained in '{actual}'"
            )
    elif mode == ValidationMode.RELAXED:
        # Case-insensitive, whitespace-normalized comparison
        normalized_expected = " ".join(expected.lower().split())
        normalized_actual = " ".join(actual.lower().split())
        success = normalized_expected in normalized_actual
        if not success:
            return ValidationResult(
                success=False,
                expected=expected,
                actual=actual,
                mode=mode,
                error_code=ValidationErrorCode.TEXT_RELAXED_MISMATCH,
                reason=f"Expected '{normalized_expected}' to be contained in '{normalized_actual}' (relaxed mode)"
            )
    else:
        return ValidationResult(
            success=False,
            expected=expected,
            actual=actual,
            mode=mode,
            error_code=ValidationErrorCode.INVALID_VALIDATION_MODE,
            reason=f"Unknown validation mode: {mode}"
        )
    
    # Success case
    return ValidationResult(
        success=True,
        expected=expected,
        actual=actual,
        mode=mode,
        error_code=None,
        reason=None
    )

def validate_link(expected_text: str, expected_href: str, 
                 actual_text: Optional[str], actual_href: Optional[str],
                 base_url: Optional[str] = None, 
                 is_relative: Optional[bool] = None,
                 mode: ValidationMode = ValidationMode.EXACT) -> ValidationResult:
    """
    Validates link text and href against expected values.
    
    Args:
        expected_text: Expected link text
        expected_href: Expected href value
        actual_text: Actual link text found
        actual_href: Actual href found
        base_url: Base URL for resolving relative URLs
        is_relative: Whether the actual href is a relative URL
        mode: ValidationMode (exact, contains, relaxed)
        
    Returns:
        ValidationResult with success flag, error code, and details
    """
    if actual_text is None or actual_href is None:
        return ValidationResult(
            success=False,
            expected={"text": expected_text, "href": expected_href},
            actual={"text": actual_text, "href": actual_href},
            mode=mode,
            error_code=ValidationErrorCode.ELEMENT_ATTRIBUTE_MISSING,
            reason="Link element not found or missing attributes"
        )
    
    # Normalize URLs for comparison
    normalized_expected_href = expected_href
    normalized_actual_href = actual_href
    
    # Handle relative URLs if needed
    if is_relative and base_url:
        normalized_actual_href = urllib.parse.urljoin(base_url, actual_href)
        # If expected URL doesn't have scheme but actual does, try to normalize
        if not (expected_href.startswith("http://") or expected_href.startswith("https://")):
            normalized_expected_href = urllib.parse.urljoin(base_url, expected_href)
    
    # Validate text first
    text_result = validate_text(expected_text, actual_text, mode)
    
    # Then validate href
    if not text_result.success:
        return ValidationResult(
            success=False,
            expected={"text": expected_text, "href": expected_href},
            actual={"text": actual_text, "href": actual_href},
            mode=mode,
            error_code=ValidationErrorCode.LINK_TEXT_MISMATCH,
            reason=f"Link text validation failed: {text_result.reason}"
        )
    
    # URL validation is always exact since it's a programmatic reference
    href_match = normalized_expected_href == normalized_actual_href
    
    if not href_match:
        return ValidationResult(
            success=False,
            expected={"text": expected_text, "href": normalized_expected_href},
            actual={"text": actual_text, "href": normalized_actual_href},
            mode=mode,
            error_code=ValidationErrorCode.LINK_HREF_MISMATCH,
            reason=f"Link href mismatch. Expected: '{normalized_expected_href}', Got: '{normalized_actual_href}'"
        )
    
    return ValidationResult(
        success=True,
        expected={"text": expected_text, "href": normalized_expected_href},
        actual={"text": actual_text, "href": normalized_actual_href},
        mode=mode,
        error_code=None,
        reason=None
    )

class TestAssertions:
    def __init__(self, agent, history):
        self.agent = agent
        self.history = history
        self.page = None

    def _extract_verification_steps(self, task_description: str) -> list:
        """Extract verification steps from the task description"""
        verification_keywords = ["verify", "validate", "check", "assert", "ensure", "confirm"]
        sentences = [s.strip() for s in task_description.replace(' and ', '. ').split('.')]
        return [s for s in sentences if any(k in s for k in verification_keywords)]

    async def _verify_condition(self, verification_step: str):
        """Verify a single condition"""
        try:
            # Ensure we have a valid page object
            if not self.page:
                self.page = await self.agent.browser_context.get_current_page()
            
            if not isinstance(self.page, Page):
                raise ValueError(f"Invalid page object type: {type(self.page)}")
                
            logger.info(f"Verifying: '{verification_step}'")

            # Check if this is a link verification
            if "link" in verification_step.lower() and "href" in verification_step.lower():
                # Extract link text and href
                parts = verification_step.split("'")
                if len(parts) >= 3:
                    link_text = parts[1]
                    href = parts[3]
                    
                    # Find and verify link
                    locator = self.page.get_by_role("link", name=link_text)
                    await expect(locator).to_be_visible(timeout=5000)
                    
                    # Verify href
                    actual_href = await locator.get_attribute("href")
                    if actual_href != href:
                        raise AssertionError(f"Link href mismatch. Expected: '{href}', Got: '{actual_href}'")
                    
                    logger.info(f"✅ Verified link: '{link_text}' with href '{href}'")
                    return
            else:
                # Extract text to verify
                text_to_verify = None
                if "'" in verification_step:
                    parts = verification_step.split("'")
                    if len(parts) >= 2:
                        text_to_verify = parts[1]
                elif '"' in verification_step:
                    parts = verification_step.split('"')
                    if len(parts) >= 2:
                        text_to_verify = parts[1]
                
                if not text_to_verify:
                    raise ValueError("Could not extract text to verify from step")
                
                # Simple text verification
                locator = self.page.get_by_text(text_to_verify)
                await expect(locator).to_be_visible(timeout=5000)
                logger.info(f"✅ Verified: '{text_to_verify}'")

        except PatchwrightTimeoutError as e:
            error_msg = f"❌ Element not found: {str(e)}"
            logger.error(error_msg)
            raise AssertionError(error_msg)
        except Exception as e:
            error_msg = f"❌ Verification failed: {str(e)}"
            logger.error(error_msg)
            raise AssertionError(error_msg)

    @staticmethod
    def register_actions(controller):
        """Register data extraction actions with the controller"""
        
        @controller.action("extract_text")
        async def extract_text(page: Page, action: VerifyTextAction) -> ExtractedTextResult:
            try:
                if not isinstance(page, Page):
                    raise ValueError(f"Invalid page object type: {type(page)}")
                
                # Find text element
                locator = page.get_by_text(action.text, exact=action.exact)
                
                # Check if element exists
                is_visible = await locator.is_visible(timeout=action.timeout)
                
                if is_visible:
                    # Get the actual text content
                    actual_text = await locator.text_content()
                    logger.info(f"[Extract Text] Found: {actual_text}")
                    return ExtractedTextResult(
                        success=True,
                        text=actual_text,
                        message=None,
                        original_requirement=action.original_requirement
                    )
                else:
                    return ExtractedTextResult(
                        success=False,
                        text=None,
                        message=f"Text '{action.text}' not found on page.",
                        original_requirement=action.original_requirement
                    )
                    
            except Exception as e:
                logger.error(f"Error extracting text '{action.text}': {str(e)}")
                return ExtractedTextResult(
                    success=False,
                    text=None,
                    message=f"Error extracting text '{action.text}': {str(e)}",
                    original_requirement=action.original_requirement
                )

        @controller.action("extract_link")
        async def extract_link(page: Page, action: VerifyLinkAction) -> ExtractedLinkResult:
            try:
                if not isinstance(page, Page):
                    raise ValueError(f"Invalid page object type: {type(page)}")
                
                # Find link by text
                locator = page.get_by_role("link", name=action.text, exact=action.exact)
                
                # Check if element exists
                is_visible = await locator.is_visible(timeout=action.timeout)
                
                if is_visible:
                    # Get the actual href and text
                    href = await locator.get_attribute("href")
                    text = await locator.text_content()
                    
                    # Get current page URL for href normalization
                    current_url = page.url
                    
                    # Determine if href is relative
                    is_relative = href and not (href.startswith('http://') or href.startswith('https://'))
                    
                    logger.info(f"[Extract Link] Found: text='{text}', href='{href}', is_relative={is_relative}")
                    
                    return ExtractedLinkResult(
                        success=True,
                        href=href,
                        text=text,
                        is_relative=is_relative,
                        base_url=current_url,
                        message=None,
                        original_requirement=action.original_requirement
                    )
                else:
                    return ExtractedLinkResult(
                        success=False,
                        href=None,
                        text=None,
                        is_relative=None,
                        base_url=None,
                        message=f"Link '{action.text}' not found on page.",
                        original_requirement=action.original_requirement
                    )
                    
            except Exception as e:
                logger.error(f"Error extracting link '{action.text}': {str(e)}")
                return ExtractedLinkResult(
                    success=False,
                    href=None,
                    text=None,
                    is_relative=None,
                    base_url=None,
                    message=f"Error extracting link '{action.text}': {str(e)}",
                    original_requirement=action.original_requirement
                )