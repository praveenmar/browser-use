import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from patchright.async_api import expect, Page, Locator, TimeoutError as PlaywrightTimeoutError


logger = logging.getLogger("test_framework.validation.assertion_module")

class AssertionType(Enum):
    """Enum defining all possible assertion types"""
    # Text Content Assertions
    TEXT_VISIBLE = "text_visible"
    TEXT_CONTAINS = "text_contains"
    TEXT_EQUALS = "text_equals"
    TEXT_NOT_PRESENT = "text_not_present"
    
    # Element State Assertions
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_HIDDEN = "element_hidden"
    ELEMENT_ENABLED = "element_enabled"
    ELEMENT_DISABLED = "element_disabled"
    ELEMENT_EXISTS = "element_exists"
    ELEMENT_SELECTED = "element_selected"
    ELEMENT_FOCUSED = "element_focused"
    ELEMENT_CLICKABLE = "element_clickable"
    
    # Input Field Assertions
    INPUT_VALUE = "input_value"
    INPUT_EMPTY = "input_empty"
    INPUT_READONLY = "input_readonly"
    INPUT_ENABLED = "input_enabled"
    INPUT_DISABLED = "input_disabled"
    
    # URL and Navigation Assertions
    URL_EQUALS = "url_equals"
    URL_CONTAINS = "url_contains"
    PAGE_TITLE = "page_title"
    PAGE_LOADED = "page_loaded"
    
    # Form Assertions
    FORM_VALID = "form_valid"
    FORM_INVALID = "form_invalid"
    FORM_SUBMITTED = "form_submitted"
    FORM_RESET = "form_reset"
    
    # State/Behavior Assertions
    ELEMENT_BECOMES_VISIBLE = "element_becomes_visible"
    ELEMENT_COUNT_EQUALS = "element_count_equals"
    ELEMENT_COUNT_GREATER = "element_count_greater"
    ELEMENT_COUNT_LESS = "element_count_less"
    MODAL_APPEARS = "modal_appears"
    MODAL_DISAPPEARS = "modal_disappears"
    DIALOG_APPEARS = "dialog_appears"
    DIALOG_DISAPPEARS = "dialog_disappears"
    POPUP_APPEARS = "popup_appears"
    POPUP_DISAPPEARS = "popup_disappears"
    
    # Message Assertions
    SUCCESS_MESSAGE = "success_message"
    ERROR_MESSAGE = "error_message"
    TOAST_MESSAGE = "toast_message"
    NOTIFICATION_MESSAGE = "notification_message"
    VALIDATION_MESSAGE = "validation_message"

@dataclass
class VerificationContext:
    """Holds the context for a verification operation"""
    step_number: int
    action_type: str
    verification_step: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    element_selector: Optional[str] = None
    attribute_name: Optional[str] = None
    action_result: Optional[Dict[str, Any]] = None
    original_requirement: str = ""  # Original user requirement
    validation_mode: ValidationMode = ValidationMode.EXACT  # Explicit validation mode
    relaxed_mode: bool = False  # Flag for relaxed validation

    def __post_init__(self):
        """Validate and normalize the context after initialization"""
        if not self.original_requirement:
            raise ValueError("Original requirement must be provided")
        
        # Set validation mode based on relaxed_mode flag
        if self.relaxed_mode:
            self.validation_mode = ValidationMode.RELAXED

class AssertionModule:
    def __init__(self):
        """Initialize the assertion module with all verification handlers"""
        self.verification_handlers = {
            # Text Content Assertions
            AssertionType.TEXT_VISIBLE: self._verify_text_visible,
            AssertionType.TEXT_CONTAINS: self._verify_text_contains,
            AssertionType.TEXT_EQUALS: self._verify_text_equals,
            AssertionType.TEXT_NOT_PRESENT: self._verify_text_not_present,
            
            # Element State Assertions
            AssertionType.ELEMENT_VISIBLE: self._verify_element_visible,
            AssertionType.ELEMENT_HIDDEN: self._verify_element_hidden,
            AssertionType.ELEMENT_ENABLED: self._verify_element_enabled,
            AssertionType.ELEMENT_DISABLED: self._verify_element_disabled,
            AssertionType.ELEMENT_EXISTS: self._verify_element_exists,
            AssertionType.ELEMENT_SELECTED: self._verify_element_selected,
            AssertionType.ELEMENT_FOCUSED: self._verify_element_focused,
            AssertionType.ELEMENT_CLICKABLE: self._verify_element_clickable,
            
            # Input Field Assertions
            AssertionType.INPUT_VALUE: self._verify_input_value,
            AssertionType.INPUT_EMPTY: self._verify_input_empty,
            AssertionType.INPUT_READONLY: self._verify_input_readonly,
            AssertionType.INPUT_ENABLED: self._verify_input_enabled,
            AssertionType.INPUT_DISABLED: self._verify_input_disabled,
            
            # URL and Navigation Assertions
            AssertionType.URL_EQUALS: self._verify_url_equals,
            AssertionType.URL_CONTAINS: self._verify_url_contains,
            AssertionType.PAGE_TITLE: self._verify_page_title,
            AssertionType.PAGE_LOADED: self._verify_page_loaded,
            
            # Form Assertions
            AssertionType.FORM_VALID: self._verify_form_valid,
            AssertionType.FORM_INVALID: self._verify_form_invalid,
            AssertionType.FORM_SUBMITTED: self._verify_form_submitted,
            AssertionType.FORM_RESET: self._verify_form_reset,
            
            # State/Behavior Assertions
            AssertionType.ELEMENT_BECOMES_VISIBLE: self._verify_element_becomes_visible,
            AssertionType.ELEMENT_COUNT_EQUALS: self._verify_element_count_equals,
            AssertionType.ELEMENT_COUNT_GREATER: self._verify_element_count_greater,
            AssertionType.ELEMENT_COUNT_LESS: self._verify_element_count_less,
            AssertionType.MODAL_APPEARS: self._verify_modal_appears,
            AssertionType.MODAL_DISAPPEARS: self._verify_modal_disappears,
            AssertionType.DIALOG_APPEARS: self._verify_dialog_appears,
            AssertionType.DIALOG_DISAPPEARS: self._verify_dialog_disappears,
            AssertionType.POPUP_APPEARS: self._verify_popup_appears,
            AssertionType.POPUP_DISAPPEARS: self._verify_popup_disappears,
            
            # Message Assertions
            AssertionType.SUCCESS_MESSAGE: self._verify_success_message,
            AssertionType.ERROR_MESSAGE: self._verify_error_message,
            AssertionType.TOAST_MESSAGE: self._verify_toast_message,
            AssertionType.NOTIFICATION_MESSAGE: self._verify_notification_message,
            AssertionType.VALIDATION_MESSAGE: self._verify_validation_message
        }

    async def verify_step(self, context: VerificationContext) -> bool:
        """
        Main verification method that handles the verification process
        """
        try:
            # 1. Validate inputs
            self._validate_context(context)
            
            # 2. Determine verification type
            assertion_type = self._determine_assertion_type(context.verification_step)
            
            # 3. Log verification attempt with original requirement
            logger.info(f"🔍 Verifying {assertion_type.value} at step {context.step_number}")
            logger.info(f"Original requirement: {context.original_requirement}")
            logger.debug(f"Action: {context.action_type}")
            logger.debug(f"Expected: '{context.expected_value}'")
            logger.debug(f"Actual: '{context.actual_value}'")
            logger.debug(f"Validation mode: {context.validation_mode}")
            
            # 4. Get the appropriate handler
            handler = self.verification_handlers.get(assertion_type)
            if not handler:
                raise ValueError(f"No handler found for assertion type: {assertion_type}")
            
            # 5. Perform verification
            result = await handler(context)
            
            # 6. Log result with original requirement
            if result:
                logger.info(f"✅ {assertion_type.value} verification passed at step {context.step_number}")
                logger.info(f"Original requirement satisfied: {context.original_requirement}")
            else:
                logger.error(f"❌ {assertion_type.value} verification failed at step {context.step_number}")
                logger.error(f"Failed requirement: {context.original_requirement}")
            
            return result

        except Exception as e:
            logger.error(f"❌ Verification error at step {context.step_number}: {str(e)}")
            logger.error(f"Failed requirement: {context.original_requirement}")
            raise

    def _validate_context(self, context: VerificationContext):
        """Validate the verification context"""
        if not context.verification_step:
            raise ValueError("Verification step is missing")
        if not context.expected_value:
            raise ValueError("Expected value is missing")
        if not context.actual_value:
            raise ValueError("Actual value is missing")

    def _determine_assertion_type(self, verification_step: str) -> AssertionType:
        """
        Determine the type of assertion needed based on the verification step
        """
        # Text Content Assertions
        if "text" in verification_step:
            if "not present" in verification_step:
                return AssertionType.TEXT_NOT_PRESENT
            elif "contains" in verification_step:
                return AssertionType.TEXT_CONTAINS
            elif "equals" in verification_step or "exact" in verification_step:
                return AssertionType.TEXT_EQUALS
            else:
                return AssertionType.TEXT_VISIBLE
        
        # Element State Assertions
        elif "element" in verification_step:
            if "hidden" in verification_step:
                return AssertionType.ELEMENT_HIDDEN
            elif "enabled" in verification_step:
                return AssertionType.ELEMENT_ENABLED
            elif "disabled" in verification_step:
                return AssertionType.ELEMENT_DISABLED
            elif "exists" in verification_step:
                return AssertionType.ELEMENT_EXISTS
            elif "selected" in verification_step:
                return AssertionType.ELEMENT_SELECTED
            elif "focused" in verification_step:
                return AssertionType.ELEMENT_FOCUSED
            elif "clickable" in verification_step:
                return AssertionType.ELEMENT_CLICKABLE
            else:
                return AssertionType.ELEMENT_VISIBLE
        
        # Input Field Assertions
        elif "input" in verification_step:
            if "empty" in verification_step:
                return AssertionType.INPUT_EMPTY
            elif "readonly" in verification_step:
                return AssertionType.INPUT_READONLY
            elif "enabled" in verification_step:
                return AssertionType.INPUT_ENABLED
            elif "disabled" in verification_step:
                return AssertionType.INPUT_DISABLED
            else:
                return AssertionType.INPUT_VALUE
        
        # URL and Navigation Assertions
        elif "url" in verification_step:
            if "contains" in verification_step:
                return AssertionType.URL_CONTAINS
            else:
                return AssertionType.URL_EQUALS
        elif "title" in verification_step:
            return AssertionType.PAGE_TITLE
        elif "loaded" in verification_step:
            return AssertionType.PAGE_LOADED
        
        # Form Assertions
        elif "form" in verification_step:
            if "invalid" in verification_step:
                return AssertionType.FORM_INVALID
            elif "submitted" in verification_step:
                return AssertionType.FORM_SUBMITTED
            elif "reset" in verification_step:
                return AssertionType.FORM_RESET
            else:
                return AssertionType.FORM_VALID
        
        # State/Behavior Assertions
        elif "count" in verification_step:
            if "greater" in verification_step:
                return AssertionType.ELEMENT_COUNT_GREATER
            elif "less" in verification_step:
                return AssertionType.ELEMENT_COUNT_LESS
            else:
                return AssertionType.ELEMENT_COUNT_EQUALS
        elif "modal" in verification_step:
            return AssertionType.MODAL_APPEARS if "appears" in verification_step else AssertionType.MODAL_DISAPPEARS
        elif "dialog" in verification_step:
            return AssertionType.DIALOG_APPEARS if "appears" in verification_step else AssertionType.DIALOG_DISAPPEARS
        elif "popup" in verification_step:
            return AssertionType.POPUP_APPEARS if "appears" in verification_step else AssertionType.POPUP_DISAPPEARS
        
        # Message Assertions
        elif "success" in verification_step:
            return AssertionType.SUCCESS_MESSAGE
        elif "error" in verification_step:
            return AssertionType.ERROR_MESSAGE
        elif "toast" in verification_step:
            return AssertionType.TOAST_MESSAGE
        elif "notification" in verification_step:
            return AssertionType.NOTIFICATION_MESSAGE
        elif "validation" in verification_step:
            return AssertionType.VALIDATION_MESSAGE
        
        # Default to text visible if no specific type is determined
        return AssertionType.TEXT_VISIBLE

    # Text Content Assertions
    async def _verify_text_visible(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_visible()
                return True
            return context.actual_value == context.expected_value
        except PlaywrightTimeoutError:
            return False

    async def _verify_text_contains(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                text = await context.actual_value.text_content()
                return context.expected_value in text
            return context.expected_value in context.actual_value
        except PlaywrightTimeoutError:
            return False

    async def _verify_text_equals(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                text = await context.actual_value.text_content()
                return text == context.expected_value
            return context.actual_value == context.expected_value
        except PlaywrightTimeoutError:
            return False

    async def _verify_text_not_present(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                text = await context.actual_value.text_content()
                return context.expected_value not in text
            return context.expected_value not in context.actual_value
        except PlaywrightTimeoutError:
            return True  # If element not found, text is not present

    # Element State Assertions
    async def _verify_element_visible(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_visible()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_hidden(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_hidden()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_enabled(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_enabled()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_disabled(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_disabled()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_exists(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_attached()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_selected(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_checked()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_focused(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_focused()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_clickable(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_enabled()
                await expect(context.actual_value).to_be_visible()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    # Input Field Assertions
    async def _verify_input_value(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                value = await context.actual_value.input_value()
                return value == context.expected_value
            return context.actual_value == context.expected_value
        except PlaywrightTimeoutError:
            return False

    async def _verify_input_empty(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                value = await context.actual_value.input_value()
                return not value
            return not context.actual_value
        except PlaywrightTimeoutError:
            return False

    async def _verify_input_readonly(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                readonly = await context.actual_value.get_attribute("readonly")
                return readonly is not None
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_input_enabled(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_enabled()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_input_disabled(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_disabled()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    # URL and Navigation Assertions
    async def _verify_url_equals(self, context: VerificationContext) -> bool:
        return context.actual_value == context.expected_value

    async def _verify_url_contains(self, context: VerificationContext) -> bool:
        return context.expected_value in context.actual_value

    async def _verify_page_title(self, context: VerificationContext) -> bool:
        return context.actual_value == context.expected_value

    async def _verify_page_loaded(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    # Form Assertions
    async def _verify_form_valid(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_form_invalid(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_form_submitted(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_form_reset(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    # State/Behavior Assertions
    async def _verify_element_becomes_visible(self, context: VerificationContext) -> bool:
        try:
            if isinstance(context.actual_value, Locator):
                await expect(context.actual_value).to_be_visible()
                return True
            return context.actual_value is True
        except PlaywrightTimeoutError:
            return False

    async def _verify_element_count_equals(self, context: VerificationContext) -> bool:
        return int(context.actual_value) == int(context.expected_value)

    async def _verify_element_count_greater(self, context: VerificationContext) -> bool:
        return int(context.actual_value) > int(context.expected_value)

    async def _verify_element_count_less(self, context: VerificationContext) -> bool:
        return int(context.actual_value) < int(context.expected_value)

    async def _verify_modal_appears(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_modal_disappears(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_dialog_appears(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_dialog_disappears(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_popup_appears(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    async def _verify_popup_disappears(self, context: VerificationContext) -> bool:
        return context.actual_value is True

    # Message Assertions
    async def _verify_success_message(self, context: VerificationContext) -> bool:
        return context.actual_value == context.expected_value

    async def _verify_error_message(self, context: VerificationContext) -> bool:
        return context.actual_value == context.expected_value

    async def _verify_toast_message(self, context: VerificationContext) -> bool:
        return context.actual_value == context.expected_value

    async def _verify_notification_message(self, context: VerificationContext) -> bool:
        return context.actual_value == context.expected_value

    async def _verify_validation_message(self, context: VerificationContext) -> bool:
        return context.actual_value == context.expected_value