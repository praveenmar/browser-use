"""
Test suite for the validation framework.
Tests different validation scenarios and modes.
"""

import pytest
from browser_use.agent.views import ActionResult
import json

from .validator import TestValidator, ValidationMode, ValidationResult
from .assertions import TestAssertions, AssertionResult

class TestValidationFramework:
    """Test cases for the validation framework"""
    
    def test_text_validation_modes(self):
        """Test text validation with different modes"""
        # Test data
        expected = "Hello World"
        exact_match = "Hello World"
        contains_match = "This is Hello World example"
        relaxed_match = "  hello   world  "
        no_match = "Goodbye World"
        
        # EXACT mode
        result = TestValidator.validate_text(
            expected=expected,
            actual=exact_match,
            mode=ValidationMode.EXACT
        )
        assert result.success
        assert result.error_code is None
        
        result = TestValidator.validate_text(
            expected=expected,
            actual=contains_match,
            mode=ValidationMode.EXACT
        )
        assert not result.success
        assert result.error_code == "TEXT_EXACT_MISMATCH"
        
        # CONTAINS mode
        result = TestValidator.validate_text(
            expected=expected,
            actual=contains_match,
            mode=ValidationMode.CONTAINS
        )
        assert result.success
        assert result.error_code is None
        
        result = TestValidator.validate_text(
            expected=expected,
            actual=no_match,
            mode=ValidationMode.CONTAINS
        )
        assert not result.success
        assert result.error_code == "TEXT_CONTAINS_FAILURE"
        
        # RELAXED mode
        result = TestValidator.validate_text(
            expected=expected,
            actual=relaxed_match,
            mode=ValidationMode.RELAXED
        )
        assert result.success
        assert result.error_code is None
        
        result = TestValidator.validate_text(
            expected=expected,
            actual=no_match,
            mode=ValidationMode.RELAXED
        )
        assert not result.success
        assert result.error_code == "TEXT_RELAXED_MISMATCH"

    def test_text_assertions(self):
        """Test text assertions with different scenarios"""
        # Success case - exact match
        action_result = ActionResult(
            success=True,
            extracted_content="Hello World",
            is_done=True
        )
        
        result = TestAssertions.verify_text(
            action_result=action_result,
            expected_text="Hello World",
            exact=True
        )
        assert result.success
        assert result.error_code is None
        
        # Success case - contains match
        action_result = ActionResult(
            success=True,
            extracted_content="This is Hello World example",
            is_done=True
        )
        
        result = TestAssertions.verify_text(
            action_result=action_result,
            expected_text="Hello World",
            exact=False
        )
        assert result.success
        assert result.error_code is None
        
        # Failure case - element not found
        action_result = ActionResult(
            success=False,
            error="Element not found",
            is_done=True
        )
        
        result = TestAssertions.verify_text(
            action_result=action_result,
            expected_text="Hello World",
            exact=True
        )
        assert not result.success
        assert result.error_code == "ELEMENT_NOT_FOUND"
        
        # Failure case - no content
        action_result = ActionResult(
            success=True,
            extracted_content=None,
            is_done=True
        )
        
        result = TestAssertions.verify_text(
            action_result=action_result,
            expected_text="Hello World",
            exact=True
        )
        assert not result.success
        assert result.error_code == "UNKNOWN_ERROR"
    
    def test_link_validation(self):
        """Test link validation with different scenarios"""
        # Test data
        expected_text = "Click Here"
        expected_href = "https://example.com/page"
        
        # Success case - text only
        action_result = ActionResult(
            success=True,
            extracted_content="Click Here",
            is_done=True
        )
        
        result = TestAssertions.verify_link(
            action_result=action_result,
            expected_text=expected_text,
            expected_href="",  # No href validation
            exact=True
        )
        assert result.success
        assert result.error_code is None
        
        # Text mismatch
        action_result = ActionResult(
            success=True,
            extracted_content="Wrong Text",
            is_done=True
        )
        
        result = TestAssertions.verify_link(
            action_result=action_result,
            expected_text=expected_text,
            expected_href=expected_href,
            exact=True
        )
        assert not result.success
        assert result.error_code == "LINK_TEXT_MISMATCH"
        
        # Href validation required but not supported
        action_result = ActionResult(
            success=True,
            extracted_content="Click Here",
            is_done=True
        )
        
        result = TestAssertions.verify_link(
            action_result=action_result,
            expected_text=expected_text,
            expected_href=expected_href,
            exact=True
        )
        assert not result.success
        assert result.error_code == "LINK_HREF_MISMATCH"

    def test_link_assertions(self):
        """Test link assertions with different scenarios"""
        # Success case - text and href match
        action_result = ActionResult(
            success=True,
            extracted_content=json.dumps({
                "text": "Click Here",
                "href": "https://example.com/page",
                "is_relative": False
            }),
            is_done=True
        )
        
        result = TestAssertions.verify_link(
            action_result=action_result,
            expected_text="Click Here",
            expected_href="https://example.com/page",
            exact=True
        )
        assert result.success
        assert result.error_code is None
        
        # Success case - contains match
        action_result = ActionResult(
            success=True,
            extracted_content=json.dumps({
                "text": "Please Click Here to continue",
                "href": "https://example.com/page",
                "is_relative": False
            }),
            is_done=True
        )
        
        result = TestAssertions.verify_link(
            action_result=action_result,
            expected_text="Click Here",
            expected_href="https://example.com/page",
            exact=False
        )
        assert result.success
        assert result.error_code is None
        
        # Failure case - invalid JSON
        action_result = ActionResult(
            success=True,
            extracted_content="Invalid JSON content",
            is_done=True
        )
        
        result = TestAssertions.verify_link(
            action_result=action_result,
            expected_text="Click Here",
            expected_href="https://example.com/page",
            exact=True
        )
        assert not result.success
        assert result.error_code == "LINK_HREF_MISMATCH"
        
        # Failure case - relative URL
        action_result = ActionResult(
            success=True,
            extracted_content=json.dumps({
                "text": "Click Here",
                "href": "/page",
                "is_relative": True,
                "base_url": "https://example.com"
            }),
            is_done=True
        )
        
        result = TestAssertions.verify_link(
            action_result=action_result,
            expected_text="Click Here",
            expected_href="https://example.com/page",
            exact=True
        )
        assert result.success
        assert result.error_code is None
    
    def test_attribute_validation(self):
        """Test attribute validation with different modes"""
        # Test data
        expected = "active"
        
        # Success case - exact match
        action_result = ActionResult(
            success=True,
            extracted_content="active",
            is_done=True
        )
        
        result = TestAssertions.verify_attribute(
            action_result=action_result,
            expected_value=expected
        )
        assert result.success
        assert result.error_code is None
        
        # Value mismatch
        action_result = ActionResult(
            success=True,
            extracted_content="inactive",
            is_done=True
        )
        
        result = TestAssertions.verify_attribute(
            action_result=action_result,
            expected_value=expected
        )
        assert not result.success
        assert result.error_code == "ATTRIBUTE_MISMATCH"
        
        # Relaxed mode
        action_result = ActionResult(
            success=True,
            extracted_content="  ACTIVE  ",
            is_done=True
        )
        
        result = TestAssertions.verify_attribute(
            action_result=action_result,
            expected_value=expected,
            relaxed=True
        )
        assert result.success
        assert result.error_code is None

    def test_attribute_assertions(self):
        """Test attribute assertions with different scenarios"""
        # Success case - exact match
        action_result = ActionResult(
            success=True,
            extracted_content="active",
            is_done=True
        )
        
        result = TestAssertions.verify_attribute(
            action_result=action_result,
            expected_value="active"
        )
        assert result.success
        assert result.error_code is None
        
        # Success case - relaxed match
        action_result = ActionResult(
            success=True,
            extracted_content="  ACTIVE  ",
            is_done=True
        )
        
        result = TestAssertions.verify_attribute(
            action_result=action_result,
            expected_value="active",
            relaxed=True
        )
        assert result.success
        assert result.error_code is None
        
        # Failure case - element not found
        action_result = ActionResult(
            success=False,
            error="Element not found",
            is_done=True
        )
        
        result = TestAssertions.verify_attribute(
            action_result=action_result,
            expected_value="active"
        )
        assert not result.success
        assert result.error_code == "ELEMENT_NOT_FOUND"
        
        # Failure case - no content
        action_result = ActionResult(
            success=True,
            extracted_content=None,
            is_done=True
        )
        
        result = TestAssertions.verify_attribute(
            action_result=action_result,
            expected_value="active"
        )
        assert not result.success
        assert result.error_code == "UNKNOWN_ERROR"

    def test_error_handling(self):
        """Test error handling in validation"""
        # Missing element
        action_result = ActionResult(
            success=False,
            error="Element not found",
            is_done=True
        )
        
        result = TestAssertions.verify_text(
            action_result=action_result,
            expected_text="Any text",
            exact=True
        )
        assert not result.success
        assert result.error_code == "ELEMENT_NOT_FOUND"
        
        # Invalid input
        action_result = ActionResult(
            success=True,
            extracted_content=None,
            is_done=True
        )
        
        result = TestAssertions.verify_text(
            action_result=action_result,
            expected_text="text",
            exact=True
        )
        assert not result.success
        assert result.error_code == "UNKNOWN_ERROR" 