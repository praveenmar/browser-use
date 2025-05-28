import pytest
from datetime import datetime
from test_framework.validation.assertions.models import (
    AssertionType,
    AssertionResult,
    ListAssertionResult,
    ExtractionResult,
    StepContext,
    VerificationContext,
    MatchingResult,
    ListMatchingResult,
    ExtractionMetadata,
    VerificationMetadata,
    StepMetadata
)
from unittest.mock import MagicMock

def test_assertion_type():
    """Test AssertionType enum values."""
    assert AssertionType.TEXT.value == "text"
    assert AssertionType.LINK.value == "link"
    assert AssertionType.ATTRIBUTE.value == "attribute"
    assert AssertionType.LIST.value == "list"
    assert AssertionType.MULTI_VALUE.value == "multi_value"

def test_assertion_result():
    """Test AssertionResult creation and defaults."""
    # Test with minimal required fields
    result = AssertionResult(success=True)
    assert result.success is True
    assert result.error_code is None
    assert result.message is None
    assert result.metadata is None

    # Test with all fields
    metadata = {"key": "value"}
    result = AssertionResult(
        success=False,
        error_code="TEST_ERROR",
        message="Test message",
        metadata=metadata
    )
    assert result.success is False
    assert result.error_code == "TEST_ERROR"
    assert result.message == "Test message"
    assert result.metadata == metadata

def test_list_assertion_result():
    """Test ListAssertionResult creation and defaults."""
    # Test with minimal required fields
    result = ListAssertionResult(success=True)
    assert result.success is True
    assert result.matched_items is None
    assert result.unmatched_items is None
    assert result.total_items == 0
    assert result.matched_count == 0

    # Test with all fields
    matched = [{"text": "Item 1"}]
    unmatched = [{"text": "Item 2"}]
    metadata = {"key": "value"}
    result = ListAssertionResult(
        success=True,
        error_code="TEST_ERROR",
        message="Test message",
        metadata=metadata,
        matched_items=matched,
        unmatched_items=unmatched,
        total_items=2,
        matched_count=1
    )
    assert result.success is True
    assert result.error_code == "TEST_ERROR"
    assert result.message == "Test message"
    assert result.metadata == metadata
    assert result.matched_items == matched
    assert result.unmatched_items == unmatched
    assert result.total_items == 2
    assert result.matched_count == 1

def test_extraction_result():
    """Test ExtractionResult creation and defaults."""
    # Test with required fields
    result = ExtractionResult(
        text="Test text",
        confidence=0.95,
        source="test_source"
    )
    assert result.text == "Test text"
    assert result.confidence == 0.95
    assert result.source == "test_source"
    assert result.metadata is None

    # Test with all fields
    metadata = {"key": "value"}
    result = ExtractionResult(
        text="Test text",
        confidence=0.95,
        source="test_source",
        metadata=metadata
    )
    assert result.text == "Test text"
    assert result.confidence == 0.95
    assert result.source == "test_source"
    assert result.metadata == metadata

def test_step_context():
    """Test StepContext creation and defaults."""
    # Test with required fields
    context = StepContext(
        step_number=1,
        requirements=["req1", "req2"],
        extractions={"key": "value"},
        assertions=[]
    )
    assert context.step_number == 1
    assert context.requirements == ["req1", "req2"]
    assert context.extractions == {"key": "value"}
    assert context.assertions == []

    # Test with assertions
    assertion = AssertionResult(success=True)
    context = StepContext(
        step_number=1,
        requirements=["req1"],
        extractions={},
        assertions=[assertion]
    )
    assert len(context.assertions) == 1
    assert context.assertions[0] == assertion

def test_verification_context():
    """Test VerificationContext creation and defaults."""
    # Test with required fields
    context = VerificationContext(
        current_step=1,
        step_contexts={},
        lookahead_window=3,
        similarity_threshold=0.8
    )
    assert context.current_step == 1
    assert context.step_contexts == {}
    assert context.lookahead_window == 3
    assert context.similarity_threshold == 0.8

    # Test with step contexts
    step_context = StepContext(
        step_number=1,
        requirements=["req1"],
        extractions={},
        assertions=[]
    )
    context = VerificationContext(
        current_step=1,
        step_contexts={1: step_context},
        lookahead_window=3,
        similarity_threshold=0.8
    )
    assert context.step_contexts[1] == step_context

def test_matching_result():
    """Test MatchingResult creation and defaults."""
    # Test with required fields
    result = MatchingResult(
        is_match=True,
        similarity=0.95
    )
    assert result.is_match is True
    assert result.similarity == 0.95
    assert result.matched_text is None
    assert result.metadata is None

    # Test with all fields
    metadata = {"key": "value"}
    result = MatchingResult(
        is_match=True,
        similarity=0.95,
        matched_text="matched text",
        metadata=metadata
    )
    assert result.is_match is True
    assert result.similarity == 0.95
    assert result.matched_text == "matched text"
    assert result.metadata == metadata

def test_list_matching_result():
    """Test ListMatchingResult creation and defaults."""
    # Test with required fields
    result = ListMatchingResult(
        matched_items=[],
        unmatched_items=[],
        total_items=0,
        matched_count=0,
        match_all=True
    )
    assert result.matched_items == []
    assert result.unmatched_items == []
    assert result.total_items == 0
    assert result.matched_count == 0
    assert result.match_all is True
    assert result.metadata is None

    # Test with all fields
    metadata = {"key": "value"}
    result = ListMatchingResult(
        matched_items=[{"text": "Item 1"}],
        unmatched_items=[{"text": "Item 2"}],
        total_items=2,
        matched_count=1,
        match_all=False,
        metadata=metadata
    )
    assert result.matched_items == [{"text": "Item 1"}]
    assert result.unmatched_items == [{"text": "Item 2"}]
    assert result.total_items == 2
    assert result.matched_count == 1
    assert result.match_all is False
    assert result.metadata == metadata

def test_extraction_metadata():
    """Test ExtractionMetadata creation and defaults."""
    # Test with required fields
    metadata = ExtractionMetadata(
        source="test_source",
        timestamp=datetime.now().isoformat(),
        confidence=0.95
    )
    assert metadata.source == "test_source"
    assert isinstance(metadata.timestamp, str)
    assert metadata.confidence == 0.95
    assert metadata.context is None

    # Test with all fields
    context = {"key": "value"}
    metadata = ExtractionMetadata(
        source="test_source",
        timestamp=datetime.now().isoformat(),
        confidence=0.95,
        context=context
    )
    assert metadata.source == "test_source"
    assert isinstance(metadata.timestamp, str)
    assert metadata.confidence == 0.95
    assert metadata.context == context

def test_verification_metadata():
    """Test VerificationMetadata creation and defaults."""
    # Test with required fields
    metadata = VerificationMetadata(
        requirement="test requirement",
        expected_value="expected",
        actual_value="actual"
    )
    assert metadata.requirement == "test requirement"
    assert metadata.expected_value == "expected"
    assert metadata.actual_value == "actual"
    assert metadata.similarity is None
    assert metadata.context is None

    # Test with all fields
    context = {"key": "value"}
    metadata = VerificationMetadata(
        requirement="test requirement",
        expected_value="expected",
        actual_value="actual",
        similarity=0.95,
        context=context
    )
    assert metadata.requirement == "test requirement"
    assert metadata.expected_value == "expected"
    assert metadata.actual_value == "actual"
    assert metadata.similarity == 0.95
    assert metadata.context == context

def test_step_metadata():
    """Test StepMetadata creation and defaults."""
    # Test with required fields
    metadata = StepMetadata(
        step_number=1,
        requirements=["req1"],
        extractions=[],
        assertions=[]
    )
    assert metadata.step_number == 1
    assert metadata.requirements == ["req1"]
    assert metadata.extractions == []
    assert metadata.assertions == []
    assert metadata.context is None

    # Test with all fields
    context = {"key": "value"}
    extraction = ExtractionResult(
        text="Test text",
        confidence=0.95,
        source="test_source"
    )
    assertion = AssertionResult(success=True)
    metadata = StepMetadata(
        step_number=1,
        requirements=["req1"],
        extractions=[extraction],
        assertions=[assertion],
        context=context
    )
    assert metadata.step_number == 1
    assert metadata.requirements == ["req1"]
    assert len(metadata.extractions) == 1
    assert metadata.extractions[0] == extraction
    assert len(metadata.assertions) == 1
    assert metadata.assertions[0] == assertion
    assert metadata.context == context

def test_get_last_extraction_result(extraction_assertions, mock_result, mock_agent):
    """Test getting the last extraction result."""
    # Setup mock result with extractions
    mock_action1 = MagicMock()
    mock_action1.extract_content = {"goal": "test goal", "text": "test content"}
    mock_result.append(mock_action1)

    mock_action2 = MagicMock()
    mock_action2.extract_content = {"goal": "other goal", "text": "other content"}
    mock_result.append(mock_action2)

    # Test finding matching extraction (should not call extract_text)
    result = extraction_assertions._get_last_extraction_result("test goal", 0)
    assert result is not None
    assert result.extracted_content == "test content"

    # Remove the matching extraction to force fallback
    mock_result.clear()
    # Test no matching extraction (should call extract_text)
    mock_agent.extract_text.reset_mock()
    mock_agent.extract_text.return_value = {"extracted_text": "direct content"}
    result = extraction_assertions._get_last_extraction_result("unknown goal", 0)
    assert result is not None
    assert result.extracted_content == "direct content"

    # Test failed direct extraction
    mock_agent.extract_text.reset_mock()
    mock_agent.extract_text.side_effect = Exception("Extraction failed")
    result = extraction_assertions._get_last_extraction_result("unknown goal", 0)
    assert result is None 