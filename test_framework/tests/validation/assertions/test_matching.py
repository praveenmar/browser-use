import pytest
from unittest.mock import MagicMock, patch
from test_framework.validation.assertions.matching import (
    MatchingAssertions,
    MatchType,
    MatchResult,
    FUZZY_MATCH_THRESHOLD,
    CASE_INSENSITIVE_THRESHOLD,
    EXACT_MATCH_THRESHOLD
)
from test_framework.validation.assertions.models import AssertionResult
from browser_use.agent.views import AgentHistoryList

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MagicMock()

@pytest.fixture
def mock_result():
    """Create a mock result for testing."""
    return MagicMock(spec=AgentHistoryList)

@pytest.fixture
def matching_assertions(mock_agent, mock_result):
    """Create a MatchingAssertions instance for testing."""
    return MatchingAssertions(mock_agent, mock_result)

def test_initialization(matching_assertions):
    """Test initialization of MatchingAssertions."""
    assert matching_assertions._similarity_threshold == 1.0
    assert matching_assertions._use_fuzzy_matching is False

def test_enable_fuzzy_matching(matching_assertions):
    """Test enabling fuzzy matching."""
    # Test valid threshold
    matching_assertions.enable_fuzzy_matching(0.8)
    assert matching_assertions._use_fuzzy_matching is True
    assert matching_assertions._similarity_threshold == 0.8
    
    # Test invalid threshold
    with pytest.raises(ValueError):
        matching_assertions.enable_fuzzy_matching(1.5)
    with pytest.raises(ValueError):
        matching_assertions.enable_fuzzy_matching(-0.1)

def test_normalize_text(matching_assertions):
    """Test text normalization."""
    # Test empty text
    assert matching_assertions._normalize_text("") == ""
    assert matching_assertions._normalize_text(None) == ""
    
    # Test case normalization
    assert matching_assertions._normalize_text("Hello World") == "hello world"
    
    # Test whitespace normalization
    assert matching_assertions._normalize_text("  hello   world  ") == "hello world"
    
    # Test punctuation removal
    assert matching_assertions._normalize_text("Hello, World!") == "hello world"
    assert matching_assertions._normalize_text("Hello-World") == "hello world"
    
    # Test combined normalization
    assert matching_assertions._normalize_text("  Hello, World!  ") == "hello world"

def test_calculate_text_similarity_strict(matching_assertions):
    """Test text similarity calculation in strict mode."""
    # Test exact matches
    assert matching_assertions._calculate_text_similarity("hello", "hello") == 1.0
    assert matching_assertions._calculate_text_similarity("Hello", "Hello") == 1.0
    
    # Test case differences (should fail in strict mode)
    assert matching_assertions._calculate_text_similarity("Hello", "hello") == 0.0
    
    # Test whitespace differences (should fail in strict mode)
    assert matching_assertions._calculate_text_similarity("hello", " hello ") == 0.0
    
    # Test punctuation differences (should fail in strict mode)
    assert matching_assertions._calculate_text_similarity("hello", "hello!") == 0.0
    
    # Test empty strings
    assert matching_assertions._calculate_text_similarity("", "") == 1.0
    assert matching_assertions._calculate_text_similarity("hello", "") == 0.0
    assert matching_assertions._calculate_text_similarity("", "world") == 0.0

def test_calculate_text_similarity_fuzzy(matching_assertions):
    """Test text similarity calculation in fuzzy mode."""
    matching_assertions.enable_fuzzy_matching(0.8)
    
    # Test exact matches
    assert matching_assertions._calculate_text_similarity("hello", "hello") == 1.0
    
    # Test case differences (should pass in fuzzy mode)
    similarity = matching_assertions._calculate_text_similarity("Hello", "hello")
    assert similarity == 0.9  # Case difference gives 0.9 similarity
    
    # Test whitespace differences (should pass in fuzzy mode)
    similarity = matching_assertions._calculate_text_similarity("hello", " hello ")
    assert similarity == 0.9  # Whitespace difference gives 0.9 similarity
    
    # Test punctuation differences (should pass in fuzzy mode)
    similarity = matching_assertions._calculate_text_similarity("hello", "hello!")
    assert similarity == 0.9  # Punctuation difference gives 0.9 similarity

def test_find_best_match_strict(matching_assertions):
    """Test finding best match in strict mode."""
    candidates = ["Hello", "World", "Test"]
    
    # Test exact match
    match, score = matching_assertions._find_best_match("Hello", candidates)
    assert match == "Hello"
    assert score == 1.0
    
    # Test case difference (should fail in strict mode)
    match, score = matching_assertions._find_best_match("hello", candidates)
    assert match is None
    assert score == 0.0
    
    # Test empty candidates
    match, score = matching_assertions._find_best_match("hello", [])
    assert match is None
    assert score == 0.0

def test_find_best_match_fuzzy(matching_assertions):
    """Test finding best match in fuzzy mode."""
    matching_assertions.enable_fuzzy_matching(0.8)
    candidates = ["Hello World", "Goodbye World", "Test"]
    
    # Test exact match
    match, score = matching_assertions._find_best_match("Hello World", candidates)
    assert match == "Hello World"
    assert score == 1.0
    
    # Test case difference (should pass in fuzzy mode)
    match, score = matching_assertions._find_best_match("hello world", candidates)
    assert match == "Hello World"
    assert score == 0.9
    
    # Test similar match
    match, score = matching_assertions._find_best_match("hello", candidates)
    assert match == "Hello World"
    assert score > 0.5

def test_verify_text_match_strict(matching_assertions):
    """Test text matching verification in strict mode."""
    # Test exact match
    result = matching_assertions._verify_text_match("Hello", "Hello")
    assert result.success is True
    assert result.error_code is None
    assert result.metadata["similarity"] == 1.0
    
    # Test case difference (should fail in strict mode)
    result = matching_assertions._verify_text_match("Hello", "hello")
    assert result.success is False
    assert result.error_code == "TEXT_MISMATCH"
    assert result.metadata["similarity"] == 0.0
    
    # Test whitespace difference (should fail in strict mode)
    result = matching_assertions._verify_text_match("Hello", " Hello ")
    assert result.success is False
    assert result.error_code == "TEXT_MISMATCH"
    assert result.metadata["similarity"] == 0.0

def test_verify_text_match_fuzzy(matching_assertions):
    """Test text matching verification in fuzzy mode."""
    matching_assertions.enable_fuzzy_matching(0.8)
    
    # Test exact match
    result = matching_assertions._verify_text_match("Hello", "Hello")
    assert result.success is True
    assert result.error_code is None
    assert result.metadata["similarity"] == 1.0
    
    # Test case difference (should pass in fuzzy mode)
    result = matching_assertions._verify_text_match("Hello", "hello")
    assert result.success is True
    assert result.error_code is None
    assert result.metadata["similarity"] == 0.9
    
    # Test custom threshold
    result = matching_assertions._verify_text_match("Hello", "hello", threshold=0.95)
    assert result.success is False
    assert result.error_code == "TEXT_MISMATCH"
    assert result.metadata["similarity"] == 0.9

def test_verify_list_match_strict(matching_assertions):
    """Test list matching verification in strict mode."""
    # Test exact matches
    expected = ["Hello", "World"]
    actual = ["Hello", "World"]
    result = matching_assertions._verify_list_match(expected, actual)
    assert result.success is True
    assert len(result.metadata["matched_items"]) == 2
    assert len(result.metadata["unmatched_items"]) == 0
    
    # Test case differences (should fail in strict mode)
    expected = ["Hello", "World"]
    actual = ["hello", "world"]
    result = matching_assertions._verify_list_match(expected, actual)
    assert result.success is False
    assert len(result.metadata["matched_items"]) == 0
    assert len(result.metadata["unmatched_items"]) == 2
    
    # Test partial matches with match_all=True
    expected = ["Hello", "World", "Test"]
    actual = ["Hello", "World", "Different"]
    result = matching_assertions._verify_list_match(expected, actual, match_all=True)
    assert result.success is False
    assert len(result.metadata["matched_items"]) == 2
    assert len(result.metadata["unmatched_items"]) == 1

def test_verify_list_match_fuzzy(matching_assertions):
    """Test list matching verification in fuzzy mode."""
    matching_assertions.enable_fuzzy_matching(0.8)
    
    # Test exact matches
    expected = ["Hello", "World"]
    actual = ["Hello", "World"]
    result = matching_assertions._verify_list_match(expected, actual)
    assert result.success is True
    assert len(result.metadata["matched_items"]) == 2
    
    # Test case differences (should pass in fuzzy mode)
    expected = ["Hello", "World"]
    actual = ["hello", "world"]
    result = matching_assertions._verify_list_match(expected, actual)
    assert result.success is True
    assert len(result.metadata["matched_items"]) == 2
    
    # Test similar matches
    expected = ["Hello World", "Goodbye World"]
    actual = ["hello world!", "goodbye world!"]
    result = matching_assertions._verify_list_match(expected, actual)
    assert result.success is True
    assert len(result.metadata["matched_items"]) == 2

def test_match_type_enum():
    """Test MatchType enum values and string representation."""
    assert str(MatchType.EXACT) == "exact"
    assert str(MatchType.CONTAINS) == "contains"
    assert str(MatchType.CASE_INSENSITIVE) == "case_insensitive"
    assert str(MatchType.FUZZY) == "fuzzy"
    assert str(MatchType.NO_MATCH) == "no_match"

def test_match_text_exact(matching_assertions):
    """Test exact text matching."""
    result = matching_assertions.match_text("Hello", "Hello")
    assert result["match_type"] == MatchType.EXACT
    assert result["success"] is True
    assert result["similarity_score"] == EXACT_MATCH_THRESHOLD
    assert result["matched_snippet"] == "Hello"

def test_match_text_contains(matching_assertions):
    """Test contains text matching."""
    result = matching_assertions.match_text("Hello", "Hello World")
    assert result["match_type"] == MatchType.CONTAINS
    assert result["success"] is True
    assert result["similarity_score"] == 1.0
    assert result["matched_snippet"] == "Hello World"

def test_match_text_case_insensitive(matching_assertions):
    """Test case-insensitive text matching."""
    result = matching_assertions.match_text("Hello", "hello")
    assert result["match_type"] == MatchType.CASE_INSENSITIVE
    assert result["success"] is True
    assert result["similarity_score"] == CASE_INSENSITIVE_THRESHOLD
    assert result["matched_snippet"] == "hello"

def test_match_text_fuzzy(matching_assertions):
    """Test fuzzy text matching."""
    result = matching_assertions.match_text("Hello", "Hallo", use_fuzzy_matching=True)
    assert result["match_type"] == MatchType.FUZZY
    assert result["success"] is True
    assert result["similarity_score"] >= FUZZY_MATCH_THRESHOLD
    assert result["matched_snippet"] == "Hallo"

def test_match_text_no_match(matching_assertions):
    """Test no match scenario."""
    result = matching_assertions.match_text("Hello", "Goodbye")
    assert result["match_type"] == MatchType.NO_MATCH
    assert result["success"] is False
    assert result["similarity_score"] < FUZZY_MATCH_THRESHOLD
    assert result["matched_snippet"] == "Goodbye"

def test_match_text_empty_strings(matching_assertions):
    """Test matching with empty strings."""
    # Both empty
    result = matching_assertions.match_text("", "")
    assert result["match_type"] == MatchType.EXACT
    assert result["success"] is True
    assert result["similarity_score"] == EXACT_MATCH_THRESHOLD
    
    # One empty
    result = matching_assertions.match_text("", "Hello")
    assert result["match_type"] == MatchType.NO_MATCH
    assert result["success"] is False

def test_match_text_whitespace(matching_assertions):
    """Test matching with whitespace."""
    result = matching_assertions.match_text("Hello", " Hello ")
    assert result["match_type"] == MatchType.NO_MATCH
    assert result["success"] is False

def test_find_best_match(matching_assertions):
    """Test finding best match from candidates."""
    text = "Hello"
    candidates = ["Hello World", "Goodbye", "hello", "Hallo"]
    
    best_match, score = matching_assertions._find_best_match(text, candidates)
    assert best_match == "hello"  # Case-insensitive match
    assert score == CASE_INSENSITIVE_THRESHOLD

def test_find_best_match_no_candidates(matching_assertions):
    """Test finding best match with no candidates."""
    text = "Hello"
    candidates = []
    
    best_match, score = matching_assertions._find_best_match(text, candidates)
    assert best_match is None
    assert score == 0.0

def test_verify_text_match_with_metadata(matching_assertions):
    """Test text match verification with metadata."""
    result = matching_assertions._verify_text_match("Hello", "hello")
    assert result.metadata["match_type"] == "case_insensitive"
    assert result.metadata["similarity"] == CASE_INSENSITIVE_THRESHOLD
    assert result.metadata["normalized_expected"] == "hello"
    assert result.metadata["normalized_actual"] == "hello"

def test_verify_list_match_with_metadata(matching_assertions):
    """Test list match verification with metadata."""
    expected = ["Hello", "World"]
    actual = ["hello", "world"]
    
    result = matching_assertions._verify_list_match(expected, actual)
    assert "matched_items" in result.metadata
    assert "unmatched_items" in result.metadata
    assert "total_items" in result.metadata
    assert "matched_count" in result.metadata
    assert "threshold" in result.metadata

@patch('test_framework.validation.assertions.matching.logger')
def test_match_text_logging(mock_logger, matching_assertions):
    """Test logging in match_text method."""
    matching_assertions.match_text("Hello", "hello")
    
    # Verify debug logs
    debug_calls = [call for call in mock_logger.debug.call_args_list if call]
    assert len(debug_calls) > 0
    
    # Verify info logs for successful match
    info_calls = [call for call in mock_logger.info.call_args_list if call]
    assert len(info_calls) > 0

@patch('test_framework.validation.assertions.matching.logger')
def test_verify_text_match_logging(mock_logger, matching_assertions):
    """Test logging in verify_text_match method."""
    matching_assertions._verify_text_match("Hello", "hello")
    
    # Verify debug logs
    debug_calls = [call for call in mock_logger.debug.call_args_list if call]
    assert len(debug_calls) > 0
    
    # Verify info/warning logs
    info_warning_calls = [call for call in mock_logger.info.call_args_list + mock_logger.warning.call_args_list if call]
    assert len(info_warning_calls) > 0 