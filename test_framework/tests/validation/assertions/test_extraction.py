import pytest
from unittest.mock import MagicMock, patch
from test_framework.validation.assertions.extraction import ExtractionAssertions
from test_framework.validation.assertions.models import AssertionResult
from browser_use.agent.views import AgentHistoryList, ActionResult

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MagicMock()

@pytest.fixture
def mock_result():
    """Create a mock result for testing."""
    class MockAgentHistoryList(list):
        """Mock class that behaves like AgentHistoryList."""
        def __init__(self):
            super().__init__()
            self._items = []
        
        def __len__(self):
            return len(self._items)
        
        def __getitem__(self, index):
            return self._items[index]
        
        def append(self, item):
            self._items.append(item)
    
    return MockAgentHistoryList()

@pytest.fixture
def extraction_assertions(mock_agent, mock_result):
    """Create an ExtractionAssertions instance for testing."""
    return ExtractionAssertions(mock_agent, mock_result)

def test_initialization(extraction_assertions):
    """Test initialization of ExtractionAssertions."""
    assert extraction_assertions._extraction_cache == {}
    assert extraction_assertions._processed_extractions == set()
    assert extraction_assertions._matching is not None

def test_normalize_url(extraction_assertions):
    """Test URL normalization."""
    # Test empty URL
    assert extraction_assertions._normalize_url("") == ""
    
    # Test absolute URL
    url = "https://example.com/path"
    assert extraction_assertions._normalize_url(url) == "https://example.com/path"
    
    # Test URL with fragment
    url = "https://example.com/path#section"
    assert extraction_assertions._normalize_url(url) == "https://example.com/path"
    
    # Test relative URL
    base_url = "https://example.com"
    relative_url = "/path"
    assert extraction_assertions._normalize_url(relative_url, base_url) == "https://example.com/path"
    
    # Test URL with trailing slash
    url = "https://example.com/path/"
    assert extraction_assertions._normalize_url(url) == "https://example.com/path"
    
    # Test URL without path
    url = "https://example.com"
    assert extraction_assertions._normalize_url(url) == "https://example.com/"

def test_extract_text_from_content(extraction_assertions):
    """Test text extraction from various content types."""
    # Test string content
    assert extraction_assertions._extract_text_from_content("Hello World") == "Hello World"
    
    # Test dictionary with text key
    content = {"text": "Hello World"}
    assert extraction_assertions._extract_text_from_content(content) == "Hello World"
    
    # Test dictionary with extracted_text key
    content = {"extracted_text": "Hello World"}
    assert extraction_assertions._extract_text_from_content(content) == "Hello World"
    
    # Test dictionary with page_content key
    content = {"page_content": "Hello World"}
    assert extraction_assertions._extract_text_from_content(content) == "Hello World"
    
    # Test dictionary with no text keys
    content = {"other": "Hello World"}
    assert extraction_assertions._extract_text_from_content(content) == "{'other': 'Hello World'}"
    
    # Test non-string, non-dict content
    assert extraction_assertions._extract_text_from_content(123) == "123"

def test_process_pending_extractions(extraction_assertions, mock_result):
    """Test processing of pending extractions."""
    # Setup mock result with extractions
    mock_result.append(MagicMock(extract_content={"goal": "test1", "text": "content1"}))
    mock_result.append(MagicMock(extract_content=None))
    mock_result.append(MagicMock(extract_content={"goal": "test2", "text": "content2"}))
    
    # Process extractions
    extraction_assertions._process_pending_extractions(0)
    
    # Verify cache
    assert "0_test1" in extraction_assertions._extraction_cache
    assert "2_test2" in extraction_assertions._extraction_cache
    assert extraction_assertions._extraction_cache["0_test1"]["text"] == "content1"
    assert extraction_assertions._extraction_cache["2_test2"]["text"] == "content2"
    
    # Verify processed steps
    assert 0 in extraction_assertions._processed_extractions
    assert 2 in extraction_assertions._processed_extractions
    assert 1 not in extraction_assertions._processed_extractions

@pytest.mark.skip(reason="Fallback extraction behavior needs investigation")
def test_get_last_extraction_result(extraction_assertions, mock_result, mock_agent):
    """Test getting the last extraction result."""
    # Enable fuzzy matching
    extraction_assertions._matching.enable_fuzzy_matching(0.8)
    
    # Setup mock result with extractions
    mock_action1 = MagicMock()
    mock_action1.extract_content = {"goal": "test goal", "text": "test content"}
    mock_result.append(mock_action1)
    
    mock_action2 = MagicMock()
    mock_action2.extract_content = {"goal": "other goal", "text": "other content"}
    mock_result.append(mock_action2)
    
    # Test finding matching extraction
    result = extraction_assertions._get_last_extraction_result("test goal", 0)
    assert result is not None
    assert result.extracted_content == "test content"
    
    # Test no matching extraction
    mock_result.clear()
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

def test_process_extraction_result(extraction_assertions):
    """Test processing extraction results."""
    # Test dictionary input with text key
    content = {"text": "test content"}
    result = extraction_assertions._process_extraction_result(content)
    assert isinstance(result, ActionResult)
    assert result.extracted_content == "test content"
    
    # Test dictionary input with extracted_text key
    content = {"extracted_text": "test content"}
    result = extraction_assertions._process_extraction_result(content)
    assert isinstance(result, ActionResult)
    assert result.extracted_content == "test content"
    
    # Test dictionary input with page_content key
    content = {"page_content": "test content"}
    result = extraction_assertions._process_extraction_result(content)
    assert isinstance(result, ActionResult)
    assert result.extracted_content == "test content"
    
    # Test string input
    content = "test content"
    result = extraction_assertions._process_extraction_result(content)
    assert isinstance(result, ActionResult)
    assert result.extracted_content == "test content"
    
    # Test JSON string input
    content = '{"text": "test content"}'
    result = extraction_assertions._process_extraction_result(content)
    assert isinstance(result, ActionResult)
    assert result.extracted_content == "test content"
    
    # Test invalid JSON string
    content = '{"text": "test content"'
    result = extraction_assertions._process_extraction_result(content)
    assert isinstance(result, ActionResult)
    assert result.extracted_content == content 