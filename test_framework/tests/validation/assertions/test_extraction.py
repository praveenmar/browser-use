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
    assert extraction_assertions._extract_text_from_content(content) == "Hello World"
    
    # Test non-string, non-dict content
    assert extraction_assertions._extract_text_from_content(123) == "123"
    
    # Test nested dictionary
    content = {
        "header": {
            "title": "Main Title",
            "subtitle": "Sub Title"
        },
        "body": {
            "text": "Main content",
            "footer": "Footer text"
        }
    }
    assert extraction_assertions._extract_text_from_content(content) == "Main Title Sub Title Main content Footer text"
    
    # Test nested list
    content = [
        "First item",
        ["Nested item 1", "Nested item 2"],
        "Last item"
    ]
    assert extraction_assertions._extract_text_from_content(content) == "First item Nested item 1 Nested item 2 Last item"
    
    # Test complex nested structure
    content = {
        "menu": {
            "items": [
                {"name": "Home", "url": "/home"},
                {"name": "About", "url": "/about"}
            ],
            "title": "Main Menu"
        },
        "content": {
            "text": "Welcome",
            "sections": [
                {"heading": "Section 1", "text": "Content 1"},
                {"heading": "Section 2", "text": "Content 2"}
            ]
        }
    }
    expected = "Home /home About /about Main Menu Welcome Section 1 Content 1 Section 2 Content 2"
    assert extraction_assertions._extract_text_from_content(content) == expected
    
    # Test with None values
    content = {
        "text": "Hello",
        "empty": None,
        "nested": {
            "value": None,
            "text": "World"
        }
    }
    assert extraction_assertions._extract_text_from_content(content) == "Hello World"
    
    # Test with empty structures
    content = {
        "empty_dict": {},
        "empty_list": [],
        "text": "Hello"
    }
    assert extraction_assertions._extract_text_from_content(content) == "Hello"

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

@pytest.mark.asyncio
async def test_get_last_extraction_result_with_xpath(extraction_assertions, mock_result):
    """Test getting extraction result with XPath targeting."""
    # Setup mock result with extractions containing elements
    mock_action = MagicMock()
    mock_action.extract_content = {
        "elements": [
            {
                "xpath": "//div[@class='header']",
                "text": "Header Text"
            },
            {
                "xpath": "//div[@class='content']",
                "text": "Content Text"
            }
        ]
    }
    mock_result.append(mock_action)
    
    # Test finding element with matching XPath
    result = await extraction_assertions._get_last_extraction_result(
        requirement='verify text "Content Text"',
        current_step=0,
        xpath="//div[@class='content']"
    )
    assert result is not None
    assert result.extracted_content == "Content Text"
    
    # Test with non-matching XPath
    result = await extraction_assertions._get_last_extraction_result(
        requirement='verify text "Content Text"',
        current_step=0,
        xpath="//div[@class='footer']"
    )
    assert result is None
    
    # Test with XPath but no elements in content
    mock_action.extract_content = {"text": "Some Text"}
    result = await extraction_assertions._get_last_extraction_result(
        requirement='verify text "Some Text"',
        current_step=0,
        xpath="//div[@class='content']"
    )
    assert result is not None
    assert result.extracted_content == "Some Text"
    
    # Test with XPath and element containing value instead of text
    mock_action.extract_content = {
        "elements": [
            {
                "xpath": "//input[@type='text']",
                "value": "Input Value"
            }
        ]
    }
    result = await extraction_assertions._get_last_extraction_result(
        requirement='verify text "Input Value"',
        current_step=0,
        xpath="//input[@type='text']"
    )
    assert result is not None
    assert result.extracted_content == "Input Value"
    
    # Test with XPath and nested elements
    mock_action.extract_content = {
        "elements": [
            {
                "xpath": "//div[@class='container']",
                "elements": [
                    {
                        "xpath": "//div[@class='container']//span",
                        "text": "Nested Text"
                    }
                ]
            }
        ]
    }
    result = await extraction_assertions._get_last_extraction_result(
        requirement='verify text "Nested Text"',
        current_step=0,
        xpath="//div[@class='container']//span"
    )
    assert result is not None
    assert result.extracted_content == "Nested Text"

def test_json_case_insensitive(extraction_assertions):
    """Test case-insensitive matching in JSON."""
    extraction_assertions.set_case_sensitive(False)  # Enable case-insensitive matching
    json_content = {"DIGITAL CONTENT AND DEVICES": "Some content"}
    assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

def test_json_whitespace_handling(extraction_assertions):
    """Test whitespace handling in JSON."""
    json_content = {"  Digital Content and Devices  ": "Some content"}
    assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

def test_no_match(extraction_assertions):
    # Test no match cases
    assert extraction_assertions._calculate_text_similarity("Hello", "World") == 0.0
    assert extraction_assertions._calculate_text_similarity("Hello", {"key": "World"}) == 0.0

def test_case_insensitive(extraction_assertions):
    """Test case insensitivity."""
    extraction_assertions.set_case_sensitive(False)  # Enable case-insensitive matching
    assert extraction_assertions._calculate_text_similarity("HELLO", "hello") == 1.0
    assert extraction_assertions._calculate_text_similarity("HELLO", {"HELLO": "world"}) == 1.0

def test_case_sensitive(extraction_assertions):
    """Test case sensitivity."""
    extraction_assertions.set_case_sensitive(True)  # Enable case-sensitive matching
    assert extraction_assertions._calculate_text_similarity("HELLO", "hello") == 0.0
    assert extraction_assertions._calculate_text_similarity("HELLO", {"HELLO": "world"}) == 1.0
    assert extraction_assertions._calculate_text_similarity("HELLO", {"hello": "world"}) == 0.0

def test_whitespace_handling(extraction_assertions):
    """Test whitespace handling."""
    assert extraction_assertions._calculate_text_similarity("  Hello  ", "Hello") == 1.0
    assert extraction_assertions._calculate_text_similarity("Hello", {"  Hello  ": "world"}) == 1.0 