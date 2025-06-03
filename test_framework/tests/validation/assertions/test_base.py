import pytest
from typing import Dict, Any
from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent
from test_framework.validation.assertions.base import (
    BaseAssertions,
    AssertionResult,
    ListAssertionResult,
    AssertionType
)

@pytest.fixture
def mock_agent(mocker):
    """Create a mock agent."""
    agent = mocker.Mock(spec=Agent)
    return agent

@pytest.fixture
def mock_result(mocker):
    """Create a mock result history."""
    result = mocker.Mock(spec=AgentHistoryList)
    return result

@pytest.fixture
def base_assertions(mock_agent, mock_result):
    """Create a BaseAssertions instance."""
    return BaseAssertions(mock_agent, mock_result)

def test_create_metadata(base_assertions):
    """Test metadata creation."""
    requirement = "Test requirement"
    expected_value = "Expected value"
    content_info = {
        "parsing_status": "success",
        "value": {"text": "Actual value"}
    }
    
    metadata = base_assertions._create_metadata(requirement, expected_value, content_info)
    
    assert metadata["requirement"] == requirement
    assert metadata["expected_value"] == expected_value
    assert metadata["normalization"]["is_relative"] is False
    assert metadata["normalization"]["base_url"] is None
    assert metadata["normalization"]["normalized"] is False
    assert metadata["parsing"] == content_info

@pytest.mark.asyncio
async def test_process_extraction_result_action_result(base_assertions):
    """Test processing an ActionResult."""
    action_result = ActionResult(
        success=True,
        extracted_content="Test content"
    )
    result = base_assertions._process_extraction_result(action_result)
    assert result == action_result

@pytest.mark.asyncio
async def test_process_extraction_result_dict(base_assertions):
    """Test processing a dictionary result."""
    content = {
        "element_text": "Test element text"
    }
    result = base_assertions._process_extraction_result(content)
    assert result.success is True
    assert result.extracted_content == "Test element text"

@pytest.mark.asyncio
async def test_process_extraction_result_page_content(base_assertions):
    """Test processing page content."""
    content = {
        "page_content": "Test page content"
    }
    result = base_assertions._process_extraction_result(content)
    assert result.success is True
    assert result.extracted_content == "Test page content"

@pytest.mark.asyncio
async def test_process_extraction_result_extracted_text(base_assertions):
    """Test processing extracted text."""
    content = {
        "extracted_text": "Test extracted text"
    }
    result = base_assertions._process_extraction_result(content)
    assert result.success is True
    assert result.extracted_content == "Test extracted text"

@pytest.mark.asyncio
async def test_process_extraction_result_raw_text(base_assertions):
    """Test processing raw text."""
    content = "Test raw text"
    result = base_assertions._process_extraction_result(content)
    assert result.success is True
    assert result.extracted_content == "Test raw text"

def test_parse_extracted_content_none(base_assertions):
    """Test parsing None content."""
    result = base_assertions._parse_extracted_content(None, "text")
    
    assert result["parsing_status"] == "unknown"
    assert result["parsing_error"] is None
    assert result["value"] is None

def test_parse_extracted_content_dict(base_assertions):
    """Test parsing dictionary content."""
    content = {
        "text": "Test text"
    }
    
    result = base_assertions._parse_extracted_content(content, "text")
    
    assert result["parsing_status"] == "dict"
    assert result["parsing_error"] is None
    assert result["value"] == {"text": "Test text"}

def test_parse_extracted_content_json(base_assertions):
    """Test parsing JSON content."""
    content = '{"text": "Test JSON text"}'
    
    result = base_assertions._parse_extracted_content(content, "text")
    
    assert result["parsing_status"] == "json"
    assert result["parsing_error"] is None
    assert result["value"] == {"text": "Test JSON text"}

def test_parse_extracted_content_invalid_json(base_assertions):
    """Test parsing invalid JSON content."""
    content = '{"text": "Test invalid JSON text"'
    
    result = base_assertions._parse_extracted_content(content, "text")
    
    assert result["parsing_status"] == "raw"
    assert result["parsing_error"] is not None
    assert result["value"] == {"text": content}

def test_assertion_result_creation():
    """Test AssertionResult creation."""
    result = AssertionResult(
        success=True,
        error_code="TEST_ERROR",
        message="Test message",
        metadata={"key": "value"}
    )
    
    assert result.success
    assert result.error_code == "TEST_ERROR"
    assert result.message == "Test message"
    assert result.metadata == {"key": "value"}

def test_list_assertion_result_creation():
    """Test ListAssertionResult creation."""
    result = ListAssertionResult(
        success=True,
        error_code="TEST_ERROR",
        message="Test message",
        metadata={"key": "value"},
        matched_items=[{"text": "Item 1"}],
        unmatched_items=[{"text": "Item 2"}],
        total_items=2,
        matched_count=1
    )
    
    assert result.success
    assert result.error_code == "TEST_ERROR"
    assert result.message == "Test message"
    assert result.metadata == {"key": "value"}
    assert result.matched_items == [{"text": "Item 1"}]
    assert result.unmatched_items == [{"text": "Item 2"}]
    assert result.total_items == 2
    assert result.matched_count == 1

def test_base_assertions_initialization(mock_agent, mock_result):
    """Test BaseAssertions initialization."""
    assertions = BaseAssertions(mock_agent, mock_result)
    assert assertions.agent == mock_agent
    assert assertions.result == mock_result
    assert assertions.page is None

def test_create_metadata_with_none_values(base_assertions):
    """Test metadata creation with None values."""
    requirement = None
    expected_value = None
    content_info = None
    
    metadata = base_assertions._create_metadata(requirement, expected_value, content_info)
    
    assert metadata["requirement"] is None
    assert metadata["expected_value"] is None
    assert metadata["normalization"]["is_relative"] is False
    assert metadata["normalization"]["base_url"] is None
    assert metadata["normalization"]["normalized"] is False
    assert metadata["parsing"] is None

def test_parse_extracted_content_with_empty_dict(base_assertions):
    """Test parsing an empty dictionary."""
    content = {}
    result = base_assertions._parse_extracted_content(content, "text")
    
    assert result["parsing_status"] == "dict"
    assert result["parsing_error"] is None
    assert result["value"] == {}

def test_parse_extracted_content_with_nested_dict(base_assertions):
    """Test parsing a nested dictionary."""
    content = {
        "text": {
            "nested": "value"
        }
    }
    result = base_assertions._parse_extracted_content(content, "text")
    
    assert result["parsing_status"] == "dict"
    assert result["parsing_error"] is None
    assert result["value"] == content

def test_parse_extracted_content_with_list(base_assertions):
    """Test parsing a list."""
    content = ["item1", "item2"]
    result = base_assertions._parse_extracted_content(content, "text")
    
    assert result["parsing_status"] == "raw"
    assert result["parsing_error"] is None  # No error for non-string content
    assert result["value"] == {"text": str(content)}
    assert result["content_type"] == "list"

def test_process_extraction_result_with_none(base_assertions):
    """Test processing None result."""
    result = base_assertions._process_extraction_result(None)
    assert result.success is True
    assert result.extracted_content == "None"

def test_process_extraction_result_with_empty_dict(base_assertions):
    """Test processing empty dictionary."""
    content = {}
    result = base_assertions._process_extraction_result(content)
    assert result.success is True
    assert result.extracted_content == "{}"

def test_process_extraction_result_with_nested_dict(base_assertions):
    """Test processing nested dictionary."""
    content = {
        "element_text": {
            "nested": "value"
        }
    }
    result = base_assertions._process_extraction_result(content)
    assert result.success is True
    assert result.extracted_content == str(content["element_text"]) 