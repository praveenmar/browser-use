import pytest
from typing import Optional
from test_framework.validation.assertions import (
    TestAssertions,
    AssertionResult,
    ListAssertionResult,
)
from browser_use.agent.views import ActionResult
from browser_use.browser.session import BrowserSession

@pytest.fixture
def browser_session(mock_browser_session):
    """Create a mock browser session."""
    return mock_browser_session

@pytest.mark.asyncio
async def test_element_exists_assertion(browser_session):
    """Test element existence assertion."""
    # Setup mock element
    element = mocker.AsyncMock()
    element.is_visible.return_value = True
    browser_session.get_current_page.return_value.wait_for_selector.return_value = element
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        ActionResult(success=True, extracted_content="Element found"),
        "Element found"
    )
    
    assert result.success
    assert "Element found" in result.message

@pytest.mark.asyncio
async def test_element_not_exists_assertion(browser_session):
    """Test element non-existence assertion."""
    # Setup mock to return None (element not found)
    browser_session.get_current_page.return_value.wait_for_selector.return_value = None
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        ActionResult(success=False, error="Element not found"),
        "Element found"
    )
    
    assert not result.success
    assert "Element not found" in result.message

@pytest.mark.asyncio
async def test_element_visibility_assertion(browser_session):
    """Test element visibility assertion."""
    # Setup mock element
    element = mocker.AsyncMock()
    element.is_visible.return_value = True
    browser_session.get_current_page.return_value.wait_for_selector.return_value = element
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        ActionResult(success=True, extracted_content="Element is visible"),
        "Element is visible"
    )
    
    assert result.success
    assert "Element is visible" in result.message

@pytest.mark.asyncio
async def test_page_title_assertion(browser_session):
    """Test page title assertion."""
    # Setup mock page
    browser_session.get_current_page.return_value.title.return_value = "Test Page"
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        ActionResult(success=True, extracted_content="Test Page"),
        "Test Page"
    )
    
    assert result.success
    assert "Test Page" in result.message

@pytest.mark.asyncio
async def test_page_url_assertion(browser_session):
    """Test page URL assertion."""
    # Setup mock page
    browser_session.get_current_page.return_value.url = "https://example.com/test"
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        ActionResult(success=True, extracted_content="https://example.com/test"),
        "https://example.com/test"
    )
    
    assert result.success
    assert "https://example.com/test" in result.message

@pytest.mark.asyncio
async def test_action_success_assertion():
    """Test action success assertion."""
    # Create mock action result
    action_result = ActionResult(
        success=True,
        extracted_content="Test content"
    )
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        action_result,
        "Test content"
    )
    
    assert result.success
    assert "Test content" in result.message

@pytest.mark.asyncio
async def test_action_failure_assertion():
    """Test action failure assertion."""
    # Create mock action result
    action_result = ActionResult(
        success=False,
        error="Test error"
    )
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        action_result,
        "Test content"
    )
    
    assert not result.success
    assert "Test error" in result.message

@pytest.mark.asyncio
async def test_assertion_chain(browser_session):
    """Test chaining multiple assertions."""
    # Setup mock page
    page = browser_session.get_current_page.return_value
    page.title.return_value = "Test Page"
    page.url = "https://example.com/test"
    
    # Create element mock
    element = mocker.AsyncMock()
    element.is_visible.return_value = True
    page.wait_for_selector.return_value = element
    
    # Create assertions
    assertions = TestAssertions(None, None)
    results = []
    
    # Run assertions
    results.append(await assertions.verify_text(
        ActionResult(success=True, extracted_content="Test Page"),
        "Test Page"
    ))
    results.append(await assertions.verify_text(
        ActionResult(success=True, extracted_content="https://example.com/test"),
        "https://example.com/test"
    ))
    results.append(await assertions.verify_text(
        ActionResult(success=True, extracted_content="Element found"),
        "Element found"
    ))
    
    assert len(results) == 3
    assert all(result.success for result in results)

@pytest.mark.asyncio
async def test_assertion_early_stop(browser_session):
    """Test assertion chain stops on first failure."""
    # Setup mock page
    page = browser_session.get_current_page.return_value
    page.title.return_value = "Wrong Title"
    page.url = "https://example.com/test"
    
    # Create assertions
    assertions = TestAssertions(None, None)
    results = []
    
    # Run assertions
    results.append(await assertions.verify_text(
        ActionResult(success=True, extracted_content="Wrong Title"),
        "Test Page"
    ))
    
    assert len(results) == 1
    assert not results[0].success
    assert "Wrong Title" in results[0].message

@pytest.mark.asyncio
async def test_assertion_error_handling(browser_session):
    """Test error handling in assertions."""
    # Setup mock to raise exception
    browser_session.get_current_page.return_value.wait_for_selector.side_effect = Exception("Test error")
    
    # Create assertion
    assertions = TestAssertions(None, None)
    result = await assertions.verify_text(
        ActionResult(success=False, error="Test error"),
        "Test content"
    )
    
    assert not result.success
    assert "Test error" in result.message 