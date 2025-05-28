import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
from patchright.async_api import Page, Locator, TimeoutError as PlaywrightTimeoutError
from test_framework.validation.assertions import (
    VerificationAssertions,
    ExtractionAssertions,
    MatchingAssertions,
    AssertionResult
)
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

@pytest_asyncio.fixture
async def mock_page():
    page = Mock(spec=Page)
    page.get_by_text = Mock(return_value=Mock(spec=Locator))
    page.get_by_role = Mock(return_value=Mock(spec=Locator))
    return page

@pytest_asyncio.fixture
async def mock_browser_context(mock_page):
    context = Mock(spec=BrowserContext)
    context.get_current_page = AsyncMock(return_value=mock_page)
    return context

@pytest_asyncio.fixture
async def mock_agent(mock_browser_context):
    agent = Mock()
    agent.browser_context = mock_browser_context
    agent.state = Mock()
    agent.state.n_steps = 1
    return agent

@pytest_asyncio.fixture
async def verification_assertions(mock_agent):
    return VerificationAssertions(mock_agent, Mock())

class TestTextVerification:
    @pytest.mark.asyncio
    async def test_verify_text_success(self, mock_page, mock_browser_context, verification_assertions):
        # Setup
        text_locator = mock_page.get_by_text.return_value
        text_locator.to_be_visible = AsyncMock()
        
        # Create verification
        result = await verification_assertions._verify_condition("verify the text 'Test Text' is present", 0)
        
        # Assert
        assert isinstance(result, AssertionResult)
        assert result.success is True
        assert result.error_code is None
        assert "Test Text" in result.message

    @pytest.mark.asyncio
    async def test_verify_text_not_found(self, mock_page, mock_browser_context, verification_assertions):
        # Setup
        text_locator = mock_page.get_by_text.return_value
        text_locator.to_be_visible = AsyncMock(side_effect=PlaywrightTimeoutError("Text not found"))
        
        # Create verification
        result = await verification_assertions._verify_condition("verify the text 'Non-existent Text' is present", 0)
        
        # Assert
        assert isinstance(result, AssertionResult)
        assert result.success is False
        assert result.error_code == "TEXT_MISMATCH"
        assert "not found" in result.message

class TestLinkVerification:
    @pytest.mark.asyncio
    async def test_verify_link_success(self, mock_page, mock_browser_context, verification_assertions):
        # Setup
        link_locator = mock_page.get_by_role.return_value
        link_locator.to_be_visible = AsyncMock()
        link_locator.get_attribute = AsyncMock(return_value="https://example.com")
        
        # Create verification
        result = await verification_assertions._verify_condition(
            "verify the link 'Test Link' with href 'https://example.com' is present",
            0
        )
        
        # Assert
        assert isinstance(result, AssertionResult)
        assert result.success is True
        assert result.error_code is None
        assert "Test Link" in result.message

    @pytest.mark.asyncio
    async def test_verify_link_not_found(self, mock_page, mock_browser_context, verification_assertions):
        # Setup
        link_locator = mock_page.get_by_role.return_value
        link_locator.to_be_visible = AsyncMock(side_effect=PlaywrightTimeoutError("Link not found"))
        
        # Create verification
        result = await verification_assertions._verify_condition(
            "verify the link 'Non-existent Link' with href 'https://example.com' is present",
            0
        )
        
        # Assert
        assert isinstance(result, AssertionResult)
        assert result.success is False
        assert result.error_code == "LINK_NOT_FOUND"
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_verify_link_href_mismatch(self, mock_page, mock_browser_context, verification_assertions):
        # Setup
        link_locator = mock_page.get_by_role.return_value
        link_locator.to_be_visible = AsyncMock()
        link_locator.get_attribute = AsyncMock(return_value="https://wrong-url.com")
        
        # Create verification
        result = await verification_assertions._verify_condition(
            "verify the link 'Test Link' with href 'https://example.com' is present",
            0
        )
        
        # Assert
        assert isinstance(result, AssertionResult)
        assert result.success is False
        assert result.error_code == "LINK_HREF_MISMATCH"
        assert "href mismatch" in result.message

class TestStepVerification:
    @pytest.mark.asyncio
    async def test_extract_verification_steps(self, verification_assertions):
        # Test text verification step
        text_step = "assert the text 'Hello World' is present in the page"
        steps = verification_assertions._extract_verification_steps(text_step)
        assert len(steps) == 1
        assert "assert the text" in steps[0].lower()

        # Test link verification step
        link_step = "verify the link 'Click here' with href 'https://example.com' is present"
        steps = verification_assertions._extract_verification_steps(link_step)
        assert len(steps) == 1
        assert "verify the link" in steps[0].lower()

    @pytest.mark.asyncio
    async def test_should_verify_now(self, verification_assertions):
        # Test text verification
        text_step = "verify the text 'Hello World' is present"
        assert verification_assertions._should_verify_now({"go_to_url": {}}, text_step) is True

        # Test link verification
        link_step = "verify the link 'Click here' with href 'https://example.com'"
        assert verification_assertions._should_verify_now({"go_to_url": {}}, link_step) is True

        # Test non-verification step
        non_verify_step = "click the button"
        assert verification_assertions._should_verify_now({"go_to_url": {}}, non_verify_step) is False