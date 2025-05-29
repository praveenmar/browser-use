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
from browser_use.browser.session import BrowserSession

@pytest_asyncio.fixture
async def mock_page():
    page = Mock(spec=Page)
    page.get_by_text = Mock(return_value=Mock(spec=Locator))
    page.get_by_role = Mock(return_value=Mock(spec=Locator))
    return page

@pytest_asyncio.fixture
async def mock_browser_session(mock_page):
    session = Mock(spec=BrowserSession)
    session.get_current_page = AsyncMock(return_value=mock_page)
    return session

@pytest_asyncio.fixture
async def mock_agent(mock_browser_session):
    agent = Mock()
    agent.browser_session = mock_browser_session
    agent.state = Mock()
    agent.state.n_steps = 1
    return agent

@pytest_asyncio.fixture
async def verification_assertions(mock_agent):
    return VerificationAssertions(mock_agent, [])

@pytest_asyncio.fixture
async def extraction_assertions(mock_agent):
    return ExtractionAssertions(mock_agent, [])

@pytest_asyncio.fixture
async def matching_assertions(mock_agent):
    return MatchingAssertions(mock_agent, [])

class TestVerificationAssertions:
    """Test verification assertions"""
    
    @pytest.mark.asyncio
    async def test_verify_text_success(self, mock_page, mock_browser_session, verification_assertions):
        """Test successful text verification"""
        # Setup
        mock_page.get_by_text.return_value.is_visible.return_value = True
        mock_page.get_by_text.return_value.text_content.return_value = "Test Text"
        
        # Execute
        result = await verification_assertions.verify_text("Test Text")
        
        # Verify
        assert result.success
        assert result.message == "Text 'Test Text' found and visible"
        
    @pytest.mark.asyncio
    async def test_verify_text_not_found(self, mock_page, mock_browser_session, verification_assertions):
        """Test text verification when text is not found"""
        # Setup
        mock_page.get_by_text.return_value.is_visible.return_value = False
        
        # Execute
        result = await verification_assertions.verify_text("Nonexistent Text")
        
        # Verify
        assert not result.success
        assert "Text 'Nonexistent Text' not found or not visible" in result.message
        
    @pytest.mark.asyncio
    async def test_verify_link_success(self, mock_page, mock_browser_session, verification_assertions):
        """Test successful link verification"""
        # Setup
        mock_page.get_by_role.return_value.is_visible.return_value = True
        mock_page.get_by_role.return_value.get_attribute.return_value = "https://example.com"
        mock_page.get_by_role.return_value.text_content.return_value = "Example Link"
        
        # Execute
        result = await verification_assertions.verify_link("Example Link", "https://example.com")
        
        # Verify
        assert result.success
        assert result.message == "Link 'Example Link' found with matching href"
        
    @pytest.mark.asyncio
    async def test_verify_link_not_found(self, mock_page, mock_browser_session, verification_assertions):
        """Test link verification when link is not found"""
        # Setup
        mock_page.get_by_role.return_value.is_visible.return_value = False
        
        # Execute
        result = await verification_assertions.verify_link("Nonexistent Link", "https://example.com")
        
        # Verify
        assert not result.success
        assert "Link 'Nonexistent Link' not found or not visible" in result.message
        
    @pytest.mark.asyncio
    async def test_verify_link_href_mismatch(self, mock_page, mock_browser_session, verification_assertions):
        """Test link verification when href doesn't match"""
        # Setup
        mock_page.get_by_role.return_value.is_visible.return_value = True
        mock_page.get_by_role.return_value.get_attribute.return_value = "https://wrong.com"
        mock_page.get_by_role.return_value.text_content.return_value = "Example Link"
        
        # Execute
        result = await verification_assertions.verify_link("Example Link", "https://example.com")
        
        # Verify
        assert not result.success
        assert "Link href mismatch" in result.message

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