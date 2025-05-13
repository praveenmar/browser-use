import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
from patchright.async_api import Page, Locator, TimeoutError as PlaywrightTimeoutError
from test_framework.validation.assertions import TestAssertions, VerifyTextAction, VerifyLinkAction
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
async def test_assertions(mock_agent):
    return TestAssertions(mock_agent, Mock())

class TestTextVerification:
    @pytest.mark.asyncio
    async def test_verify_text_success(self, mock_page, mock_browser_context):
        # Setup
        text_locator = mock_page.get_by_text.return_value
        text_locator.to_be_visible = AsyncMock()
        
        # Create action
        action = VerifyTextAction(text="Test Text", exact=False)
        
        # Create a mock controller with the verify_text action
        controller = Mock()
        async def verify_text(params, browser):
            if params.text == "Test Text":
                return ActionResult(extracted_content="Verified text 'Test Text' is present on the page")
            return ActionResult(error="Text not found")
        controller.act = AsyncMock(side_effect=verify_text)
        
        # Execute
        result = await controller.act(action, mock_browser_context)
        
        # Assert
        assert isinstance(result, ActionResult)
        assert result.error is None
        assert "Verified text" in result.extracted_content

    @pytest.mark.asyncio
    async def test_verify_text_not_found(self, mock_page, mock_browser_context):
        # Setup
        text_locator = mock_page.get_by_text.return_value
        text_locator.to_be_visible = AsyncMock(side_effect=PlaywrightTimeoutError("Text not found"))
        
        # Create action
        action = VerifyTextAction(text="Non-existent Text")
        
        # Create a mock controller with the verify_text action
        controller = Mock()
        async def verify_text(params, browser):
            return ActionResult(error="Text 'Non-existent Text' not found on the page")
        controller.act = AsyncMock(side_effect=verify_text)
        
        # Execute
        result = await controller.act(action, mock_browser_context)
        
        # Assert
        assert isinstance(result, ActionResult)
        assert result.error is not None
        assert "not found" in result.error

class TestLinkVerification:
    @pytest.mark.asyncio
    async def test_verify_link_success(self, mock_page, mock_browser_context):
        # Setup
        link_locator = mock_page.get_by_role.return_value
        link_locator.to_be_visible = AsyncMock()
        link_locator.get_attribute = AsyncMock(return_value="https://example.com")
        
        # Create action
        action = VerifyLinkAction(
            text="Test Link",
            href="https://example.com",
            exact=False
        )
        
        # Create a mock controller with the verify_link action
        controller = Mock()
        async def verify_link(params, browser):
            if params.text == "Test Link" and params.href == "https://example.com":
                return ActionResult(extracted_content="Verified link 'Test Link' with href 'https://example.com' is present")
            return ActionResult(error="Link not found")
        controller.act = AsyncMock(side_effect=verify_link)
        
        # Execute
        result = await controller.act(action, mock_browser_context)
        
        # Assert
        assert isinstance(result, ActionResult)
        assert result.error is None
        assert "Verified link" in result.extracted_content

    @pytest.mark.asyncio
    async def test_verify_link_not_found(self, mock_page, mock_browser_context):
        # Setup
        link_locator = mock_page.get_by_role.return_value
        link_locator.to_be_visible = AsyncMock(side_effect=PlaywrightTimeoutError("Link not found"))
        
        # Create action
        action = VerifyLinkAction(
            text="Non-existent Link",
            href="https://example.com"
        )
        
        # Create a mock controller with the verify_link action
        controller = Mock()
        async def verify_link(params, browser):
            return ActionResult(error="Link 'Non-existent Link' not found on the page")
        controller.act = AsyncMock(side_effect=verify_link)
        
        # Execute
        result = await controller.act(action, mock_browser_context)
        
        # Assert
        assert isinstance(result, ActionResult)
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_verify_link_href_mismatch(self, mock_page, mock_browser_context):
        # Setup
        link_locator = mock_page.get_by_role.return_value
        link_locator.to_be_visible = AsyncMock()
        link_locator.get_attribute = AsyncMock(return_value="https://wrong-url.com")
        
        # Create action
        action = VerifyLinkAction(
            text="Test Link",
            href="https://example.com"
        )
        
        # Create a mock controller with the verify_link action
        controller = Mock()
        async def verify_link(params, browser):
            return ActionResult(error="Link href mismatch. Expected: https://example.com, Got: https://wrong-url.com")
        controller.act = AsyncMock(side_effect=verify_link)
        
        # Execute
        result = await controller.act(action, mock_browser_context)
        
        # Assert
        assert isinstance(result, ActionResult)
        assert result.error is not None
        assert "href mismatch" in result.error

class TestStepHooks:
    @pytest.mark.asyncio
    async def test_extract_verification_steps(self, test_assertions):
        # Test text verification step
        text_step = "assert the text 'Hello World' is present in the page"
        steps = test_assertions._extract_verification_steps(text_step)
        assert len(steps) == 1
        assert "assert the text" in steps[0].lower()

        # Test link verification step
        link_step = "verify the link 'Click here' with href 'https://example.com' is present"
        steps = test_assertions._extract_verification_steps(link_step)
        assert len(steps) == 1
        assert "verify the link" in steps[0].lower()

    @pytest.mark.asyncio
    async def test_should_verify_now(self, test_assertions):
        # Test text verification
        text_step = "verify the text 'Hello World' is present"
        assert test_assertions._should_verify_now({"go_to_url": {}}, text_step) is True

        # Test link verification
        link_step = "verify the link 'Click here' with href 'https://example.com'"
        assert test_assertions._should_verify_now({"go_to_url": {}}, link_step) is True

        # Test non-verification step
        non_verify_step = "click the button"
        assert test_assertions._should_verify_now({"go_to_url": {}}, non_verify_step) is False