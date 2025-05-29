import asyncio
import pytest
from browser_use.browser.session import BrowserSession
from browser_use.browser.profile import BrowserProfile

@pytest.fixture(scope='session')
def event_loop():
    """Create and provide an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope='session')
async def browser_session(event_loop):
    """Create and provide a BrowserSession instance with security disabled."""
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            disable_security=True,
        )
    )
    await browser_session.start()
    yield browser_session
    await browser_session.stop()

@pytest.fixture
def mock_browser_session(mocker):
    """Create a mock browser session for testing."""
    session = mocker.Mock(spec=BrowserSession)
    page = mocker.AsyncMock()
    session.get_current_page.return_value = page
    session.agent_current_page = page
    session.human_current_page = page
    return session 