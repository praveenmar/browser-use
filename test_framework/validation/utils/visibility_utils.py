"""
Utility functions for checking element visibility.
"""

from typing import TYPE_CHECKING, Optional
from browser_use.browser.session import BrowserSession
from playwright.async_api import ElementHandle, Page

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession

async def is_element_visible_by_handle(element: ElementHandle, browser_session: BrowserSession) -> bool:
    """Check if an element is visible using browser-use's enhanced visibility checking.
    
    Args:
        element: The element handle to check
        browser_session: The browser session instance
        
    Returns:
        bool: True if element is visible, False otherwise
    """
    return await browser_session.is_visible_by_handle(element)

async def is_element_in_viewport(element: ElementHandle, page: Page) -> bool:
    """Check if an element is within the viewport bounds.
    
    Args:
        element: The element handle to check
        page: The page instance
        
    Returns:
        bool: True if element is in viewport, False otherwise
    """
    bbox = await element.bounding_box()
    if not bbox:
        return False
        
    viewport = await page.viewport_size()
    if not viewport:
        return False
        
    return (
        bbox['x'] >= 0 and
        bbox['y'] >= 0 and
        bbox['x'] + bbox['width'] <= viewport['width'] and
        bbox['y'] + bbox['height'] <= viewport['height']
    ) 