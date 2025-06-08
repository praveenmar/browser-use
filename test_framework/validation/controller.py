"""
Test framework controller that can be mounted into the agent.
This controller provides test-specific actions for extraction.
"""

import logging
import json
from typing import Dict, Any, Optional
from browser_use.browser.context import BrowserContext
from browser_use.browser.session import BrowserSession
from browser_use.controller.service import Controller
from browser_use.agent.views import ActionResult
from playwright.async_api import Locator, Page
from ..utils.visibility_utils import is_element_visible_by_handle

logger = logging.getLogger("test_framework.validation.controller")

class TestController(Controller):
    """Controller for test framework actions that can be mounted into the agent"""
    
    def __init__(self):
        super().__init__()
        self._register_extraction_actions()

    def _register_extraction_actions(self):
        """Register extraction actions that will be mounted into the agent"""
        
        @self.action(
            'Extract text content from an element using a selector',
            param_model=Dict[str, Any]
        )
        async def extract_text(params: Dict[str, Any], browser_session: BrowserSession) -> ActionResult:
            """Extract text content from an element using a selector"""
            try:
                page = await browser_session.get_current_page()
                locator = page.locator(params['selector'])
                
                is_visible = await is_element_visible_by_handle(locator, browser_session)
                if not is_visible:
                    return ActionResult(
                        success=False,
                        error=f"Element with selector '{params['selector']}' not found or not visible"
                    )

                text = await locator.text_content()
                content = {
                    "text": text,
                    "selector": params['selector'],
                    "metadata": {
                        "is_visible": is_visible,
                        "timeout": params.get('timeout', 5000)
                    }
                }
                return ActionResult(
                    success=True,
                    extracted_content=json.dumps(content)
                )
            except Exception as e:
                logger.error(f"Error extracting text: {str(e)}")
                return ActionResult(
                    success=False,
                    error=str(e)
                )

        @self.action(
            'Extract link information from an element using a selector',
            param_model=Dict[str, Any]
        )
        async def extract_link(params: Dict[str, Any], browser_session: BrowserSession) -> ActionResult:
            """Extract link information from an element using a selector"""
            try:
                page = await browser_session.get_current_page()
                locator = page.locator(params['selector'])
                
                is_visible = await is_element_visible_by_handle(locator, browser_session)
                if not is_visible:
                    return ActionResult(
                        success=False,
                        error=f"Element with selector '{params['selector']}' not found or not visible"
                    )

                href = await locator.get_attribute("href")
                text = await locator.text_content()
                current_url = page.url
                is_relative = href and not (href.startswith('http://') or href.startswith('https://'))

                result = {
                    "text": text,
                    "href": href,
                    "is_relative": is_relative,
                    "base_url": current_url
                }
                
                return ActionResult(
                    success=True,
                    extracted_content=json.dumps(result)
                )
            except Exception as e:
                logger.error(f"Error extracting link: {str(e)}")
                return ActionResult(
                    success=False,
                    error=str(e)
                )

        @self.action(
            'Extract attribute value from an element using a selector',
            param_model=Dict[str, Any]
        )
        async def extract_attribute(params: Dict[str, Any], browser_session: BrowserSession) -> ActionResult:
            """Extract attribute from an element using a selector"""
            try:
                page = await browser_session.get_current_page()
                locator = page.locator(params['selector'])
                
                is_visible = await is_element_visible_by_handle(locator, browser_session)
                if not is_visible:
                    return ActionResult(
                        success=False,
                        error=f"Element with selector '{params['selector']}' not found or not visible"
                    )

                value = await locator.get_attribute(params['attribute'])
                if value is None:
                    return ActionResult(
                        success=False,
                        error=f"Attribute '{params['attribute']}' not found on element"
                    )

                result = {
                    "value": value
                }
                
                return ActionResult(
                    success=True,
                    extracted_content=json.dumps(result)
                )
            except Exception as e:
                logger.error(f"Error extracting attribute: {str(e)}")
                return ActionResult(
                    success=False,
                    error=str(e)
                )

        @self.action(
            'Extract element from the page',
            param_model=Dict[str, Any]
        )
        async def extract_element(params: Dict[str, Any], browser_session: BrowserSession) -> ActionResult:
            """Extract element from the page"""
            try:
                page = await browser_session.get_current_page()
                locator = page.locator(params['selector'])
                
                is_visible = await is_element_visible_by_handle(locator, browser_session)
                if not is_visible:
                    return ActionResult(
                        success=False,
                        error=f"Element with selector '{params['selector']}' not found or not visible"
                    )

                element = await locator.element_handle()
                if element:
                    return ActionResult(
                        success=True,
                        extracted_content=json.dumps({
                            "success": True,
                            "element": element,
                            "is_visible": is_visible,
                        })
                    )
                return ActionResult(
                    success=False,
                    error="Element not found"
                )
            except Exception as e:
                logger.error(f"Error extracting element: {str(e)}")
                return ActionResult(
                    success=False,
                    error=str(e)
                ) 