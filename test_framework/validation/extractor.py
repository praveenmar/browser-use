"""
Extractor layer for handling all browser interactions.
This layer is responsible for extracting data from the browser without any validation logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from browser_use import Page
from browser_use.dom.service import DomService
from browser_use.browser.context import BrowserContext
from browser_use.browser.session import BrowserSession
import json

logger = logging.getLogger("test_framework.validation.extractor")

@dataclass
class ExtractedTextResult:
    """Result of text extraction from browser"""
    success: bool
    text: Optional[str]
    message: Optional[str]
    original_requirement: str
    metadata: Optional[Dict[str, Any]] = None
    element_info: Optional[Dict[str, Any]] = None

@dataclass
class ExtractedLinkResult:
    """Result of link extraction from browser"""
    success: bool
    text: Optional[str]
    href: Optional[str]
    is_relative: Optional[bool]
    base_url: Optional[str]
    message: Optional[str]
    original_requirement: str
    metadata: Optional[Dict[str, Any]] = None
    element_info: Optional[Dict[str, Any]] = None

@dataclass
class ExtractedAttributeResult:
    """Result of attribute extraction from browser"""
    success: bool
    value: Optional[str]
    message: Optional[str]
    original_requirement: str
    metadata: Optional[Dict[str, Any]] = None
    element_info: Optional[Dict[str, Any]] = None

class BrowserExtractor:
    """Handles all browser interactions for data extraction without validation"""
    
    def __init__(self, browser_session: BrowserSession, agent):
        self.browser_session = browser_session
        self.agent = agent
        self.page: Optional[Page] = None

    async def ensure_page(self) -> Page:
        """Ensure we have a valid page object"""
        if not self.page:
            self.page = await self.browser_session.get_current_page()
        return self.page

    def _parse_content(self, content: Any) -> Dict[str, Any]:
        """Parse content that could be a string or dictionary"""
        if content is None:
            return {}
            
        if isinstance(content, dict):
            return content
            
        if not isinstance(content, str):
            return {"text": str(content)}
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}

    async def extract_text(self, selector: str, timeout: int = 5000, original_requirement: str = "") -> ExtractedTextResult:
        """Extract text from page without validation"""
        try:
            action_result = await self.agent.act({
                'extract_text': {
                    'selector': selector,
                    'timeout': timeout,
                    'original_requirement': original_requirement
                }
            })
            
            if not action_result.success:
                return ExtractedTextResult(
                    success=False,
                    text=None,
                    message=action_result.error,
                    original_requirement=original_requirement
                )

            data = self._parse_content(action_result.extracted_content)
            if data:
                return ExtractedTextResult(
                    success=True,
                    text=data.get('text'),
                    message=None,
                    original_requirement=original_requirement,
                    metadata=data.get('metadata'),
                    element_info=data.get('element_info')
                )
            else:
                return ExtractedTextResult(
                    success=True,
                    text=action_result.extracted_content,
                    message=None,
                    original_requirement=original_requirement
                )

        except Exception as e:
            logger.error(f"Error extracting text for selector '{selector}': {str(e)}")
            return ExtractedTextResult(
                success=False,
                text=None,
                message=f"Error extracting text for selector '{selector}': {str(e)}",
                original_requirement=original_requirement
            )

    async def extract_link(self, selector: str, timeout: int = 5000, original_requirement: str = "") -> ExtractedLinkResult:
        """Extract link from page without validation"""
        try:
            action_result = await self.agent.act({
                'extract_link': {
                    'selector': selector,
                    'timeout': timeout,
                    'original_requirement': original_requirement
                }
            })
            
            if not action_result.success:
                return ExtractedLinkResult(
                    success=False,
                    text=None,
                    href=None,
                    is_relative=None,
                    base_url=None,
                    message=action_result.error,
                    original_requirement=original_requirement
                )

            data = self._parse_content(action_result.extracted_content)
            if data:
                return ExtractedLinkResult(
                    success=True,
                    text=data.get('text'),
                    href=data.get('href'),
                    is_relative=data.get('is_relative'),
                    base_url=data.get('base_url'),
                    message=None,
                    original_requirement=original_requirement,
                    metadata=data.get('metadata'),
                    element_info=data.get('element_info')
                )
            else:
                return ExtractedLinkResult(
                    success=True,
                    text=action_result.extracted_content,
                    href=None,
                    is_relative=None,
                    base_url=None,
                    message=None,
                    original_requirement=original_requirement
                )

        except Exception as e:
            logger.error(f"Error extracting link for selector '{selector}': {str(e)}")
            return ExtractedLinkResult(
                success=False,
                text=None,
                href=None,
                is_relative=None,
                base_url=None,
                message=f"Error extracting link for selector '{selector}': {str(e)}",
                original_requirement=original_requirement
            )

    async def extract_attribute(self, selector: str, attribute: str, timeout: int = 5000, original_requirement: str = "") -> ExtractedAttributeResult:
        """Extract attribute from element without validation"""
        try:
            action_result = await self.agent.act({
                'extract_attribute': {
                    'selector': selector,
                    'attribute': attribute,
                    'timeout': timeout,
                    'original_requirement': original_requirement
                }
            })
            
            if not action_result.success:
                return ExtractedAttributeResult(
                    success=False,
                    value=None,
                    message=action_result.error,
                    original_requirement=original_requirement
                )

            data = self._parse_content(action_result.extracted_content)
            if data:
                return ExtractedAttributeResult(
                    success=True,
                    value=data.get('value'),
                    message=None,
                    original_requirement=original_requirement,
                    metadata=data.get('metadata'),
                    element_info=data.get('element_info')
                )
            else:
                return ExtractedAttributeResult(
                    success=True,
                    value=action_result.extracted_content,
                    message=None,
                    original_requirement=original_requirement
                )

        except Exception as e:
            logger.error(f"Error extracting attribute '{attribute}' for selector '{selector}': {str(e)}")
            return ExtractedAttributeResult(
                success=False,
                value=None,
                message=f"Error extracting attribute '{attribute}' for selector '{selector}': {str(e)}",
                original_requirement=original_requirement
            )

    async def _get_element_metadata(self, locator: Locator) -> Dict[str, Any]:
        """Get element metadata without validation"""
        try:
            element_info = {
                "tag_name": await locator.evaluate("el => el.tagName.toLowerCase()"),
                "class_list": await locator.evaluate("el => Array.from(el.classList)"),
                "id": await locator.get_attribute("id"),
                "is_visible": True,
                "is_enabled": await locator.is_enabled(),
                "is_editable": await locator.is_editable(),
                "bounding_box": await locator.bounding_box()
            }
            
            styles = await locator.evaluate("""
                el => {
                    const styles = window.getComputedStyle(el);
                    return {
                        color: styles.color,
                        background: styles.backgroundColor,
                        font_size: styles.fontSize,
                        font_family: styles.fontFamily,
                        display: styles.display,
                        position: styles.position,
                        z_index: styles.zIndex
                    }
                }
            """)
            
            return {
                "element_info": element_info,
                "styles": styles
            }
        except Exception as e:
            logger.warning(f"Could not collect full metadata: {str(e)}")
            return {} 