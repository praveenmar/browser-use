async def find_element_by_text(context, text: str):
    """Try to find a visible element by its text content."""
    try:
        element = await context.get_locate_element_by_text(text)
        if element and not await element.is_hidden():
            return element
    except Exception:
        pass
    return None

async def find_element_by_selector(context, selector: str):
    """Try to find a visible element by CSS selector."""
    try:
        element = await context.get_locate_element_by_css_selector(selector)
        if element and not await element.is_hidden():
            return element
    except Exception:
        pass
    return None

async def find_element_by_xpath(context, xpath: str):
    """Try to find a visible element by XPath."""
    try:
        element = await context.get_locate_element_by_xpath(xpath)
        if element and not await element.is_hidden():
            return element
    except Exception:
        pass
    return None