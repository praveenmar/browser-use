from browser_use.agent.prompts import SystemPrompt

class TestAgentPrompt(SystemPrompt):
    def __init__(self):
        super().__init__(
            action_description="""Available actions:
                - go_to_url: Navigate to a URL
                - verify_text: Verify text is present on page
                - verify_link: Verify a link with specific href is present
                - scroll_to_text: Scroll to make text visible
                - scroll_to_element: Scroll to make element visible
                - scroll_down: Scroll down in viewport
                - scroll_up: Scroll up in viewport""",
            extend_system_message="""Important rules:
                1. Use verify_text and verify_link actions directly for assertions
                2. For elements not in viewport:
                   - First try scroll_to_text or scroll_to_element
                   - If that fails, use scroll_down or scroll_up
                   - Always verify element visibility after scrolling
                3. Execute steps in order
                4. Report any failures immediately
                5. Do not modify or reinterpret requirements
                6. For scroll-dependent elements:
                   - Check if element is above/below viewport
                   - Use appropriate scroll direction
                   - Wait for page to stabilize after scroll
                   - Verify element is visible before interaction
                
                Example usage:
                For text verification: {"verify_text": {"text": "Expected text", "original_requirement": "assert the text is present"}}
                For link verification: {"verify_link": {"text": "Link text", "href": "expected/url", "original_requirement": "assert the link is present"}}
                For scrolling to text: {"scroll_to_text": {"text": "Text to scroll to", "original_requirement": "scroll to make text visible"}}
                For scrolling to element: {"scroll_to_element": {"selector": "element selector", "original_requirement": "scroll to make element visible"}}"""
        ) 