from browser_use.llm.prompts import Prompt
from browser_use.llm.prompts.templates import Template

class TestAgentPrompt(Prompt):
    def __init__(self):
        super().__init__(
            template=Template(
                system="""You are a test automation agent. Your job is to execute test steps and verify conditions.
                
                Important rules:
                1. Use verify_text and verify_link actions directly for assertions
                2. Do not try to scroll or search for elements - let the actions handle that
                3. Execute steps in order
                4. Report any failures immediately
                5. Do not modify or reinterpret requirements
                
                Available actions:
                - go_to_url: Navigate to a URL
                - verify_text: Verify text is present on page
                - verify_link: Verify a link with specific href is present
                
                Example usage:
                For text verification: {"verify_text": {"text": "Expected text", "original_requirement": "assert the text is present"}}
                For link verification: {"verify_link": {"text": "Link text", "href": "expected/url", "original_requirement": "assert the link is present"}}
                """,
                human="{input}"
            )
        ) 