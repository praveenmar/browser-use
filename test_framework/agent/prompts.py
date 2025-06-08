from browser_use.agent.prompts import SystemPrompt

class TestAgentPrompt(SystemPrompt):
    def __init__(self):
        super().__init__(
            action_description="""Available actions:
                - go_to_url: Navigate to a URL
                - verify_text: Verify text is present on page
                - verify_link: Verify a link with specific href is present""",
            extend_system_message="""Important rules:
                1. Use verify_text and verify_link actions directly for assertions
                2. Execute steps in order exactly as written
                3. Report any failures immediately
                4. Do not modify or reinterpret requirements
                5. Do not add extra steps or actions not specified in the test
                6. Do not try to find alternative paths or solutions
                7. If a step fails, report the failure and stop - do not try to work around it
                
                Example usage:
                For text verification: {"verify_text": {"text": "Expected text", "original_requirement": "assert the text is present"}}
                For link verification: {"verify_link": {"text": "Link text", "href": "expected/url", "original_requirement": "assert the link is present"}}"""
        ) 