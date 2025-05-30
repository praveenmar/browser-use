"""
Output processor module for cleaning up test output.
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("test_framework.output_processor")

class OutputProcessor:
    """Processes and cleans up test output."""
    
    # Common patterns for cleaning up web content
    CLEANUP_PATTERNS = [
        # HTML and formatting
        (r'<[^>]+>', ''),  # HTML tags
        (r'^>\s*', ''),  # Markdown blockquotes
        (r'\*\*(.*?)\*\*', r'\1'),  # Bold markdown
        (r'\*(.*?)\*', r'\1'),  # Italic markdown
        (r'\[(.*?)\]\(.*?\)', r'\1'),  # Links
        
        # Navigation and UI elements
        (r'Skip to.*?Main content', ''),  # Skip to main content links
        (r'Back to top', ''),  # Back to top buttons
        (r'Menu|Navigation|Home|Search|Sign in|Log in|Register|Sign up', ''),  # Common navigation elements
        (r'Cookie|Privacy|Terms|Conditions|Policy', ''),  # Common footer elements
        
        # Media and embeds
        (r'!\[.*?\]\(data:image.*?\)', ''),  # Base64 images
        (r'IFRAME.*?$', ''),  # IFRAME content
        (r'<img.*?>', ''),  # Image tags
        (r'<video.*?>.*?</video>', ''),  # Video tags
        (r'<audio.*?>.*?</audio>', ''),  # Audio tags
        
        # Error messages and notifications
        (r'Error|Warning|Notice|Alert|Notification', ''),  # Common error/notification text
        (r'This site can\'t be reached.*?Details', ''),  # Network error messages
        
        # Whitespace and formatting
        (r'\n\s*\n', ' '),  # Multiple empty lines to single space
        (r'\s+', ' '),  # Multiple spaces to single space
        (r'[^\w\s\-.,;:!?]', ''),  # Special characters
    ]
    
    @staticmethod
    def clean_output(content: str) -> str:
        """Clean up the output content by removing unnecessary elements.
        
        Args:
            content: The raw output content
            
        Returns:
            str: Cleaned output content
        """
        if not content:
            return ""
            
        # Apply all cleanup patterns
        for pattern, replacement in OutputProcessor.CLEANUP_PATTERNS:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
        # Final cleanup
        content = content.strip()  # Remove leading/trailing whitespace
        content = re.sub(r'\s+', ' ', content)  # Normalize all whitespace to single spaces
        
        return content
        
    @staticmethod
    def process_extraction_result(result: Dict[str, Any]) -> Optional[str]:
        """Process an extraction result and return cleaned content.
        
        Args:
            result: The extraction result dictionary
            
        Returns:
            Optional[str]: Cleaned content or None if no content found
        """
        if not result:
            return None
            
        # Extract text content
        content = None
        if isinstance(result, dict):
            for key in ["text", "extracted_text", "quote", "content", "page_content", "exact_text"]:
                if key in result:
                    content = result[key]
                    break
        elif isinstance(result, str):
            content = result
            
        if not content:
            return None
            
        # Clean the content
        return OutputProcessor.clean_output(str(content))

    @staticmethod
    def add_website_patterns(website: str, patterns: List[tuple]) -> None:
        """Add new website-specific patterns.
        
        Args:
            website: The website name
            patterns: List of (pattern, replacement) tuples
        """
        # This method is no longer used in the new implementation
        pass 