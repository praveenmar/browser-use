"""
Shared data structures for the validation framework.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlparse

@dataclass
class LinkExtractionResult:
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinkExtractionResult':
        """
        Create a LinkExtractionResult from a dictionary.
        
        Args:
            data: Dictionary containing link extraction data
            
        Returns:
            LinkExtractionResult instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
            
        # Extract and validate required fields
        success = bool(data.get('success', False))
        text = str(data.get('text', '')) if data.get('text') is not None else None
        href = str(data.get('href', '')) if data.get('href') is not None else None
        message = str(data.get('message', '')) if data.get('message') is not None else None
        original_requirement = str(data.get('original_requirement', ''))
        
        # Handle URL fields
        base_url = None
        is_relative = None
        if href:
            try:
                parsed = urlparse(href)
                is_relative = not (parsed.scheme and parsed.netloc)
                if is_relative and data.get('base_url'):
                    base_url = str(data.get('base_url'))
            except Exception:
                # If URL parsing fails, assume it's relative
                is_relative = True
                base_url = data.get('base_url')
        
        # Validate metadata and element_info
        metadata = data.get('metadata')
        if metadata is not None and not isinstance(metadata, dict):
            metadata = None
            
        element_info = data.get('element_info')
        if element_info is not None and not isinstance(element_info, dict):
            element_info = None
        
        return cls(
            success=success,
            text=text,
            href=href,
            is_relative=is_relative,
            base_url=base_url,
            message=message,
            original_requirement=original_requirement,
            metadata=metadata,
            element_info=element_info
        ) 