"""
Models module for test assertions.
Contains data models and types used across assertion modules.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from enum import Enum

class AssertionType(Enum):
    """Types of assertions supported"""
    TEXT = "text"
    LINK = "link"
    ATTRIBUTE = "attribute"
    LIST = "list"
    MULTI_VALUE = "multi_value"

@dataclass
class AssertionResult:
    """Result of an assertion check"""
    success: bool
    error_code: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ListAssertionResult(AssertionResult):
    """Result of a list-based assertion check"""
    matched_items: List[Dict[str, Any]] = None
    unmatched_items: List[Dict[str, Any]] = None
    total_items: int = 0
    matched_count: int = 0

@dataclass
class ExtractionResult:
    """Result of a text extraction"""
    text: str
    confidence: float
    source: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StepContext:
    """Context information for a step"""
    step_number: int
    requirements: List[str]
    extractions: Dict[str, Any]
    assertions: List[AssertionResult]

@dataclass
class VerificationContext:
    """Context for verification process"""
    current_step: int
    step_contexts: Dict[int, StepContext]
    lookahead_window: int
    similarity_threshold: float

@dataclass
class MatchingResult:
    """Result of a text matching operation"""
    is_match: bool
    similarity: float
    matched_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ListMatchingResult:
    """Result of a list matching operation"""
    matched_items: List[Dict[str, Any]]
    unmatched_items: List[Dict[str, Any]]
    total_items: int
    matched_count: int
    match_all: bool
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExtractionMetadata:
    """Metadata for an extraction operation"""
    source: str
    timestamp: str
    confidence: float
    context: Optional[Dict[str, Any]] = None

@dataclass
class VerificationMetadata:
    """Metadata for a verification operation"""
    requirement: str
    expected_value: Any
    actual_value: Any
    similarity: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class StepMetadata:
    """Metadata for a step operation"""
    step_number: int
    requirements: List[str]
    extractions: List[ExtractionResult]
    assertions: List[AssertionResult]
    context: Optional[Dict[str, Any]] = None 