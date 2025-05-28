"""
Matching module for test assertions.
Handles text matching with strict exact matching by default.
Basic fuzzy matching is available but must be explicitly enabled.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import re
from difflib import SequenceMatcher

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

from .base import BaseAssertions, AssertionResult, ListAssertionResult

logger = logging.getLogger("test_framework.validation.assertions")

class MatchingAssertions(BaseAssertions):
    """Assertions for matching content against requirements."""
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize matching assertions.
        
        Args:
            agent: The agent instance
            result: The result history list
        """
        super().__init__(agent, result)
        self._similarity_threshold = 1.0  # Default to exact matching
        self._use_fuzzy_matching = False  # Default to strict matching
        
    def enable_fuzzy_matching(self, threshold: float = 0.8):
        """Enable fuzzy matching with specified threshold.
        
        Args:
            threshold: Similarity threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
            
        self._use_fuzzy_matching = True
        self._similarity_threshold = threshold
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by converting to lowercase and removing extra whitespace."""
        if not text:
            return ""
        # Replace hyphens with spaces before normalization
        text = text.replace('-', ' ')
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase and remove extra whitespace
        return ' '.join(text.lower().split())
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using string-based similarity."""
        # Handle empty strings first
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
            
        # If fuzzy matching is disabled, use exact matching
        if not self._use_fuzzy_matching:
            return 1.0 if text1 == text2 else 0.0
            
        # Normalize texts
        text1_normalized = self._normalize_text(text1)
        text2_normalized = self._normalize_text(text2)
        
        # If texts are identical after normalization, calculate case similarity
        if text1_normalized == text2_normalized:
            # If case is also identical, return 1.0
            if text1 == text2:
                return 1.0
            # Otherwise, return a high but not perfect similarity
            return 0.9
            
        # Calculate string-based similarity
        return SequenceMatcher(None, text1_normalized, text2_normalized).ratio()
        
    def _find_best_match(self, text: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        """Find the best matching text from a list of candidates.
        
        Args:
            text: Text to match
            candidates: List of candidate texts
            
        Returns:
            Tuple[Optional[str], float]: Best matching text and similarity score
        """
        if not candidates:
            return None, 0.0
            
        # If fuzzy matching is disabled, look for exact match
        if not self._use_fuzzy_matching:
            for candidate in candidates:
                if candidate == text:
                    return candidate, 1.0
            return None, 0.0
            
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self._calculate_text_similarity(text, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate
                
        return best_match, best_score
        
    def _verify_text_match(self, expected: str, actual: str, threshold: Optional[float] = None) -> AssertionResult:
        """Verify if two texts match.
        
        Args:
            expected: Expected text
            actual: Actual text
            threshold: Optional similarity threshold
            
        Returns:
            AssertionResult: Result of the verification
        """
        if threshold is None:
            threshold = self._similarity_threshold
            
        # Calculate similarity
        similarity = self._calculate_text_similarity(expected, actual)
        
        # Create metadata
        metadata = {
            "expected": expected,
            "actual": actual,
            "similarity": similarity,
            "threshold": threshold,
            "normalized_expected": self._normalize_text(expected),
            "normalized_actual": self._normalize_text(actual)
        }
        
        # Check if similarity meets threshold
        success = similarity >= threshold
        
        return AssertionResult(
            success=success,
            error_code=None if success else "TEXT_MISMATCH",
            message=f"Text similarity: {similarity:.2f} (threshold: {threshold})",
            metadata=metadata
        )
        
    def _verify_list_match(self, expected_items: List[str], actual_items: List[str], 
                          match_all: bool = True, threshold: Optional[float] = None) -> AssertionResult:
        """Verify if lists of items match.
        
        Args:
            expected_items: List of expected items
            actual_items: List of actual items
            match_all: Whether all items must match
            threshold: Optional similarity threshold
            
        Returns:
            AssertionResult: Result of the verification
        """
        if threshold is None:
            threshold = self._similarity_threshold
            
        matched_items = []
        unmatched_items = []
        
        for expected in expected_items:
            best_match, score = self._find_best_match(expected, actual_items)
            
            if score >= threshold:
                matched_items.append({
                    "expected": expected,
                    "actual": best_match,
                    "similarity": score
                })
            else:
                unmatched_items.append({
                    "expected": expected,
                    "best_match": best_match,
                    "similarity": score
                })
                
        # Determine success based on match_all flag
        if match_all:
            success = len(matched_items) == len(expected_items)
        else:
            success = len(matched_items) > 0
            
        return AssertionResult(
            success=success,
            error_code=None if success else "LIST_MISMATCH",
            message=f"Matched {len(matched_items)}/{len(expected_items)} items",
            metadata={
                "matched_items": matched_items,
                "unmatched_items": unmatched_items,
                "total_items": len(expected_items),
                "matched_count": len(matched_items),
                "threshold": threshold
            }
        ) 