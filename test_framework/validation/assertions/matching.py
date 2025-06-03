"""
Matching module for test assertions.
Handles text matching with strict exact matching by default.
Basic fuzzy matching is available but must be explicitly enabled.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union, TypedDict
from enum import Enum, auto
import re
from difflib import SequenceMatcher

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

from .base import BaseAssertions, AssertionResult, ListAssertionResult

logger = logging.getLogger("test_framework.validation.assertions")

# Constants for similarity thresholds
FUZZY_MATCH_THRESHOLD = 0.85
CASE_INSENSITIVE_THRESHOLD = 1.0
EXACT_MATCH_THRESHOLD = 1.0

class MatchType(Enum):
    """Types of text matches"""
    EXACT = auto()
    CONTAINS = auto()
    CASE_INSENSITIVE = auto()
    FUZZY = auto()
    NO_MATCH = auto()
    
    def __str__(self) -> str:
        return self.name.lower()

class MatchResult(TypedDict):
    """Result of a text matching operation"""
    match_type: MatchType
    success: bool
    similarity_score: float
    matched_snippet: str

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
        """Normalize text for comparison."""
        if not text:
            return ""
        return text.lower().strip()
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, text1, text2).ratio()
        
    def match_text(self, expected: str, actual: str, use_fuzzy_matching: bool = True) -> MatchResult:
        """Match text using a multi-tiered matching strategy.
        
        Args:
            expected: Expected text to match
            actual: Actual text to match against
            use_fuzzy_matching: Whether to use fuzzy matching
            
        Returns:
            MatchResult: Result of the matching operation with type, success, score and snippet
        """
        logger.debug(f"Starting text matching: expected='{expected}', actual='{actual}'")
        
        # Initialize result
        result: MatchResult = {
            "match_type": MatchType.NO_MATCH,
            "success": False,
            "similarity_score": 0.0,
            "matched_snippet": actual
        }
        
        # 1. Try exact match
        logger.debug(f"Attempting exact match: '{expected}' == '{actual}'")
        if expected == actual:
            logger.info(f"Exact match found: '{expected}'")
            result.update({
                "match_type": MatchType.EXACT,
                "success": True,
                "similarity_score": EXACT_MATCH_THRESHOLD
            })
            return result
            
        # 2. Try contains match
        logger.debug(f"Attempting contains match: '{expected}' in '{actual}'")
        if expected in actual:
            logger.info(f"Contains match found: '{expected}' in '{actual}'")
            result.update({
                "match_type": MatchType.CONTAINS,
                "success": True,
                "similarity_score": 1.0
            })
            return result
            
        # 3. Try case-insensitive match
        logger.debug(f"Attempting case-insensitive match: '{expected.lower()}' == '{actual.lower()}'")
        if expected.lower() == actual.lower():
            logger.warning(f"Case-insensitive match found but exact match failed. Potential case mismatch: '{expected}' vs '{actual}'")
            result.update({
                "match_type": MatchType.CASE_INSENSITIVE,
                "success": True,
                "similarity_score": CASE_INSENSITIVE_THRESHOLD
            })
            return result
            
        # 4. Try fuzzy match if enabled
        if use_fuzzy_matching:
            logger.debug(f"Attempting fuzzy match with threshold {FUZZY_MATCH_THRESHOLD}")
            similarity = self._calculate_text_similarity(expected, actual)
            logger.debug(f"Fuzzy match similarity score: {similarity:.2f}")
            
            if similarity >= FUZZY_MATCH_THRESHOLD:
                logger.info(f"Fuzzy match successful with score {similarity:.2f}")
                result.update({
                    "match_type": MatchType.FUZZY,
                    "success": True,
                    "similarity_score": similarity
                })
                return result
            else:
                logger.debug(f"Fuzzy match failed with score {similarity:.2f} (below threshold {FUZZY_MATCH_THRESHOLD})")
                
        # No match found
        logger.warning(f"No match found for text: '{expected}'")
        return result
        
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
            
        best_match = None
        best_score = 0.0
        
        logger.debug(f"Finding best match for '{text}' among {len(candidates)} candidates")
        for candidate in candidates:
            match_result = self.match_text(text, candidate)
            if match_result["success"] and match_result["similarity_score"] > best_score:
                best_score = match_result["similarity_score"]
                best_match = candidate
                logger.debug(f"New best match found: '{candidate}' with score {best_score:.2f}")
                
        if best_match:
            logger.info(f"Best match found: '{best_match}' with score {best_score:.2f}")
        else:
            logger.warning(f"No match found for text: '{text}'")
            
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
            
        logger.debug(f"Verifying text match with threshold {threshold}")
        logger.debug(f"Expected: '{expected}'")
        logger.debug(f"Actual: '{actual}'")
        
        # Get match result
        match_result = self.match_text(expected, actual, self._use_fuzzy_matching)
        
        # Create metadata
        metadata = {
            "expected": expected,
            "actual": actual,
            "match_type": str(match_result["match_type"]),
            "similarity": match_result["similarity_score"],
            "threshold": threshold,
            "normalized_expected": self._normalize_text(expected),
            "normalized_actual": self._normalize_text(actual)
        }
        
        if match_result["success"]:
            logger.info(f"Text match successful: {match_result['match_type']} with similarity {match_result['similarity_score']:.2f}")
        else:
            logger.warning(f"Text match failed: {match_result['match_type']} with similarity {match_result['similarity_score']:.2f}")
            
        return AssertionResult(
            success=match_result["success"],
            error_code=None if match_result["success"] else "TEXT_MISMATCH",
            message=f"Text match type: {match_result['match_type']}, similarity: {match_result['similarity_score']:.2f}",
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
            
        logger.debug(f"Verifying list match with threshold {threshold}")
        logger.debug(f"Expected items: {expected_items}")
        logger.debug(f"Actual items: {actual_items}")
        logger.debug(f"Match all: {match_all}")
        
        matched_items = []
        unmatched_items = []
        
        for expected in expected_items:
            logger.debug(f"Processing expected item: '{expected}'")
            best_match, score = self._find_best_match(expected, actual_items)
            
            if score >= threshold:
                logger.info(f"Item matched: '{expected}' -> '{best_match}' with score {score:.2f}")
                matched_items.append({
                    "expected": expected,
                    "actual": best_match,
                    "similarity": score
                })
            else:
                logger.warning(f"Item not matched: '{expected}' (best match: '{best_match}' with score {score:.2f})")
                unmatched_items.append({
                    "expected": expected,
                    "best_match": best_match,
                    "similarity": score
                })
                
        # Determine success based on match_all flag
        if match_all:
            success = len(matched_items) == len(expected_items)
            logger.debug(f"Match all mode: {len(matched_items)}/{len(expected_items)} items matched")
        else:
            success = len(matched_items) > 0
            logger.debug(f"Match any mode: {len(matched_items)} items matched")
            
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