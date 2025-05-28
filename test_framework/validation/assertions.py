"""
Assertions module for validating test results.
Provides high-level interface for test assertions.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from urllib.parse import urljoin
import json
import re
import logging
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from playwright.async_api import Page
from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

from .models import LinkExtractionResult
from .validator import TestValidator, ValidationMode, ValidationResult

logger = logging.getLogger("test_framework.validation.assertions")

class AssertionType(Enum):
    """Types of assertions supported"""
    TEXT = "text"
    LINK = "link"
    ATTRIBUTE = "attribute"
    LIST = "list"  # New type for list-based assertions
    MULTI_VALUE = "multi_value"  # New type for multi-value extractions

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

class TestAssertions:
    """High-level interface for test assertions"""
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize assertions with agent and result history"""
        logger.debug(f"Initializing TestAssertions with agent and result history")
        self.agent = agent
        self.result = result
        self.page = None
        self.step_extractions = self._build_step_extraction_map()
        self.current_step = 0  # Track current step
        self.requirement_step_map = {}  # Map requirements to their steps
        self.step_metadata = {}  # Store step metadata for better tracking
        self.extraction_links = {}  # Map assertions to their specific extractions
        
        # Calculate dynamic context window based on task characteristics
        self.step_context_window = self._calculate_context_window()
        logger.debug(f"Using dynamic context window of {self.step_context_window} steps")
        
        # Initialize semantic matching components
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized semantic matching model")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic model: {e}")
            self.semantic_model = None
            
    def _calculate_context_window(self) -> int:
        """Calculate appropriate context window based on task characteristics.
        
        Factors considered:
        1. Number of steps in the task
        2. Average extractions per step
        3. Task complexity indicators
        4. Requirement dependencies
        
        Returns:
            int: Number of previous steps to consider for context
        """
        if not self.step_extractions:
            return 1  # Default to 1 if no steps yet
            
        total_steps = len(self.step_extractions)
        total_extractions = sum(len(extractions) for extractions in self.step_extractions.values())
        avg_extractions_per_step = total_extractions / total_steps if total_steps > 0 else 0
        
        # Base window on task size
        if total_steps <= 3:
            base_window = 1  # Small tasks
        elif total_steps <= 7:
            base_window = 2  # Medium tasks
        else:
            base_window = 3  # Large tasks
            
        # Adjust based on extraction density
        if avg_extractions_per_step > 2:
            base_window += 1  # More extractions per step suggests more complex steps
            
        # Look for task complexity indicators
        complexity_indicators = [
            "form", "wizard", "multi-step", "sequence", "workflow",
            "progressive", "guided", "tutorial"
        ]
        
        task_description = getattr(self.agent, 'task_description', '').lower()
        if any(indicator in task_description for indicator in complexity_indicators):
            base_window += 1  # Increase window for complex tasks
            
        # Cap the window to prevent excessive lookback
        max_window = min(5, total_steps - 1)  # Never look back more than 5 steps or total steps - 1
        return min(base_window, max_window)
        
    def _update_context_window(self, requirement: str):
        """Update context window based on new requirement.
        
        Args:
            requirement: The new requirement being processed
        """
        # Check for cross-step dependencies
        if "from previous step" in requirement.lower() or \
           "from step" in requirement.lower() or \
           "across steps" in requirement.lower():
            # Increase window for requirements that explicitly reference other steps
            self.step_context_window = min(5, self.step_context_window + 1)
            logger.debug(f"Increased context window to {self.step_context_window} due to cross-step reference")
            
        # Check for complex validation requirements
        if "verify sequence" in requirement.lower() or \
           "verify workflow" in requirement.lower() or \
           "verify process" in requirement.lower():
            # Increase window for complex validations
            self.step_context_window = min(5, self.step_context_window + 1)
            logger.debug(f"Increased context window to {self.step_context_window} due to complex validation")
            
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using multiple methods.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Normalize texts
        text1 = ' '.join(text1.lower().split())
        text2 = ' '.join(text2.lower().split())
        
        # Calculate string similarity
        string_sim = SequenceMatcher(None, text1, text2).ratio()
        
        # If semantic model is available, calculate semantic similarity
        if self.semantic_model:
            try:
                # Get embeddings
                emb1 = self.semantic_model.encode([text1])[0]
                emb2 = self.semantic_model.encode([text2])[0]
                
                # Calculate cosine similarity
                semantic_sim = cosine_similarity([emb1], [emb2])[0][0]
                
                # Combine similarities (weighted average)
                return 0.3 * string_sim + 0.7 * semantic_sim
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
                return string_sim
                
        return string_sim
        
    def _is_goal_match(self, goal: str, expected_text: str, threshold: float = 0.8) -> Tuple[bool, float, str]:
        """Check if a goal matches the expected text using semantic similarity.
        
        Args:
            goal: The goal text to check
            expected_text: The expected text to match against
            threshold: Similarity threshold for considering a match
            
        Returns:
            Tuple[bool, float, str]: (is_match, similarity_score, match_type)
        """
        # Normalize texts
        goal_norm = ' '.join(goal.lower().split())
        expected_norm = ' '.join(expected_text.lower().split())
        
        # Check for exact goal match
        if f"extract the text '{expected_text}'" in goal_norm or \
           f"extract the text \"{expected_text}\"" in goal_norm:
            return True, 1.0, 'exact_goal_match'
            
        # Check for semantic similarity
        similarity = self._calculate_similarity(goal_norm, expected_norm)
        
        # Check for common variations
        variations = [
            f"extract {expected_norm}",
            f"get {expected_norm}",
            f"find {expected_norm}",
            f"locate {expected_norm}",
            f"check for {expected_norm}",
            f"verify {expected_norm}"
        ]
        
        for variation in variations:
            if variation in goal_norm:
                return True, 0.9, 'variation_match'
                
        # Return semantic match if similarity is above threshold
        if similarity >= threshold:
            return True, similarity, 'semantic_match'
            
        return False, similarity, 'no_match'
        
    def _build_step_extraction_map(self) -> Dict[int, List[Dict[str, Any]]]:
        """Build a map of step numbers to their extractions."""
        logger.debug("Building step extraction map")
        step_map = {}
        current_step = 1  # Start from step 1
        extraction_sequence = 0
        
        for history_item in self.result.history:
            if not history_item.model_output:
                continue
                
            # Extract step number from the action goal if available
            if hasattr(history_item.model_output, 'action') and history_item.model_output.action:
                action = history_item.model_output.action[-1]
                goal = ''
                
                # Get the goal from different possible locations
                if hasattr(action, 'extract_content'):
                    if isinstance(action.extract_content, dict):
                        goal = action.extract_content.get('goal', '')
                    elif isinstance(action.extract_content, str):
                        goal = action.extract_content
                elif hasattr(action, 'extract_text'):
                    if isinstance(action.extract_text, dict):
                        goal = action.extract_text.get('goal', '')
                    elif isinstance(action.extract_text, str):
                        goal = action.extract_text
                        
                # Try to extract step number from the goal
                if goal:
                    step_match = re.search(r'step\s+(\d+)', goal.lower())
                    if step_match:
                        current_step = int(step_match.group(1))
                        logger.debug(f"Found step number {current_step} in goal: {goal}")
                        
            # If this step hasn't been seen before, initialize its list
            if current_step not in step_map:
                step_map[current_step] = []
                logger.debug(f"Initialized step {current_step}")
                
            # Check for extractions in this step
            if history_item.model_output.action and history_item.result:
                action = history_item.model_output.action[-1]
                result = history_item.result[-1]
                
                # Handle different extraction formats
                extraction_goal = ''
                if hasattr(action, 'extract_content'):
                    if isinstance(action.extract_content, dict):
                        extraction_goal = action.extract_content.get('goal', '')
                    elif isinstance(action.extract_content, str):
                        extraction_goal = action.extract_content
                elif hasattr(action, 'extract_text'):
                    if isinstance(action.extract_text, dict):
                        extraction_goal = action.extract_text.get('goal', '')
                    elif isinstance(action.extract_text, str):
                        extraction_goal = action.extract_text
                        
                if extraction_goal:
                    extraction_sequence += 1
                    extraction_id = f"ext_{current_step}_{extraction_sequence}"
                    
                    # Check for linked assertions in the extraction metadata
                    linked_assertions = []
                    if hasattr(action, 'metadata') and isinstance(action.metadata, dict):
                        assertion_links = action.metadata.get('assertion_links', [])
                        if isinstance(assertion_links, list):
                            linked_assertions = assertion_links
                            
                    extraction_info = {
                        'id': extraction_id,
                        'action': action,
                        'result': result,
                        'goal': extraction_goal,
                        'sequence': extraction_sequence,
                        'timestamp': history_item.timestamp if hasattr(history_item, 'timestamp') else None,
                        'step_metadata': self.step_metadata.get(current_step, {}),
                        'linked_assertions': linked_assertions
                    }
                    
                    # Add extraction to step's list
                    step_map[current_step].append(extraction_info)
                    logger.debug(
                        f"Added extraction {extraction_id} to step {current_step} "
                        f"(sequence: {extraction_sequence}, "
                        f"goal: {extraction_info['goal']}, "
                        f"linked assertions: {linked_assertions})"
                    )
                    
        # Sort extractions within each step by sequence number
        for step in step_map:
            step_map[step].sort(key=lambda x: x['sequence'])
            logger.debug(f"Step {step} has {len(step_map[step])} extractions")
            
        logger.debug(f"Built step map with {len(step_map)} steps and {extraction_sequence} total extractions")
        return step_map
        
    def _extract_step_number(self, requirement: str) -> Optional[int]:
        """Extract step number from requirement text or metadata."""
        # First try to extract step number from the requirement text
        step_match = re.search(r'step\s+(\d+)', requirement.lower())
        if step_match:
            step_num = int(step_match.group(1))
            logger.debug(f"Found step number {step_num} in requirement: {requirement}")
            return step_num
            
        # If no step number in requirement, check if this requirement was previously mapped
        if requirement in self.requirement_step_map:
            step_num = self.requirement_step_map[requirement]
            logger.debug(f"Found cached step number {step_num} for requirement")
            return step_num
            
        # If still no step found, use current step
        logger.debug(f"No step number found in requirement or cache, using current step {self.current_step}")
        return self.current_step
        
    def _update_step_tracking(self, requirement: str, step_num: int):
        """Update step tracking information.
        
        Args:
            requirement: The requirement being processed
            step_num: The step number to associate with the requirement
        """
        self.current_step = step_num
        self.requirement_step_map[requirement] = step_num
        
        # Update context window based on step dependencies
        if step_num in self.step_metadata:
            dependencies = self.step_metadata[step_num].get('dependencies', [])
            if dependencies:
                # Ensure context window is large enough to cover dependencies
                max_dependency_step = max(dependencies)
                required_window = self.current_step - max_dependency_step
                if required_window > self.step_context_window:
                    self.step_context_window = min(5, required_window)
                    logger.debug(f"Adjusted context window to {self.step_context_window} based on step dependencies")
                    
        logger.debug(f"Updated step tracking - Current step: {step_num}, Total mapped requirements: {len(self.requirement_step_map)}")
        
    def _is_content_relevant(self, content: str, expected_text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if content is relevant to the expected text.
        
        Args:
            content: The content to check
            expected_text: The expected text to match against
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (is_relevant, reason, details)
        """
        if not content or not expected_text:
            return False, "Empty content or expected text", {
                "content_length": len(content) if content else 0,
                "expected_length": len(expected_text) if expected_text else 0
            }
            
        # Normalize both texts
        content_norm = ' '.join(content.lower().split())
        expected_norm = ' '.join(expected_text.lower().split())
        
        # Check for exact match
        if content_norm == expected_norm:
            return True, "Exact match found", {
                "content": content_norm,
                "expected": expected_norm
            }
            
        # Check for contains match with all terms
        expected_terms = set(expected_norm.split())
        content_terms = set(content_norm.split())
        
        # Calculate term overlap
        missing_terms = expected_terms - content_terms
        extra_terms = content_terms - expected_terms
        
        if not missing_terms:
            return True, "All expected terms found", {
                "content": content_norm,
                "expected": expected_norm,
                "matching_terms": list(expected_terms),
                "extra_terms": list(extra_terms)
            }
            
        return False, "Missing expected terms", {
            "content": content_norm,
            "expected": expected_norm,
            "missing_terms": list(missing_terms),
            "matching_terms": list(expected_terms - missing_terms),
            "extra_terms": list(extra_terms)
        }
        
    def _calculate_context_similarity(self, content: str, expected_text: str) -> float:
        """Calculate context similarity between content and expected text.
        
        Considers:
        1. Term proximity
        2. Term order
        3. Surrounding context
        4. HTML structure (if available)
        
        Args:
            content: The content to check
            expected_text: The expected text to match against
            
        Returns:
            float: Context similarity score between 0 and 1
        """
        if not content or not expected_text:
            return 0.0
            
        # Normalize texts
        content_norm = ' '.join(content.lower().split())
        expected_norm = ' '.join(expected_text.lower().split())
        
        # Get term positions
        content_terms = content_norm.split()
        expected_terms = expected_norm.split()
        
        # Calculate term proximity score
        proximity_score = 0.0
        if len(expected_terms) > 1:
            # Find positions of expected terms in content
            term_positions = {}
            for i, term in enumerate(content_terms):
                if term in expected_terms:
                    if term not in term_positions:
                        term_positions[term] = []
                    term_positions[term].append(i)
                    
            # Calculate average distance between consecutive terms
            if len(term_positions) == len(expected_terms):
                distances = []
                for i in range(len(expected_terms) - 1):
                    term1 = expected_terms[i]
                    term2 = expected_terms[i + 1]
                    if term1 in term_positions and term2 in term_positions:
                        # Find closest pair of positions
                        min_dist = float('inf')
                        for pos1 in term_positions[term1]:
                            for pos2 in term_positions[term2]:
                                dist = abs(pos2 - pos1)
                                min_dist = min(min_dist, dist)
                        if min_dist != float('inf'):
                            distances.append(min_dist)
                            
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    # Convert distance to score (closer = higher score)
                    proximity_score = 1.0 / (1.0 + avg_distance)
                    
        # Calculate term order score
        order_score = 0.0
        if len(expected_terms) > 1:
            # Find all possible sequences of expected terms in content
            sequences = []
            for i in range(len(content_terms) - len(expected_terms) + 1):
                sequence = content_terms[i:i + len(expected_terms)]
                if all(term in sequence for term in expected_terms):
                    sequences.append(sequence)
                    
            if sequences:
                # Calculate how well each sequence matches the expected order
                best_order_score = 0.0
                for sequence in sequences:
                    # Count how many terms are in the same relative order
                    correct_order = 0
                    for i in range(len(expected_terms) - 1):
                        term1 = expected_terms[i]
                        term2 = expected_terms[i + 1]
                        pos1 = sequence.index(term1)
                        pos2 = sequence.index(term2)
                        if pos1 < pos2:
                            correct_order += 1
                    sequence_score = correct_order / (len(expected_terms) - 1)
                    best_order_score = max(best_order_score, sequence_score)
                order_score = best_order_score
                
        # Combine scores with weights
        return 0.6 * proximity_score + 0.4 * order_score
        
    def _find_relevant_extraction(self, extractions: List[Dict[str, Any]], expected_text: str) -> Optional[Tuple[ActionResult, Dict[str, Any]]]:
        """Find the most relevant extraction from a list of extractions.
        
        Selection criteria (in order of priority):
        1. Exact goal match (highest priority)
        2. Semantic goal match with high similarity
        3. Content match with all expected terms
        4. Most recent extraction (lowest priority)
        
        Args:
            extractions: List of extractions to search through
            expected_text: The expected text to match against
            
        Returns:
            Optional[Tuple[ActionResult, Dict[str, Any]]]: The most relevant extraction and its metadata, or None
        """
        if not extractions:
            return None
            
        # Track best matches for each category
        best_matches = {
            'exact_goal': None,
            'semantic_goal': None,
            'content': None
        }
        
        # Track rejection reasons for debugging
        rejection_reasons = []
        
        # Track all semantic matches for disambiguation
        semantic_matches = []
        
        # Process extractions in reverse order (most recent first)
        for extraction in reversed(extractions):
            goal = extraction['goal']
            result = extraction['result']
            
            # 1. Check for exact goal match
            goal_normalized = ' '.join(goal.lower().split())
            if f"extract the text '{expected_text}'" in goal_normalized or \
               f"extract the text \"{expected_text}\"" in goal_normalized:
                logger.debug(f"Found exact goal match: {goal}")
                best_matches['exact_goal'] = (result, {
                    'selection_reason': 'exact_goal_match',
                    'goal': goal,
                    'sequence': extraction.get('sequence'),
                    'timestamp': extraction.get('timestamp'),
                    'match_score': 1.0
                })
                # Exact match is highest priority, so we can stop here
                return best_matches['exact_goal']
            else:
                rejection_reasons.append({
                    'type': 'exact_goal',
                    'goal': goal,
                    'reason': "Goal doesn't contain exact expected text"
                })
                
            # 2. Check for semantic goal match
            is_match, score, match_type = self._is_goal_match(goal, expected_text)
            if is_match and score >= 0.8:
                # Store all semantic matches for potential disambiguation
                semantic_matches.append({
                    'result': result,
                    'goal': goal,
                    'score': score,
                    'match_type': match_type,
                    'sequence': extraction.get('sequence'),
                    'timestamp': extraction.get('timestamp')
                })
                
                current_best = best_matches['semantic_goal']
                if not current_best or score > current_best[1]['match_score']:
                    logger.debug(f"Found semantic goal match (score: {score:.2f}): {goal}")
                    best_matches['semantic_goal'] = (result, {
                        'selection_reason': match_type,
                        'goal': goal,
                        'sequence': extraction.get('sequence'),
                        'timestamp': extraction.get('timestamp'),
                        'match_score': score
                    })
            else:
                rejection_reasons.append({
                    'type': 'semantic_goal',
                    'goal': goal,
                    'reason': f"Semantic match failed (score: {score:.2f})"
                })
                    
            # 3. Check for content match
            if hasattr(result, 'extracted_content'):
                content = result.extracted_content
                if isinstance(content, dict):
                    content_text = content.get('text', '')
                    is_relevant, reason, details = self._is_content_relevant(content_text, expected_text)
                    if is_relevant:
                        # Calculate context similarity for content matches
                        context_score = self._calculate_context_similarity(content_text, expected_text)
                        current_best = best_matches['content']
                        if not current_best or context_score > current_best[1].get('context_score', 0):
                            logger.debug(f"Found content match with context score {context_score:.2f}: {content_text}")
                            best_matches['content'] = (result, {
                                'selection_reason': 'content_match',
                                'goal': goal,
                                'content': content_text,
                                'sequence': extraction.get('sequence'),
                                'timestamp': extraction.get('timestamp'),
                                'match_score': 0.7,  # Base score for content matches
                                'context_score': context_score,
                                'match_details': details
                            })
                    else:
                        rejection_reasons.append({
                            'type': 'content',
                            'content': content_text,
                            'reason': reason,
                            'details': details
                        })
                else:
                    rejection_reasons.append({
                        'type': 'content',
                        'reason': "Extracted content is not a dictionary"
                    })
            else:
                rejection_reasons.append({
                    'type': 'content',
                    'reason': "No extracted content found"
                })
                            
        # If we have multiple semantic matches, try to disambiguate
        if len(semantic_matches) > 1:
            logger.debug(f"Found {len(semantic_matches)} semantic matches, attempting disambiguation")
            
            # Sort matches by score
            semantic_matches.sort(key=lambda x: x['score'], reverse=True)
            
            # If top matches are close in score, use additional criteria
            if semantic_matches[0]['score'] - semantic_matches[1]['score'] < 0.1:
                logger.debug("Top semantic matches are close in score, using additional criteria")
                
                # Check content context for top matches
                for match in semantic_matches[:2]:
                    if hasattr(match['result'], 'extracted_content'):
                        content = match['result'].extracted_content
                        if isinstance(content, dict):
                            content_text = content.get('text', '')
                            context_score = self._calculate_context_similarity(content_text, expected_text)
                            match['context_score'] = context_score
                            
                # Sort by context score if available
                semantic_matches.sort(key=lambda x: x.get('context_score', 0), reverse=True)
                
                # Update best match with disambiguation details
                best_match = semantic_matches[0]
                best_matches['semantic_goal'] = (best_match['result'], {
                    'selection_reason': 'disambiguated_semantic_match',
                    'goal': best_match['goal'],
                    'sequence': best_match['sequence'],
                    'timestamp': best_match['timestamp'],
                    'match_score': best_match['score'],
                    'context_score': best_match.get('context_score', 0),
                    'disambiguation_details': {
                        'total_matches': len(semantic_matches),
                        'top_matches': [
                            {
                                'score': m['score'],
                                'context_score': m.get('context_score', 0),
                                'goal': m['goal']
                            }
                            for m in semantic_matches[:2]
                        ]
                    }
                })
                
        # Select the best match based on priority
        if best_matches['exact_goal']:
            logger.debug("Using exact goal match")
            return best_matches['exact_goal']
        elif best_matches['semantic_goal']:
            logger.debug(f"Using semantic goal match (score: {best_matches['semantic_goal'][1]['match_score']:.2f})")
            return best_matches['semantic_goal']
        elif best_matches['content']:
            logger.debug("Using content match")
            return best_matches['content']
            
        # If no good match found, log detailed rejection reasons
        logger.warning("No relevant extraction found. Rejection reasons:")
        for reason in rejection_reasons:
            logger.warning(f"- {reason['type']}: {reason['reason']}")
            if 'details' in reason:
                logger.warning(f"  Details: {reason['details']}")
                
        # Use the most recent extraction as fallback
        if extractions:
            last_extraction = extractions[-1]
            logger.debug("No good match found, using most recent extraction")
            return last_extraction['result'], {
                'selection_reason': 'most_recent',
                'goal': last_extraction['goal'],
                'sequence': last_extraction.get('sequence'),
                'timestamp': last_extraction.get('timestamp'),
                'match_score': 0.0,
                'rejection_reasons': rejection_reasons  # Include rejection reasons in metadata
            }
            
        return None
        
    def _get_last_extraction_result(self, action_type: str, requirement: str) -> Optional[Tuple[ActionResult, Dict[str, Any]]]:
        """Get the extraction result that matches the current requirement.
        
        Matching strategy (in order of priority):
        1. Direct link between requirement and extraction
        2. Extraction in current step
        3. Extraction in previous steps (within context window)
        4. Extraction in future steps (within lookahead window)
        5. Best semantic match across all steps
        6. Extract text if not found
        7. Verify text directly if no extraction possible
        
        Args:
            action_type: Type of extraction action to look for
            requirement: The current requirement being verified
            
        Returns:
            Optional[Tuple[ActionResult, Dict[str, Any]]]: The matching extraction result and its metadata
        """
        logger.debug(f"Looking for extraction matching requirement: {requirement}")
        
        # Extract the expected text from the requirement
        expected_text = None
        if "assert" in requirement.lower() and "text" in requirement.lower():
            match = re.search(r'["\'](.*?)["\']', requirement)
            if match:
                expected_text = match.group(1)
                logger.debug(f"Extracted expected text from requirement: {expected_text}")
                
        # Extract step number from requirement
        step_num = self._extract_step_number(requirement)
        self._update_step_tracking(requirement, step_num)
        
        # Calculate lookahead window based on task characteristics
        lookahead_window = self._calculate_lookahead_window()
        logger.debug(f"Using lookahead window of {lookahead_window} steps")
        
        # Track all potential matches for semantic comparison
        all_matches = []
        
        # First, ensure all extractions are processed
        self._process_pending_extractions()
        
        # 1. First try to find a direct link between requirement and extraction
        if step_num in self.step_extractions:
            extractions = self.step_extractions[step_num]
            for extraction in extractions:
                linked_assertions = extraction.get('linked_assertions', [])
                if requirement in linked_assertions:
                    logger.debug(f"Found direct link between requirement and extraction {extraction['id']}")
                    return extraction['result'], {
                        'selection_reason': 'direct_link',
                        'extraction_id': extraction['id'],
                        'goal': extraction['goal'],
                        'sequence': extraction['sequence'],
                        'timestamp': extraction['timestamp'],
                        'match_score': 1.0
                    }
                    
        # 2. Try to find extraction in current step
        if step_num in self.step_extractions:
            extractions = self.step_extractions[step_num]
            if expected_text:
                result = self._find_relevant_extraction(extractions, expected_text)
                if result:
                    logger.debug(f"Found relevant extraction in current step {step_num}")
                    return result
                    
        # 3. Look in previous steps within context window
        for prev_step in range(step_num - 1, max(0, step_num - self.step_context_window - 1), -1):
            if prev_step in self.step_extractions:
                logger.debug(f"Looking for extraction in previous step {prev_step}")
                prev_extractions = self.step_extractions[prev_step]
                if expected_text:
                    result = self._find_relevant_extraction(prev_extractions, expected_text)
                    if result:
                        logger.debug(f"Found relevant extraction in previous step {prev_step}")
                        return result
                        
        # 4. Look in future steps within lookahead window
        for future_step in range(step_num + 1, min(len(self.step_extractions) + 1, step_num + lookahead_window + 1)):
            if future_step in self.step_extractions:
                logger.debug(f"Looking for extraction in future step {future_step}")
                future_extractions = self.step_extractions[future_step]
                if expected_text:
                    result = self._find_relevant_extraction(future_extractions, expected_text)
                    if result:
                        logger.debug(f"Found relevant extraction in future step {future_step}")
                        return result
                        
        # 5. If no match found in any step, try semantic matching across all steps
        if expected_text:
            logger.debug("No direct match found, trying semantic matching across all steps")
            for step, extractions in self.step_extractions.items():
                for extraction in extractions:
                    goal = extraction['goal']
                    result = extraction['result']
                    
                    # Calculate semantic similarity between goal and expected text
                    similarity = self._calculate_similarity(goal, expected_text)
                    if similarity >= 0.8:  # High similarity threshold for cross-step matching
                        all_matches.append({
                            'result': result,
                            'goal': goal,
                            'step': step,
                            'sequence': extraction['sequence'],
                            'timestamp': extraction['timestamp'],
                            'similarity': similarity
                        })
                        
            if all_matches:
                # Sort matches by similarity score
                all_matches.sort(key=lambda x: x['similarity'], reverse=True)
                best_match = all_matches[0]
                logger.debug(f"Found best semantic match in step {best_match['step']} with similarity {best_match['similarity']:.2f}")
                return best_match['result'], {
                    'selection_reason': 'semantic_match_across_steps',
                    'goal': best_match['goal'],
                    'step': best_match['step'],
                    'sequence': best_match['sequence'],
                    'timestamp': best_match['timestamp'],
                    'match_score': best_match['similarity'],
                    'all_matches': all_matches
                }
                
        # 6. If still no match found, try to extract the text
        if expected_text and hasattr(self.agent, 'page'):
            logger.debug(f"No extraction found, attempting to extract text: {expected_text}")
            try:
                # Create a new extraction action
                extraction_action = {
                    'extract_content': {
                        'goal': f"extract the text '{expected_text}'",
                        'should_strip_link_urls': True
                    }
                }
                
                # Execute the extraction
                result = self.agent.execute_action(extraction_action)
                if result and result.success:
                    logger.debug(f"Successfully extracted text: {expected_text}")
                    return result, {
                        'selection_reason': 'new_extraction',
                        'goal': f"extract the text '{expected_text}'",
                        'sequence': len(self.step_extractions.get(step_num, [])) + 1,
                        'timestamp': None,
                        'match_score': 1.0
                    }
            except Exception as e:
                logger.warning(f"Failed to extract text: {str(e)}")
                
        # 7. If extraction fails or is not possible, try direct verification
        if expected_text and hasattr(self.agent, 'page'):
            logger.debug(f"No extraction possible, attempting direct verification: {expected_text}")
            try:
                # Create a verification action
                verify_action = {
                    'verify_text': {
                        'text': expected_text,
                        'exact': True
                    }
                }
                
                # Execute the verification
                result = self.agent.execute_action(verify_action)
                if result and result.success:
                    logger.debug(f"Successfully verified text directly: {expected_text}")
                    return result, {
                        'selection_reason': 'direct_verification',
                        'goal': f"verify text '{expected_text}'",
                        'sequence': len(self.step_extractions.get(step_num, [])) + 1,
                        'timestamp': None,
                        'match_score': 1.0,
                        'verification_method': 'direct'
                    }
            except Exception as e:
                logger.warning(f"Failed to verify text directly: {str(e)}")
                
        # If still no match found, log warning and return None
        logger.warning(f"No relevant extraction or verification found for requirement in step {step_num}")
        return None
        
    def _process_pending_extractions(self):
        """Process any pending extractions in the history.
        
        This ensures that all extractions are processed before assertions are checked.
        """
        logger.debug("Processing pending extractions")
        
        # Get the current step number
        current_step = self.current_step
        
        # Process extractions in the current step
        if current_step in self.step_extractions:
            extractions = self.step_extractions[current_step]
            for extraction in extractions:
                if not extraction.get('processed', False):
                    logger.debug(f"Processing extraction {extraction['id']} in step {current_step}")
                    # Process the extraction
                    if hasattr(extraction['result'], 'extracted_content'):
                        content = extraction['result'].extracted_content
                        if isinstance(content, dict):
                            if 'content' in content:
                                content['text'] = content['content']
                            elif 'extracted_text' in content:
                                content['text'] = content['extracted_text']
                    extraction['processed'] = True
                    
        # Process extractions in future steps within lookahead window
        lookahead_window = self._calculate_lookahead_window()
        for future_step in range(current_step + 1, min(len(self.step_extractions) + 1, current_step + lookahead_window + 1)):
            if future_step in self.step_extractions:
                extractions = self.step_extractions[future_step]
                for extraction in extractions:
                    if not extraction.get('processed', False):
                        logger.debug(f"Processing extraction {extraction['id']} in future step {future_step}")
                        # Process the extraction
                        if hasattr(extraction['result'], 'extracted_content'):
                            content = extraction['result'].extracted_content
                            if isinstance(content, dict):
                                if 'content' in content:
                                    content['text'] = content['content']
                                elif 'extracted_text' in content:
                                    content['text'] = content['extracted_text']
                        extraction['processed'] = True

    def _calculate_lookahead_window(self) -> int:
        """Calculate appropriate lookahead window based on task characteristics.
        
        Returns:
            int: Number of future steps to look ahead
        """
        if not self.step_extractions:
            return 1  # Default to 1 if no steps yet
            
        total_steps = len(self.step_extractions)
        
        # Base window on task size
        if total_steps <= 3:
            base_window = 1  # Small tasks
        elif total_steps <= 7:
            base_window = 2  # Medium tasks
        else:
            base_window = 3  # Large tasks
            
        # Look for task complexity indicators
        complexity_indicators = [
            "form", "wizard", "multi-step", "sequence", "workflow",
            "progressive", "guided", "tutorial"
        ]
        
        task_description = getattr(self.agent, 'task_description', '').lower()
        if any(indicator in task_description for indicator in complexity_indicators):
            base_window += 1  # Increase window for complex tasks
            
        # Cap the window to prevent excessive lookahead
        max_window = min(5, total_steps - 1)  # Never look ahead more than 5 steps or total steps - 1
        return min(base_window, max_window)
        
    def _process_extraction_result(self, result: Any) -> ActionResult:
        """Process an extraction result into a standardized ActionResult format.
        
        Args:
            result: The raw extraction result
            
        Returns:
            ActionResult: Processed result in standard format
        """
        logger.debug(f"Processing extraction result: {result}")
        
        # If already an ActionResult with extracted_content, return as is
        if hasattr(result, 'extracted_content'):
            return result
            
        # Create a new ActionResult with the content
        content = result
        if isinstance(content, dict):
            logger.debug(f"Processing dictionary content: {content}")
            # Handle element-specific extraction
            if any(key.startswith('element_') for key in content.keys()):
                # Get the first element's text
                element_text = next(iter(content.values()))
                content = {"text": element_text}
                logger.debug(f"Converted element text: {content}")
            # Handle page content extraction
            elif "page_content" in content:
                content = {"text": content["page_content"]}
                logger.debug(f"Converted page content: {content}")
            # Handle extracted_text format
            elif "extracted_text" in content:
                content = {"text": content["extracted_text"]}
                logger.debug(f"Converted extracted text: {content}")
        else:
            # Handle raw text content
            content = {"text": str(content)}
            logger.debug(f"Converted raw content: {content}")
            
        return ActionResult(
            success=True,
            extracted_content=content,
            data=content
        )
        
    def _parse_extracted_content(self, content: Any, expected_type: str) -> Dict[str, Any]:
        """Parse extracted content with proper error handling and logging.
        
        Args:
            content: The content to parse
            expected_type: Type of content being parsed (for logging)
            
        Returns:
            Dict[str, Any]: Parsed content info with status and value
        """
        logger.debug(f"Parsing extracted content for {expected_type}: {content}")
        result = {
            "parsing_status": "unknown",
            "parsing_error": None,
            "content_type": type(content).__name__,
            "value": None
        }
        
        if content is None:
            logger.warning(f"Extracted content is None for {expected_type}")
            return result
            
        if isinstance(content, dict):
            logger.info(f"Extracted content is a dictionary for {expected_type}")
            result["parsing_status"] = "dict"
            # Handle different dictionary formats
            if "extracted_text" in content:
                result["value"] = {"text": content["extracted_text"]}
            elif "text" in content:
                result["value"] = content
            else:
                result["value"] = content
            return result
            
        try:
            data = json.loads(content)
            logger.info(f"Successfully parsed JSON content for {expected_type}")
            result["parsing_status"] = "json"
            # Handle different JSON formats
            if isinstance(data, dict):
                if "extracted_text" in data:
                    result["value"] = {"text": data["extracted_text"]}
                elif "text" in data:
                    result["value"] = data
                else:
                    result["value"] = data
            else:
                result["value"] = {"text": str(data)}
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON content for {expected_type}: {str(e)}")
            result["parsing_status"] = "raw"
            result["parsing_error"] = str(e)
            result["value"] = {"text": str(content)}
            return result
            
    def _create_metadata(self, requirement: str, expected_value: Any, content_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized metadata for verification results.
        
        Args:
            requirement: The original requirement string
            expected_value: The expected value to verify
            content_info: Parsed content information
            
        Returns:
            Dict[str, Any]: Standardized metadata dictionary
        """
        return {
            "requirement": requirement,
            "expected_value": expected_value,
            "normalization": {
                "is_relative": False,
                "base_url": None,
                "normalized": False
            },
            "parsing": content_info
        }
        
    async def _verify_condition(self, requirement: str) -> AssertionResult:
        """Verify a single requirement condition using history.
        
        Args:
            requirement: The requirement string to verify
            
        Returns:
            AssertionResult: Result of the verification
        """
        logger.debug(f"Verifying condition: {requirement}")
        
        # Check for list-based requirements first
        if any(keyword in requirement.lower() for keyword in ["verify all", "verify any", "verify list"]):
            logger.debug("Processing list-based requirement")
            return self._process_list_requirement(requirement)
            
        # Extract the condition type and parameters from the requirement
        if "assert" in requirement.lower() and "text" in requirement.lower():
            logger.debug("Processing text assertion")
            # Extract text to verify
            match = re.search(r'["\'](.*?)["\']', requirement)
            if not match:
                logger.error(f"Could not extract text from requirement: {requirement}")
                return AssertionResult(
                    success=False,
                    error_code="INVALID_REQUIREMENT",
                    message="Could not extract text to verify",
                    metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
                )
            expected_text = match.group(1)
            logger.debug(f"Extracted expected text: {expected_text}")
            
            # Get the last content extraction
            logger.debug("Getting last content extraction")
            extraction_result = self._get_last_extraction_result("extract_text", requirement)
            if not extraction_result:
                logger.error("No content extraction found in history")
                return AssertionResult(
                    success=False,
                    error_code="NO_EXTRACTION",
                    message="No content extraction found in history",
                    metadata=self._create_metadata(requirement, expected_text, {"parsing_status": "error"})
                )
                
            last_extraction, extraction_metadata = extraction_result
            
            # Parse extracted content
            logger.debug(f"Parsing extracted content: {last_extraction}")
            content_info = self._parse_extracted_content(last_extraction.extracted_content, "text")
            logger.debug(f"Content info: {content_info}")
            
            # Create metadata
            metadata = self._create_metadata(requirement, expected_text, content_info)
            
            # Add extraction selection details
            metadata.update({
                "extraction_selection": {
                    "reason": extraction_metadata['selection_reason'],
                    "goal": extraction_metadata['goal'],
                    "sequence": extraction_metadata['sequence'],
                    "timestamp": extraction_metadata['timestamp']
                }
            })
            
            # Add actual text based on parsing result
            if content_info["parsing_status"] in ["dict", "json"]:
                logger.debug("Using dictionary/JSON content")
                metadata["actual_text"] = content_info["value"].get('text', str(content_info["value"]))
            else:
                logger.debug("Using raw content")
                metadata["actual_text"] = content_info["value"]
            
            # Verify the text
            logger.debug(f"Verifying text with metadata: {metadata}")
            return self.verify_text(last_extraction, expected_text, exact=False, metadata=metadata)
            
        elif "verify link" in requirement.lower():
            logger.debug("Processing link verification")
            # Extract link text and href
            text_match = re.search(r'verify link ["\'](.*?)["\']', requirement.lower())
            href_match = re.search(r'href=["\'](.*?)["\']', requirement.lower())
            
            if not text_match:
                logger.error(f"Could not extract link text from requirement: {requirement}")
                return AssertionResult(
                    success=False,
                    error_code="INVALID_REQUIREMENT",
                    message="Could not extract link text to verify",
                    metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
                )
                
            expected_text = text_match.group(1)
            expected_href = href_match.group(1) if href_match else None
            logger.debug(f"Extracted expected link text: {expected_text}, href: {expected_href}")
            
            # Get the last content extraction
            last_extraction = self._get_last_extraction_result("extract_link", requirement)
            if not last_extraction:
                logger.error("No content extraction found in history")
                return AssertionResult(
                    success=False,
                    error_code="NO_EXTRACTION",
                    message="No content extraction found in history",
                    metadata=self._create_metadata(requirement, expected_text, {"parsing_status": "error"})
                )
            
            # Parse extracted content
            content_info = self._parse_extracted_content(last_extraction.extracted_content, "link")
            logger.debug(f"Content info: {content_info}")
            
            # Create metadata
            metadata = self._create_metadata(requirement, expected_text, content_info)
            metadata["expected_href"] = expected_href
            
            # Add actual values based on parsing result
            if content_info["parsing_status"] in ["dict", "json"]:
                logger.debug("Using dictionary/JSON content")
                metadata["actual_text"] = content_info["value"].get('text')
                metadata["actual_href"] = content_info["value"].get('href')
            else:
                logger.debug("Using raw content")
                metadata["actual_text"] = content_info["value"]
                metadata["actual_href"] = None
            
            return self.verify_link(last_extraction, expected_text, expected_href, metadata=metadata)
            
        elif "verify attribute" in requirement.lower():
            logger.debug("Processing attribute verification")
            # Extract attribute value
            match = re.search(r'verify attribute ["\'](.*?)["\']', requirement.lower())
            if not match:
                logger.error(f"Could not extract attribute value from requirement: {requirement}")
                return AssertionResult(
                    success=False,
                    error_code="INVALID_REQUIREMENT",
                    message="Could not extract attribute value to verify",
                    metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
                )
            expected_value = match.group(1)
            logger.debug(f"Extracted expected attribute value: {expected_value}")
            
            # Get the last content extraction
            last_extraction = self._get_last_extraction_result("extract_attribute", requirement)
            if not last_extraction:
                logger.error("No content extraction found in history")
                return AssertionResult(
                    success=False,
                    error_code="NO_EXTRACTION",
                    message="No content extraction found in history",
                    metadata=self._create_metadata(requirement, expected_value, {"parsing_status": "error"})
                )
            
            # Parse extracted content
            content_info = self._parse_extracted_content(last_extraction.extracted_content, "attribute")
            logger.debug(f"Content info: {content_info}")
            
            # Create metadata
            metadata = self._create_metadata(requirement, expected_value, content_info)
            
            # Add actual value based on parsing result
            if content_info["parsing_status"] in ["dict", "json"]:
                logger.debug("Using dictionary/JSON content")
                metadata["actual_value"] = content_info["value"].get('value', str(content_info["value"]))
            else:
                logger.debug("Using raw content")
                metadata["actual_value"] = content_info["value"]
            
            return self.verify_attribute(last_extraction, expected_value, metadata=metadata)
            
        logger.error(f"Unknown requirement type: {requirement}")
        return AssertionResult(
            success=False,
            error_code="UNKNOWN_REQUIREMENT",
            message=f"Unknown requirement type: {requirement}",
            metadata=self._create_metadata(requirement, None, {"parsing_status": "error"})
        )
    
    @staticmethod
    def verify_text(
        action_result: ActionResult,
        expected_text: str,
        exact: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssertionResult:
        """Verify text content"""
        logger.debug(f"Verifying text: expected='{expected_text}', exact={exact}")
        
        if not action_result.success:
            logger.error(f"Text verification failed: {action_result.error}")
            return AssertionResult(
                success=False,
                error_code="ACTION_FAILED",
                message=f"Action failed: {action_result.error}",
                metadata=metadata
            )
            
        # Handle extracted content safely
        actual_text = None
        if action_result.extracted_content:
            logger.debug(f"Processing extracted content: {action_result.extracted_content}")
            if isinstance(action_result.extracted_content, dict):
                logger.debug("Content is a dictionary")
                actual_text = action_result.extracted_content.get('text', str(action_result.extracted_content))
            else:
                try:
                    data = json.loads(action_result.extracted_content)
                    logger.debug("Successfully parsed JSON content")
                    actual_text = data.get('text', action_result.extracted_content)
                except json.JSONDecodeError:
                    logger.debug("Using raw string content")
                    actual_text = action_result.extracted_content

        if not actual_text:
            logger.error("No text content found")
            return AssertionResult(
                success=False,
                error_code="TEXT_EMPTY",
                message="No text content found",
                metadata=metadata
            )
            
        # Use the validator to check the text
        mode = ValidationMode.EXACT if exact else ValidationMode.CONTAINS
        logger.debug(f"Validating text with mode: {mode}")
        result = TestValidator.validate_text(
            expected=expected_text,
            actual=actual_text,
            mode=mode,
            original_requirement=metadata.get("requirement", "") if metadata else ""
        )
        
        if not result.success:
            logger.error(f"Text validation failed: {result.reason}")
        else:
            logger.info("Text validation succeeded")
        
        # Update metadata with actual values
        if metadata is None:
            metadata = {}
        metadata.update({
            "expected_text": expected_text,
            "actual_text": actual_text,
            "validation_mode": mode.value,
            "validation_result": {
                "success": result.success,
                "error_code": result.error_code,
                "reason": result.reason
            }
        })
        
        return AssertionResult(
            success=result.success,
            error_code=result.error_code,
            message=result.reason,
            metadata=metadata
        )
    
    @staticmethod
    def verify_link(
        action_result: ActionResult,
        expected_text: str,
        expected_href: str,
        exact: bool = True,
        is_relative: bool = False,
        base_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssertionResult:
        """Verify link text and href"""
        logger.debug(f"Verifying link: text='{expected_text}', href='{expected_href}', exact={exact}")
        
        if not action_result.success:
            logger.error(f"Link verification failed: {action_result.error}")
            return AssertionResult(
                success=False,
                error_code="ACTION_FAILED",
                message=f"Action failed: {action_result.error}",
                metadata=metadata
            )
            
        # Handle extracted content safely
        actual_text = None
        actual_href = None
        
        if action_result.extracted_content:
            logger.debug(f"Processing extracted content: {action_result.extracted_content}")
            if isinstance(action_result.extracted_content, dict):
                logger.debug("Content is a dictionary")
                actual_text = action_result.extracted_content.get('text')
                actual_href = action_result.extracted_content.get('href')
            else:
                try:
                    data = json.loads(action_result.extracted_content)
                    logger.debug("Successfully parsed JSON content")
                    actual_text = data.get('text')
                    actual_href = data.get('href')
                except json.JSONDecodeError:
                    logger.debug("Using raw string content")
                    actual_text = action_result.extracted_content
                    actual_href = None
            
        if not actual_text:
            logger.error("Link element not found")
            return AssertionResult(
                success=False,
                error_code="LINK_NOT_FOUND",
                message="Link element not found",
                metadata=metadata
            )
            
        # Use the validator to check the link
        mode = ValidationMode.EXACT if exact else ValidationMode.CONTAINS
        logger.debug(f"Validating link with mode: {mode}")
        result = TestValidator.validate_link(
            expected_text=expected_text,
            expected_href=expected_href,
            actual_text=actual_text,
            actual_href=actual_href,
            base_url=base_url,
            is_relative=is_relative,
            mode=mode,
            original_requirement=metadata.get("requirement", "") if metadata else ""
        )
        
        if not result.success:
            logger.error(f"Link validation failed: {result.reason}")
        else:
            logger.info("Link validation succeeded")
        
        # Update metadata with actual values
        if metadata is None:
            metadata = {}
        metadata.update({
            "expected_text": expected_text,
            "expected_href": expected_href,
            "actual_text": actual_text,
            "actual_href": actual_href,
            "validation_mode": mode.value,
            "validation_result": {
                "success": result.success,
                "error_code": result.error_code,
                "reason": result.reason
            }
        })
        
        return AssertionResult(
            success=result.success,
            error_code=result.error_code,
            message=result.reason,
            metadata=metadata
        )
    
    @staticmethod
    def verify_attribute(
        action_result: ActionResult,
        expected_value: str,
        relaxed: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssertionResult:
        """Verify attribute value"""
        logger.debug(f"Verifying attribute: expected='{expected_value}', relaxed={relaxed}")
        
        if not action_result.success:
            logger.error(f"Attribute verification failed: {action_result.error}")
            return AssertionResult(
                success=False,
                error_code="ACTION_FAILED",
                message=f"Action failed: {action_result.error}",
                metadata=metadata
            )
            
        # Handle extracted content safely
        actual_value = None
        if action_result.extracted_content:
            logger.debug(f"Processing extracted content: {action_result.extracted_content}")
            if isinstance(action_result.extracted_content, dict):
                logger.debug("Content is a dictionary")
                actual_value = action_result.extracted_content.get('value', str(action_result.extracted_content))
            else:
                try:
                    data = json.loads(action_result.extracted_content)
                    logger.debug("Successfully parsed JSON content")
                    actual_value = data.get('value', action_result.extracted_content)
                except json.JSONDecodeError:
                    logger.debug("Using raw string content")
                    actual_value = action_result.extracted_content

        if not actual_value:
            logger.error("Attribute not found")
            return AssertionResult(
                success=False,
                error_code="ATTRIBUTE_NOT_FOUND",
                message="Attribute not found",
                metadata=metadata
            )
            
        # Use the validator to check the attribute
        mode = ValidationMode.RELAXED if relaxed else ValidationMode.EXACT
        logger.debug(f"Validating attribute with mode: {mode}")
        result = TestValidator.validate_text(
            expected=expected_value,
            actual=actual_value,
            mode=mode,
            original_requirement=metadata.get("requirement", "") if metadata else ""
        )
        
        if not result.success:
            logger.error(f"Attribute validation failed: {result.reason}")
        else:
            logger.info("Attribute validation succeeded")
        
        # Update metadata with actual values
        if metadata is None:
            metadata = {}
        metadata.update({
            "expected_value": expected_value,
            "actual_value": actual_value,
            "validation_mode": mode.value,
            "validation_result": {
                "success": result.success,
                "error_code": result.error_code,
                "reason": result.reason
            }
        })
        
        return AssertionResult(
            success=result.success,
            error_code=result.error_code,
            message=result.reason,
            metadata=metadata
        )

    @staticmethod
    def verify_list(
        action_result: ActionResult,
        expected_items: List[str],
        match_all: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ListAssertionResult:
        """Verify a list of items against expected values.
        
        Args:
            action_result: The action result containing extracted items
            expected_items: List of expected items to find
            match_all: If True, all items must be found. If False, at least one item must be found.
            metadata: Optional metadata for the assertion
            
        Returns:
            ListAssertionResult: Result of the list verification
        """
        logger.debug(f"Verifying list: expected_items={expected_items}, match_all={match_all}")
        
        if not action_result.success:
            logger.error(f"List verification failed: {action_result.error}")
            return ListAssertionResult(
                success=False,
                error_code="ACTION_FAILED",
                message=f"Action failed: {action_result.error}",
                metadata=metadata,
                matched_items=[],
                unmatched_items=[],
                total_items=0,
                matched_count=0
            )
            
        # Handle extracted content safely
        actual_items = []
        if action_result.extracted_content:
            logger.debug(f"Processing extracted content: {action_result.extracted_content}")
            if isinstance(action_result.extracted_content, dict):
                # Handle dictionary format
                if "items" in action_result.extracted_content:
                    actual_items = action_result.extracted_content["items"]
                elif "list" in action_result.extracted_content:
                    actual_items = action_result.extracted_content["list"]
                else:
                    # Try to convert dictionary values to list
                    actual_items = list(action_result.extracted_content.values())
            elif isinstance(action_result.extracted_content, list):
                actual_items = action_result.extracted_content
            else:
                try:
                    # Try to parse as JSON
                    data = json.loads(action_result.extracted_content)
                    if isinstance(data, list):
                        actual_items = data
                    elif isinstance(data, dict):
                        if "items" in data:
                            actual_items = data["items"]
                        elif "list" in data:
                            actual_items = data["list"]
                        else:
                            actual_items = list(data.values())
                except json.JSONDecodeError:
                    # Treat as single item
                    actual_items = [str(action_result.extracted_content)]
                    
        if not actual_items:
            logger.error("No items found in extraction")
            return ListAssertionResult(
                success=False,
                error_code="NO_ITEMS",
                message="No items found in extraction",
                metadata=metadata,
                matched_items=[],
                unmatched_items=[],
                total_items=0,
                matched_count=0
            )
            
        # Normalize items for comparison
        expected_items = [str(item).lower().strip() for item in expected_items]
        actual_items = [str(item).lower().strip() for item in actual_items]
        
        # Find matches and non-matches
        matched_items = []
        unmatched_items = []
        
        for actual_item in actual_items:
            # Check for exact match first
            if actual_item in expected_items:
                matched_items.append({
                    "item": actual_item,
                    "match_type": "exact",
                    "score": 1.0
                })
            else:
                # Try semantic matching
                best_match = None
                best_score = 0.0
                
                for expected_item in expected_items:
                    similarity = TestAssertions._calculate_similarity(actual_item, expected_item)
                    if similarity > best_score and similarity >= 0.8:  # Threshold for semantic match
                        best_score = similarity
                        best_match = expected_item
                        
                if best_match:
                    matched_items.append({
                        "item": actual_item,
                        "match_type": "semantic",
                        "score": best_score,
                        "matched_with": best_match
                    })
                else:
                    unmatched_items.append({
                        "item": actual_item,
                        "reason": "No matching expected item found"
                    })
                    
        # Determine success based on match_all flag
        total_items = len(actual_items)
        matched_count = len(matched_items)
        
        if match_all:
            success = matched_count == len(expected_items)
            if not success:
                missing_items = set(expected_items) - {m.get("matched_with", m["item"]) for m in matched_items}
                message = f"Missing expected items: {', '.join(missing_items)}"
            else:
                message = "All expected items found"
        else:
            success = matched_count > 0
            message = f"Found {matched_count} matching items" if success else "No matching items found"
            
        # Update metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "expected_items": expected_items,
            "actual_items": actual_items,
            "match_all": match_all,
            "matching_details": {
                "total_items": total_items,
                "matched_count": matched_count,
                "unmatched_count": len(unmatched_items)
            }
        })
        
        return ListAssertionResult(
            success=success,
            error_code=None if success else "LIST_MISMATCH",
            message=message,
            metadata=metadata,
            matched_items=matched_items,
            unmatched_items=unmatched_items,
            total_items=total_items,
            matched_count=matched_count
        )
        
    def _process_list_requirement(self, requirement: str) -> ListAssertionResult:
        """Process a list-based requirement.
        
        Args:
            requirement: The requirement string to process
            
        Returns:
            ListAssertionResult: Result of the list verification
        """
        logger.debug(f"Processing list requirement: {requirement}")
        
        # Extract expected items from requirement
        expected_items = []
        match_all = True
        
        # Handle different list requirement formats
        if "verify all" in requirement.lower():
            match_all = True
            # Extract items after "verify all"
            items_text = requirement.lower().split("verify all")[-1].strip()
            expected_items = [item.strip() for item in items_text.split(",")]
        elif "verify any" in requirement.lower():
            match_all = False
            # Extract items after "verify any"
            items_text = requirement.lower().split("verify any")[-1].strip()
            expected_items = [item.strip() for item in items_text.split(",")]
        elif "verify list" in requirement.lower():
            # Extract items between quotes or after "verify list"
            match = re.search(r'["\'](.*?)["\']', requirement)
            if match:
                items_text = match.group(1)
                expected_items = [item.strip() for item in items_text.split(",")]
            else:
                items_text = requirement.lower().split("verify list")[-1].strip()
                expected_items = [item.strip() for item in items_text.split(",")]
                
        if not expected_items:
            logger.error(f"Could not extract expected items from requirement: {requirement}")
            return ListAssertionResult(
                success=False,
                error_code="INVALID_REQUIREMENT",
                message="Could not extract expected items",
                metadata={"requirement": requirement}
            )
            
        # Get the last content extraction
        extraction_result = self._get_last_extraction_result("extract_list", requirement)
        if not extraction_result:
            logger.error("No content extraction found in history")
            return ListAssertionResult(
                success=False,
                error_code="NO_EXTRACTION",
                message="No content extraction found in history",
                metadata={"requirement": requirement}
            )
            
        last_extraction, extraction_metadata = extraction_result
        
        # Create metadata
        metadata = {
            "requirement": requirement,
            "expected_items": expected_items,
            "match_all": match_all,
            "extraction_selection": extraction_metadata
        }
        
        # Verify the list
        return self.verify_list(last_extraction, expected_items, match_all, metadata)