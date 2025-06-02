"""
Step tracking module for test assertions.
Handles step context and history management.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.agent.service import Agent

from .base import BaseAssertions, AssertionResult
from .matching import MatchingAssertions

logger = logging.getLogger("test_framework.validation.assertions")

@dataclass
class StepContext:
    """Context information for a step"""
    step_number: int
    requirements: List[str]
    extractions: Dict[str, Any]
    assertions: List[AssertionResult]
    
class StepTrackingAssertions(MatchingAssertions):
    """Handles step context and history management"""
    
    def __init__(self, agent: Agent, result: AgentHistoryList):
        """Initialize step tracking assertions"""
        super().__init__(agent, result)
        self._step_contexts: Dict[int, StepContext] = {}
        self._current_step = 0
        self._step_context_window = 3  # Increased from 2 to 3 for better context
        self._last_navigation_step = -1
        
    def _build_step_extraction_map(self, current_step: int) -> Dict[str, Any]:
        """Build a map of extractions for the current step and context window.
        
        Args:
            current_step: Current step number
            
        Returns:
            Dict[str, Any]: Map of extractions
        """
        logger.debug(f"Building extraction map for step {current_step}")
        extraction_map = {}
        
        # Process current and previous steps within context window
        start_step = max(0, current_step - self._step_context_window)
        for step in range(start_step, current_step + 1):
            action = self.result[step]
            if not action.extract_content:
                continue
                
            # Get extraction goal
            goal = ""
            if isinstance(action.extract_content, dict):
                goal = action.extract_content.get('goal', '')
            elif isinstance(action.extract_content, str):
                try:
                    import json
                    content = json.loads(action.extract_content)
                    goal = content.get('goal', '')
                except json.JSONDecodeError:
                    goal = str(action.extract_content)
                    
            if not goal:
                continue
                
            # Add to map
            extraction_map[goal] = {
                'step': step,
                'content': action.extract_content
            }
            
        return extraction_map
        
    def update_step_context(self, step_number: int, requirements: List[str], extractions: Dict[str, Any], assertions: List[AssertionResult]):
        """Update the context for a step.
        
        Args:
            step_number: Step number
            requirements: List of requirements
            extractions: Map of extractions
            assertions: List of assertions
        """
        self._step_contexts[step_number] = StepContext(
            step_number=step_number,
            requirements=requirements,
            extractions=extractions,
            assertions=assertions
        )
        
        # Update current step
        self._current_step = step_number
        
        # Check if this was a navigation step
        if any('navigate' in req.lower() for req in requirements):
            self._last_navigation_step = step_number
            
    def get_step_context(self, step_number: int) -> Optional[StepContext]:
        """Get the context for a step.
        
        Args:
            step_number: Step number
            
        Returns:
            Optional[StepContext]: Step context if available
        """
        return self._step_contexts.get(step_number)
        
    def get_last_navigation_step(self) -> int:
        """Get the last navigation step number.
        
        Returns:
            int: Last navigation step number
        """
        return self._last_navigation_step
        
    def _get_previous_extractions(self, current_step: int, requirement: str) -> List[Tuple[int, Any]]:
        """Get extractions from previous steps that might be relevant.
        
        Args:
            current_step: Current step number
            requirement: Requirement to find extractions for
            
        Returns:
            List[Tuple[int, Any]]: List of (step, content) tuples
        """
        logger.debug(f"Getting previous extractions for step {current_step}")
        
        relevant_extractions = []
        start_step = max(0, current_step - self._step_context_window)
        
        for step in range(start_step, current_step):
            action = self.result[step]
            if not action.extract_content:
                continue
                
            # Calculate similarity between requirement and extraction goal
            goal = ""
            if isinstance(action.extract_content, dict):
                goal = action.extract_content.get('goal', '')
            elif isinstance(action.extract_content, str):
                try:
                    import json
                    content = json.loads(action.extract_content)
                    goal = content.get('goal', '')
                except json.JSONDecodeError:
                    goal = str(action.extract_content)
                    
            if not goal:
                continue
                
            similarity = self._calculate_text_similarity(requirement, goal)
            if similarity > 0.8:
                relevant_extractions.append((step, action.extract_content))
                
        return relevant_extractions
        
    def _get_future_extractions(self, current_step: int, requirement: str) -> List[Tuple[int, Any]]:
        """Get extractions from future steps that might be relevant.
        
        Args:
            current_step: Current step number
            requirement: Requirement to find extractions for
            
        Returns:
            List[Tuple[int, Any]]: List of (step, content) tuples
        """
        logger.debug(f"Getting future extractions for step {current_step}")
        
        relevant_extractions = []
        end_step = min(current_step + self._step_context_window + 1, len(self.result))
        
        for step in range(current_step + 1, end_step):
            action = self.result[step]
            if not action.extract_content:
                continue
                
            # Calculate similarity between requirement and extraction goal
            goal = ""
            if isinstance(action.extract_content, dict):
                goal = action.extract_content.get('goal', '')
            elif isinstance(action.extract_content, str):
                try:
                    import json
                    content = json.loads(action.extract_content)
                    goal = content.get('goal', '')
                except json.JSONDecodeError:
                    goal = str(action.extract_content)
                    
            if not goal:
                continue
                
            similarity = self._calculate_text_similarity(requirement, goal)
            if similarity > 0.8:
                relevant_extractions.append((step, action.extract_content))
                
        return relevant_extractions
        
    def _calculate_lookahead_window(self, total_steps: int) -> int:
        """Calculate appropriate lookahead window based on task characteristics.
        
        Args:
            total_steps: Total number of steps in the task
            
        Returns:
            int: Number of steps to look ahead
        """
        # Base window on task size
        if total_steps <= 5:
            return 1
        elif total_steps <= 10:
            return 2
        else:
            return 3
            
    def _is_step_complete(self, step_number: int) -> bool:
        """Check if a step is complete (all requirements verified).
        
        Args:
            step_number: Step number to check
            
        Returns:
            bool: True if step is complete
        """
        context = self.get_step_context(step_number)
        if not context:
            return False
            
        # Check if all requirements have assertions
        return len(context.assertions) == len(context.requirements)
        
    def _get_incomplete_steps(self) -> List[int]:
        """Get list of incomplete steps.
        
        Returns:
            List[int]: List of incomplete step numbers
        """
        return [
            step for step, context in self._step_contexts.items()
            if not self._is_step_complete(step)
        ] 