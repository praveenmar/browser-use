"""
Step hooks for the test framework.
Provides lifecycle hooks for test steps including before, after, error, and completion handlers.
"""

import logging
from typing import Optional, Dict, Any, NoReturn
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.session import BrowserSession
from browser_use.agent.views import ActionModel, ActionResult

logger = logging.getLogger("test_framework.agent.step_hooks")

class StepHooks:
    """Hooks for test steps.
    
    This class provides lifecycle hooks for test steps, allowing for custom behavior
    before and after each step, error handling, and test completion.
    """
    
    def __init__(self, agent: Any) -> None:
        """Initialize step hooks.
        
        Args:
            agent: The test agent instance that owns these hooks.
            
        Raises:
            ValueError: If agent is None.
        """
        if agent is None:
            raise ValueError("Agent cannot be None")
        self.agent = agent
        
    async def before_step(self, step_number: int, action: ActionModel) -> None:
        """Hook called before each step.
        
        Args:
            step_number: The current step number.
            action: The action model for the current step.
            
        Raises:
            ValueError: If step_number is negative or action is None.
        """
        if step_number < 0:
            raise ValueError(f"Invalid step number: {step_number}")
        if action is None:
            raise ValueError("Action cannot be None")
            
        logger.debug(f"Before step {step_number}")
        # Add any pre-step validation or setup here
        
    async def after_step(self, step_number: int, action: ActionModel, result: ActionResult) -> None:
        """Hook called after each step.
        
        Args:
            step_number: The current step number.
            action: The action model for the current step.
            result: The result of the step execution.
            
        Raises:
            ValueError: If any parameter is None or step_number is negative.
        """
        if step_number < 0:
            raise ValueError(f"Invalid step number: {step_number}")
        if action is None:
            raise ValueError("Action cannot be None")
        if result is None:
            raise ValueError("Result cannot be None")
            
        logger.debug(f"After step {step_number}")
        
        # If step failed, try to verify the failure
        if not result.success and hasattr(self.agent, 'browser_session'):
            try:
                # Create verification action
                verify_action = ActionModel(
                    verify={
                        "goal": f"Verify failure of step {step_number}",
                        "should_strip_link_urls": True
                    }
                )
                
                # Use browser session for verification
                if not hasattr(self.agent, 'controller'):
                    logger.error("Agent does not have controller attribute")
                    return
                    
                result = await self.agent.controller.act(verify_action, self.agent.browser_session)
                
                if result.success:
                    logger.info(f"Verified failure of step {step_number}")
                else:
                    logger.warning(f"Failed to verify step {step_number} failure: {result.error}")
                    
            except Exception as e:
                logger.error(f"Error in after_step hook: {str(e)}", exc_info=True)
                
    async def on_error(self, step_number: int, action: ActionModel, error: Exception) -> None:
        """Hook called when a step fails.
        
        Args:
            step_number: The current step number.
            action: The action model for the current step.
            error: The exception that caused the failure.
            
        Raises:
            ValueError: If any parameter is None or step_number is negative.
        """
        if step_number < 0:
            raise ValueError(f"Invalid step number: {step_number}")
        if action is None:
            raise ValueError("Action cannot be None")
        if error is None:
            raise ValueError("Error cannot be None")
            
        logger.error(f"Error in step {step_number}: {str(error)}", exc_info=True)
        
    async def on_complete(self, success: bool) -> None:
        """Hook called when all steps are complete.
        
        Args:
            success: Whether the test completed successfully.
        """
        logger.info(f"Test {'succeeded' if success else 'failed'}")
        # Add any cleanup or final reporting here