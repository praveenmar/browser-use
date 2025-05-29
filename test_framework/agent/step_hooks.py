"""
Step hooks for the test framework.
"""

import logging
from typing import Optional, Dict, Any
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.session import BrowserSession

logger = logging.getLogger("test_framework.agent.step_hooks")

class StepHooks:
    """Hooks for test steps"""
    
    def __init__(self, agent):
        """Initialize step hooks"""
        self.agent = agent
        
    async def before_step(self, step_number: int, action: ActionModel) -> None:
        """Hook called before each step"""
        logger.debug(f"Before step {step_number}")
        
    async def after_step(self, step_number: int, action: ActionModel, result: ActionResult) -> None:
        """Hook called after each step"""
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
                result = await self.agent.controller.act(verify_action, self.agent.browser_session)
                
                if result.success:
                    logger.info(f"Verified failure of step {step_number}")
                else:
                    logger.warning(f"Failed to verify step {step_number} failure: {result.error}")
                    
            except Exception as e:
                logger.error(f"Error in after_step hook: {str(e)}")
                
    async def on_error(self, step_number: int, action: ActionModel, error: Exception) -> None:
        """Hook called when a step fails"""
        logger.error(f"Error in step {step_number}: {str(error)}")
        
    async def on_complete(self, success: bool) -> None:
        """Hook called when all steps are complete"""
        logger.info(f"Test {'succeeded' if success else 'failed'}")