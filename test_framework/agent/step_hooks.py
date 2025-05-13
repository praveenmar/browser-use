import logging
from patchright.async_api import expect
from test_framework.validation.assertions import TestAssertions
from browser_use.agent.views import ActionModel

logger = logging.getLogger("test_framework.agent.step_hooks")

async def step_end_hook(agent):
    """
    Hook to be called after each agent step.
    Receives the agent instance, can access state, last action/result, etc.
    """
    try:
        # Get the task description
        task_description = agent.task
        if not task_description:
            return

        # Extract verification steps
        verification_steps = extract_verification_steps(task_description)
        if not verification_steps:
            return

        # Get the last action
        last_action = agent.state.last_action
        if not last_action:
            return

        # For each verification step, check if we should verify now
        for step in verification_steps:
            if _should_verify_now(last_action, step):
                logger.info(f"[Step Hook] 🔍 Verifying at step {agent.state.n_steps}: {step}")
                
                # Extract text to verify
                text_to_verify = None
                if "'" in step:
                    parts = step.split("'")
                    if len(parts) >= 2:
                        text_to_verify = parts[1]
                elif '"' in step:
                    parts = step.split('"')
                    if len(parts) >= 2:
                        text_to_verify = parts[1]
                
                if not text_to_verify:
                    logger.error(f"[Step Hook] ❌ Could not extract text to verify from step: {step}")
                    continue

                # Create verify text action
                verify_action = ActionModel(**{
                    "verify_text": {
                        "text": text_to_verify,
                        "exact": True,  # Use exact match for verification steps
                        "timeout": 5000  # 5 second timeout
                    }
                })

                # Execute the verification
                try:
                    result = await agent.controller.act(verify_action, agent.browser_context)
                    if result.error:
                        logger.error(f"[Step Hook] ❌ Verification failed at step {agent.state.n_steps}: {step} - {result.error}")
                        raise AssertionError(f"Verification failed: {result.error}")
                    logger.info(f"[Step Hook] ✅ Verification passed at step {agent.state.n_steps}: {step}")
                except Exception as e:
                    logger.error(f"[Step Hook] ❌ Error in step_end_hook: {str(e)}")
                    raise

    except Exception as e:
        logger.error(f"[Step Hook] Error in step_end_hook: {str(e)}")
        raise

def extract_verification_steps(current_step: str) -> list:
    """Extract verification steps from the current step text"""
    verification_steps = []
    
    # Split the step into lines
    lines = current_step.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines that start with "verify" or contain "verify the text"
        if line.lower().startswith('verify') or 'verify the text' in line.lower():
            verification_steps.append(line)
    
    return verification_steps

def _should_verify_now(last_action, verification_step: str) -> bool:
    """Determine if a verification step should be executed now based on the last action"""
    # Get the action name from the last action
    action_name = None
    if isinstance(last_action, dict):
        action_name = next(iter(last_action.keys()), None)
    elif hasattr(last_action, '__dict__'):
        action_name = next(iter(last_action.__dict__.keys()), None)
    
    if not action_name:
        return False
        
    # List of actions that should trigger immediate verification
    immediate_verify_actions = [
        'go_to_url',
        'click_element',
        'input_text',
        'submit_form',
        'select_option',
        'check_checkbox',
        'uncheck_checkbox',
        'press_key',
        'hover',
        'scroll',
        'wait',
        'refresh'
    ]
    
    # If the last action is in our immediate verify list, verify now
    if action_name in immediate_verify_actions:
        return True
        
    # For other actions, check if the verification step matches the action
    verification_type = verification_step.lower()
    if 'text' in verification_type or 'content' in verification_type:
        return True
    elif 'link' in verification_type or 'href' in verification_type or 'url' in verification_type:
        return True
        
    return False

def _get_action_type(last_action) -> str:
    """
    Extract the action type from the last action
    """
    if hasattr(last_action, 'action') and last_action.action:
        first_action = last_action.action[0] if last_action.action else None
        if first_action:
            return next(iter(first_action.__dict__.keys()), None)
    return None