import logging
from patchright.async_api import expect, Page, Locator, TimeoutError as PlaywrightTimeoutError
from test_framework.validation.assertion_module import AssertionModule, AssertionType, VerificationContext
from pydantic import BaseModel
from browser_use.browser.context import BrowserContext
from browser_use.agent.views import ActionResult

logger = logging.getLogger("test_framework.validation.assertions")

class VerifyTextAction(BaseModel):
    """Model for text verification action"""
    text: str
    exact: bool = False
    timeout: int = 5000  # 5 seconds default timeout

class VerifyLinkAction(BaseModel):
    """Model for hyperlink verification action"""
    text: str  # The link text to find
    href: str  # The expected href/URL
    exact: bool = False
    timeout: int = 5000  # 5 seconds default timeout

class TestAssertions:
    def __init__(self, agent, history):
        self.agent = agent
        self.history = history
        self.page = None
        self.assertion_module = AssertionModule()

    def _extract_verification_steps(self, task_description: str) -> list:
        """Extract verification steps from the task description"""
        verification_keywords = ["verify", "validate", "check", "assert", "ensure", "confirm"]
        sentences = [s.strip() for s in task_description.replace(' and ', '. ').split('.')]
        return [s for s in sentences if any(k in s for k in verification_keywords)]

    async def _verify_condition(self, verification_step: str):
        """Verify a single condition using browser-use's validation system"""
        try:
            # Get the current page when needed
            if not self.page:
                self.page = await self.agent.browser_context.get_current_page()
                
            logger.info(f"Original verification step: '{verification_step}'")

            # Extract the text to verify from the step
            text_to_verify = None
            if "'" in verification_step:
                parts = verification_step.split("'")
                if len(parts) >= 2:
                    text_to_verify = parts[1]
            elif '"' in verification_step:
                parts = verification_step.split('"')
                if len(parts) >= 2:
                    text_to_verify = parts[1]
            
            if not text_to_verify:
                raise ValueError("Could not extract text to verify from step")
            
            logger.info(f"Expected text to verify: '{text_to_verify}'")
            
            # Create verification context
            context = VerificationContext(
                step_number=self.agent.state.n_steps if hasattr(self.agent.state, 'n_steps') else 0,
                action_type="verification",
                verification_step=verification_step,
                expected_value=text_to_verify,
                actual_value=None  # Will be set by the assertion module
            )

            # Use browser-use's assertion module to verify
            result = await self.assertion_module.verify_step(context)
            
            if not result:
                raise AssertionError(f"Verification failed for step: '{verification_step}'")

        except Exception as e:
            error_msg = f"Assertion failed for step '{verification_step}': {str(e)}"
            logger.error(error_msg)
            raise AssertionError(error_msg)

    @staticmethod
    def register_actions(controller):
        """Register assertion actions with the controller"""
        
        @controller.action(
            'Verify text is present on the page',
            param_model=VerifyTextAction
        )
        async def verify_text(params: VerifyTextAction, browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                # Use patchwright's expect functionality
                if params.exact:
                    await expect(page.get_by_text(params.text, exact=True)).to_be_visible(timeout=params.timeout)
                else:
                    await expect(page.get_by_text(params.text)).to_be_visible(timeout=params.timeout)
                
                msg = f'✅ Verified text "{params.text}" is present on the page'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except PlaywrightTimeoutError:
                error_msg = f'❌ Text "{params.text}" not found on the page'
                logger.error(error_msg)
                return ActionResult(error=error_msg)
            except Exception as e:
                error_msg = f'❌ Error verifying text: {str(e)}'
                logger.error(error_msg)
                return ActionResult(error=error_msg)

        @controller.action(
            'Verify hyperlink is present on the page',
            param_model=VerifyLinkAction
        )
        async def verify_link(params: VerifyLinkAction, browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                # Find the link by text
                link = page.get_by_role("link", name=params.text, exact=params.exact)
                
                # Verify the link is visible
                await expect(link).to_be_visible(timeout=params.timeout)
                
                # Get the href attribute
                href = await link.get_attribute("href")
                
                # Verify the href matches
                if href != params.href:
                    error_msg = f'❌ Link href mismatch. Expected: {params.href}, Got: {href}'
                    logger.error(error_msg)
                    return ActionResult(error=error_msg)
                
                msg = f'✅ Verified link "{params.text}" with href "{params.href}" is present on the page'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except PlaywrightTimeoutError:
                error_msg = f'❌ Link "{params.text}" not found on the page'
                logger.error(error_msg)
                return ActionResult(error=error_msg)
            except Exception as e:
                error_msg = f'❌ Error verifying link: {str(e)}'
                logger.error(error_msg)
                return ActionResult(error=error_msg)