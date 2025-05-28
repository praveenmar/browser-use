import pytest
from typing import Optional, Dict, List
from pydantic import BaseModel
from browser_use.controller.service import Controller
from browser_use.browser.context import BrowserContext
from browser_use.agent.views import ActionModel, ActionResult
from browser_use.controller.views import (
    ClickElementAction,
    InputTextAction,
    SearchGoogleAction,
    DoneAction,
)

class TestOutputModel(BaseModel):
    """Test output model for controller tests."""
    value: str
    count: int

@pytest.fixture
def controller():
    """Create a controller instance for testing."""
    return Controller[Dict]()

@pytest.fixture
def browser_context(mocker):
    """Create a mock browser context."""
    context = mocker.Mock(spec=BrowserContext)
    context.get_current_page.return_value = mocker.AsyncMock()
    return context

@pytest.mark.asyncio
async def test_multi_step_assertion_chain(controller, browser_context):
    """Test a chain of assertions that depend on previous steps."""
    # Step 1: Search Google
    search_action = ActionModel(
        name="search_google",
        params=SearchGoogleAction(query="test query")
    )
    result1 = await controller.act(search_action, browser_context)
    assert result1.success
    assert "test query" in result1.extracted_content

    # Step 2: Input text (depends on search results)
    input_action = ActionModel(
        name="input_text",
        params=InputTextAction(index=1, text="test input")
    )
    result2 = await controller.act(input_action, browser_context)
    assert result2.success
    assert "test input" in result2.extracted_content

    # Step 3: Click element (depends on input)
    click_action = ActionModel(
        name="click_element_by_index",
        params=ClickElementAction(index=2)
    )
    result3 = await controller.act(click_action, browser_context)
    assert result3.success

@pytest.mark.asyncio
async def test_error_handling_and_recovery(controller, browser_context):
    """Test error handling and recovery in action chain."""
    # Step 1: Trigger an error
    browser_context.get_current_page.side_effect = Exception("Network error")
    
    search_action = ActionModel(
        name="search_google",
        params=SearchGoogleAction(query="test query")
    )
    result1 = await controller.act(search_action, browser_context)
    assert not result1.success
    assert "Network error" in result1.error

    # Step 2: Recover and try again
    browser_context.get_current_page.side_effect = None
    result2 = await controller.act(search_action, browser_context)
    assert result2.success

@pytest.mark.asyncio
async def test_sensitive_data_handling(controller, browser_context):
    """Test handling of sensitive data in actions."""
    sensitive_data = {"password": "secret123"}
    
    input_action = ActionModel(
        name="input_text",
        params=InputTextAction(index=1, text="secret123")
    )
    result = await controller.act(
        input_action,
        browser_context,
        sensitive_data=sensitive_data
    )
    assert result.success
    assert "sensitive data" in result.extracted_content
    assert "secret123" not in result.extracted_content

@pytest.mark.asyncio
async def test_output_model_serialization(controller, browser_context):
    """Test serialization of complex output models."""
    controller_with_model = Controller[Dict](output_model=TestOutputModel)
    
    done_action = ActionModel(
        name="done",
        params=DoneAction(
            success=True,
            text='{"value": "test", "count": 42}'
        )
    )
    result = await controller_with_model.act(done_action, browser_context)
    assert result.success
    assert result.is_done
    
    # Verify the output is properly serialized
    output_data = TestOutputModel.model_validate_json(result.extracted_content)
    assert output_data.value == "test"
    assert output_data.count == 42

@pytest.mark.asyncio
async def test_concurrent_action_handling(controller, browser_context):
    """Test handling of concurrent actions."""
    import asyncio
    
    async def execute_action(action: ActionModel) -> ActionResult:
        return await controller.act(action, browser_context)
    
    # Create multiple actions to execute concurrently
    actions = [
        ActionModel(
            name="search_google",
            params=SearchGoogleAction(query=f"test query {i}")
        )
        for i in range(3)
    ]
    
    # Execute actions concurrently
    results = await asyncio.gather(
        *[execute_action(action) for action in actions]
    )
    
    # Verify all actions completed successfully
    assert all(result.success for result in results)
    assert len(results) == 3

@pytest.mark.asyncio
async def test_action_validation(controller, browser_context):
    """Test validation of action parameters."""
    # Test invalid index
    input_action = ActionModel(
        name="input_text",
        params=InputTextAction(index=-1, text="test")
    )
    result = await controller.act(input_action, browser_context)
    assert not result.success
    assert "index" in result.error.lower()

    # Test empty text
    input_action = ActionModel(
        name="input_text",
        params=InputTextAction(index=1, text="")
    )
    result = await controller.act(input_action, browser_context)
    assert not result.success
    assert "text" in result.error.lower()

@pytest.mark.asyncio
async def test_browser_state_persistence(controller, browser_context):
    """Test persistence of browser state between actions."""
    # Step 1: Navigate to a page
    search_action = ActionModel(
        name="search_google",
        params=SearchGoogleAction(query="initial state")
    )
    await controller.act(search_action, browser_context)
    
    # Step 2: Verify state is maintained
    input_action = ActionModel(
        name="input_text",
        params=InputTextAction(index=1, text="verify state")
    )
    result = await controller.act(input_action, browser_context)
    assert result.success
    
    # Verify browser context methods were called in correct order
    browser_context.get_current_page.assert_called()
    assert browser_context.get_current_page.call_count == 2 