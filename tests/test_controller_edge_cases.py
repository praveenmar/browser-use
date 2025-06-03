import pytest
import pytest_asyncio
from typing import Optional, Dict, List
from pydantic import BaseModel
from browser_use.controller.service import Controller
from browser_use.browser.session import BrowserSession
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

@pytest_asyncio.fixture
async def controller():
    """Create a controller instance"""
    return Controller()

@pytest_asyncio.fixture
async def browser_session():
    """Create a mock browser session"""
    session = Mock(spec=BrowserSession)
    session.get_current_page = AsyncMock()
    return session

@pytest.mark.asyncio
async def test_multi_step_assertion_chain(controller, browser_session):
    """Test chaining multiple assertions"""
    # Create search action
    search_action = ActionModel(**{
        "search": {
            "query": "test query"
        }
    })
    
    # Execute first action
    result1 = await controller.act(search_action, browser_session)
    assert result1.success
    
    # Create input action
    input_action = ActionModel(**{
        "input": {
            "selector": "#search-input",
            "text": "test text"
        }
    })
    
    # Execute second action
    result2 = await controller.act(input_action, browser_session)
    assert result2.success
    
    # Create click action
    click_action = ActionModel(**{
        "click": {
            "selector": "#submit-button"
        }
    })
    
    # Execute third action
    result3 = await controller.act(click_action, browser_session)
    assert result3.success

@pytest.mark.asyncio
async def test_error_handling_and_recovery(controller, browser_session):
    """Test error handling and recovery"""
    # Create search action
    search_action = ActionModel(**{
        "search": {
            "query": "test query"
        }
    })
    
    # Simulate network error
    browser_session.get_current_page.side_effect = Exception("Network error")
    
    # Execute action with error
    result1 = await controller.act(search_action, browser_session)
    assert not result1.success
    assert "Network error" in result1.error
    
    # Reset error and retry
    browser_session.get_current_page.side_effect = None
    result2 = await controller.act(search_action, browser_session)
    assert result2.success

@pytest.mark.asyncio
async def test_sensitive_data_handling(controller, browser_session):
    """Test handling of sensitive data"""
    # Create action with sensitive data
    sensitive_action = ActionModel(**{
        "input": {
            "selector": "#password-input",
            "text": "secret123",
            "sensitive": True
        }
    })
    
    # Execute action
    result = await controller.act(
        sensitive_action,
        browser_session
    )
    
    # Verify sensitive data is not logged
    assert result.success
    assert "secret123" not in str(result)

@pytest.mark.asyncio
async def test_output_model_serialization(controller, browser_session):
    """Test serialization of output models"""
    # Create controller with custom output model
    class CustomOutputModel(ActionModel):
        custom_field: str = "test"
        
    controller_with_model = Controller(output_model=CustomOutputModel)
    
    # Create done action
    done_action = ActionModel(**{
        "done": {
            "message": "Test complete"
        }
    })
    
    # Execute action
    result = await controller_with_model.act(done_action, browser_session)
    
    # Verify output model
    assert result.success
    assert isinstance(result, CustomOutputModel)
    assert result.custom_field == "test"

@pytest.mark.asyncio
async def test_concurrent_action_handling(controller, browser_session):
    """Test handling of concurrent actions"""
    import asyncio
    
    # Create action
    action = ActionModel(**{
        "click": {
            "selector": "#test-button"
        }
    })
    
    # Define action executor
    async def execute_action():
        return await controller.act(action, browser_session)
    
    # Execute actions concurrently
    results = await asyncio.gather(
        execute_action(),
        execute_action(),
        execute_action()
    )
    
    # Verify all actions succeeded
    assert all(result.success for result in results)

@pytest.mark.asyncio
async def test_action_validation(controller, browser_session):
    """Test action validation"""
    # Create invalid action
    invalid_action = ActionModel(**{
        "click": {
            "invalid_param": "test"
        }
    })
    
    # Execute invalid action
    result = await controller.act(invalid_action, browser_session)
    assert not result.success
    assert "validation" in result.error.lower()
    
    # Create valid action
    valid_action = ActionModel(**{
        "click": {
            "selector": "#test-button"
        }
    })
    
    # Execute valid action
    result = await controller.act(valid_action, browser_session)
    assert result.success

@pytest.mark.asyncio
async def test_browser_state_persistence(controller, browser_session):
    """Test browser state persistence between actions"""
    # Create search action
    search_action = ActionModel(**{
        "search": {
            "query": "test query"
        }
    })
    
    # Execute first action
    await controller.act(search_action, browser_session)
    
    # Create input action
    input_action = ActionModel(**{
        "input": {
            "selector": "#search-input",
            "text": "test text"
        }
    })
    
    # Execute second action
    result = await controller.act(input_action, browser_session)
    
    # Verify browser state was maintained
    assert result.success
    browser_session.get_current_page.assert_called()
    assert browser_session.get_current_page.call_count == 2 