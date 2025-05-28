import pytest
from browser_use.controller.views import (
    SearchGoogleAction,
    GoToUrlAction,
    ClickElementAction,
    InputTextAction,
    ScrollAction,
    SendKeysAction,
    SwitchTabAction,
    OpenTabAction,
    CloseTabAction,
    DragDropAction,
    DoneAction,
    Position,
)

def test_search_google_action_validation():
    """Test validation of SearchGoogleAction parameters."""
    # Test valid query
    action = SearchGoogleAction(query="test query")
    assert action.query == "test query"

    # Test empty query
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        SearchGoogleAction(query="")

    # Test whitespace-only query
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        SearchGoogleAction(query="   ")

    # Test query that's too long
    with pytest.raises(ValueError, match="Search query is too long"):
        SearchGoogleAction(query="x" * 501)

def test_go_to_url_action_validation():
    """Test validation of GoToUrlAction parameters."""
    # Test valid URL
    action = GoToUrlAction(url="https://example.com")
    assert action.url == "https://example.com"

    # Test invalid URL
    with pytest.raises(ValueError, match="URL must start with http:// or https://"):
        GoToUrlAction(url="example.com")

def test_click_element_action_validation():
    """Test validation of ClickElementAction parameters."""
    # Test valid index
    action = ClickElementAction(index=0)
    assert action.index == 0

    # Test negative index
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
        ClickElementAction(index=-1)

def test_input_text_action_validation():
    """Test validation of InputTextAction parameters."""
    # Test valid input
    action = InputTextAction(index=0, text="test input")
    assert action.index == 0
    assert action.text == "test input"

    # Test empty text
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        InputTextAction(index=0, text="")

    # Test whitespace-only text
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        InputTextAction(index=0, text="   ")

    # Test negative index
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
        InputTextAction(index=-1, text="test")

def test_scroll_action_validation():
    """Test validation of ScrollAction parameters."""
    # Test with amount
    action = ScrollAction(amount=100)
    assert action.amount == 100

    # Test without amount
    action = ScrollAction()
    assert action.amount is None

def test_send_keys_action_validation():
    """Test validation of SendKeysAction parameters."""
    # Test valid keys
    action = SendKeysAction(keys="Enter")
    assert action.keys == "Enter"

    # Test empty keys
    with pytest.raises(ValueError, match="Keys string cannot be empty"):
        SendKeysAction(keys="")

    # Test whitespace-only keys
    with pytest.raises(ValueError, match="Keys string cannot be empty"):
        SendKeysAction(keys="   ")

def test_switch_tab_action_validation():
    """Test validation of SwitchTabAction parameters."""
    # Test valid page_id
    action = SwitchTabAction(page_id=1)
    assert action.page_id == 1

def test_open_tab_action_validation():
    """Test validation of OpenTabAction parameters."""
    # Test valid URL
    action = OpenTabAction(url="https://example.com")
    assert action.url == "https://example.com"

    # Test invalid URL
    with pytest.raises(ValueError, match="URL must start with http:// or https://"):
        OpenTabAction(url="example.com")

def test_close_tab_action_validation():
    """Test validation of CloseTabAction parameters."""
    # Test valid page_id
    action = CloseTabAction(page_id=1)
    assert action.page_id == 1

def test_drag_drop_action_validation():
    """Test validation of DragDropAction parameters."""
    # Test valid parameters
    action = DragDropAction(
        source_selector="#source",
        target_selector="#target",
        source_position=Position.CENTER,
        target_position=Position.CENTER,
        steps=10,
        delay_ms=50
    )
    assert action.source_selector == "#source"
    assert action.target_selector == "#target"
    assert action.source_position == Position.CENTER
    assert action.target_position == Position.CENTER
    assert action.steps == 10
    assert action.delay_ms == 50

    # Test invalid steps
    with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
        DragDropAction(
            source_selector="#source",
            target_selector="#target",
            steps=0
        )

    # Test invalid delay
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
        DragDropAction(
            source_selector="#source",
            target_selector="#target",
            delay_ms=-1
        )

def test_done_action_validation():
    """Test validation of DoneAction parameters."""
    # Test valid parameters
    action = DoneAction(success=True, text="Task completed")
    assert action.success is True
    assert action.text == "Task completed"

    # Test with failure
    action = DoneAction(success=False, text="Task failed")
    assert action.success is False
    assert action.text == "Task failed"

def test_position_enum():
    """Test Position enum values."""
    assert Position.TOP == "top"
    assert Position.BOTTOM == "bottom"
    assert Position.LEFT == "left"
    assert Position.RIGHT == "right"
    assert Position.CENTER == "center"

    # Test invalid position
    with pytest.raises(ValueError):
        Position("invalid") 