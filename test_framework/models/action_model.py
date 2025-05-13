from pydantic import BaseModel
from browser_use.agent.views import ActionModel as BrowserUseActionModel

class ActionModel(BrowserUseActionModel):
    """Extended ActionModel for test framework specific actions"""
    pass 