from browser_use.browser.browser import Browser, BrowserConfig

class BrowserManager:
    def __init__(self, config: dict):
        self.config = config
        self.browser = None

    async def launch_browser(self):
        self.browser = Browser(config=BrowserConfig(headless=self.config.get("headless", False)))
        await self.browser.get_playwright_browser()  # ✅ Correct way to initialize the browser

    async def execute_action(self, action: dict):
        if action.get("type") == "click":
            return await self.browser.click(action.get("selector"))
        elif action.get("type") == "type":
            return await self.browser.type(action.get("selector"), action.get("text"))
        elif action.get("type") == "navigate":
            return await self.browser.goto(action.get("url"))
        else:
            return {"status": "unknown action"}

    async def close_browser(self):
        if self.browser:
            await self.browser.close()
