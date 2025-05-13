class WorkflowRunner:
    def __init__(self, agent):
        self.agent = agent

    async def execute_workflow(self):
        await self.agent.run()
