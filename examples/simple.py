import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(
	model='gemini-2.0-flash-exp'
)
task = 'Go to kayak.com and find the cheapest flight from Zurich to San Francisco on 2025-05-01'

agent = Agent(task=task, llm=llm)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
