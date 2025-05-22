import asyncio
import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

# Initialize the model
llm = ChatGoogleGenerativeAI(
	model='gemini-2.0-flash-exp'
)
task = 'Go to kayak.com and find the cheapest one-way flight from Zurich to San Francisco in 3 weeks.'
agent = Agent(task=task, llm=llm)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
