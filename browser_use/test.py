from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from dotenv import load_dotenv

# Read GOOGLE_API_KEY into env
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)