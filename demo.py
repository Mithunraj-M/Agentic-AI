import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_types import AgentType

# Load .env variables
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not openrouter_api_key:
    raise ValueError("Please set OPENROUTER_API_KEY in your .env file.")
if not serpapi_api_key:
    raise ValueError("Please set SERPAPI_API_KEY in your .env file.")

# Use OpenRouter via OpenAI-compatible endpoint
os.environ["OPENAI_API_KEY"] = openrouter_api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"  # <-- corrected here

# Set up the model (use the correct model ID)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # corrected model name
    temperature=0.7,
    max_tokens=200
)

# Search tool via SerpAPI
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching recent information on the web."
    )
]

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Run the agent
user_input = input("Enter a research topic: ")
response = agent.run(f"Find the latest info on: {user_input}. Then summarize it concisely.")

print("\nResearch Summary:\n", response)
