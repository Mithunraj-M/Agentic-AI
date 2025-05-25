import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain import LLMChain
from langchain.prompts import PromptTemplate


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

tools_agent1 = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching recent information on the web."
    )
]

# Agent setup
agent = initialize_agent(
    tools=tools_agent1,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Run the agent
user_input = input("Enter a research topic: ")
response = agent.run(f"Find the latest info on: {user_input}. Then summarize it concisely.")

print("\nResearch Summary:\n", response)

# Agent 2: Takes summary and formats it for learning
keyword_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Extract the most relevant keywords or key phrases from the following text.
Return exactly two important terms as a Python list of strings.

Text:
\"\"\"{text}\"\"\"
"""
)

keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt)
keywords_text = keyword_chain.run(response)

# Evaluate and clean keyword list
try:
    keywords = eval(keywords_text)  # Convert string list to Python list
except Exception:
    keywords = [keywords_text.strip()]  # Fallback in case of malformed output


print("\n Extracted Keywords by Agent 2:\n", keywords)


tools_agent2=[
    Tool(
        name="WebVerifier",
        func=search.run,
        description="Use this tool to verify or explain individual keywords on the web."
    )
]
agent2 = initialize_agent(
    tools=tools_agent2,  # No tools needed, just processing
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

print("\n Cross-Verification by Agent 2:\n")
for kw in keywords:
    print(f"Verifying keyword: {kw}")
    result = agent2.run(f"Search '{kw}' and explain what it means in the context of AI or recent news.")
    print(result[:200])  # Limit output for readability
    print("-" * 80)