import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

# 🔐 Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

# 🧠 LLM via OpenRouter
llm = ChatOpenAI(
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/mistral-7b-instruct",
    max_tokens=300
)

# 🌐 Search Tool using SerpAPI
search_tool = Tool.from_function(
    func=SerpAPIWrapper().run,
    name="search",
    description="Useful for answering questions about current events or general knowledge."
)

# 🧮 Calculator Tool
def calculate(input_str: str) -> str:
    try:
        return str(eval(input_str))
    except Exception as e:
        return f"Error: {e}"

calculator_tool = Tool.from_function(
    func=calculate,
    name="calculator",
    description="Evaluates basic math expressions like '3 * (4 + 2)'"
)

# 📝 Summarizer Tool
def summarize(text: str) -> str:
    return f"(Summary) {text[:50]}..."

summarizer_tool = Tool.from_function(
    func=summarize,
    name="summarizer",
    description="Summarizes long text into a short summary."
)

# 🔧 Tool List
tool_list = [search_tool, calculator_tool, summarizer_tool]

# 🧾 Define App State
class AppState(TypedDict):
    user_input: str
    tool_used: str
    output: str

# 🧠 Agent setup
agent = initialize_agent(
    tools=tool_list,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True  # Needed to track tool used
)

# 🧠 Agent node function
def agent_runner(state: AppState):
    user_input = state["user_input"]
    result = agent.invoke({"input": user_input})

    # Extract tool used from intermediate steps
    intermediate_steps = result.get("intermediate_steps", [])
    tool_used = intermediate_steps[0][0].tool if intermediate_steps else "unknown"
    
    return {
        "tool_used": tool_used,
        "output": result["output"]
    }

# 🧬 LangGraph setup
builder = StateGraph(AppState)
builder.add_node("agent_runner", agent_runner)
builder.set_entry_point("agent_runner")
builder.add_edge("agent_runner", END)
graph = builder.compile()

# 🧪 Test Run
if __name__ == "__main__":
    inputs = [
        "What's 45 * 3?",
        "Summarize this: The quick brown fox jumps over the lazy dog.",
        "Who is the president of the United States?"
    ]
    for query in inputs:
        print(f"\n📨 Input: {query}")
        res = graph.invoke({"user_input": query})
        print("🔧 Tool used:", res["tool_used"])
        print("📤 Output:", res["output"])
