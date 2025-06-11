import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --- Load .env ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

# --- Define the App State ---
class AppState(TypedDict):
    user_input: str
    tool_used: str
    output: str

# --- Real Search Tool (SerpAPI) ---
search = SerpAPIWrapper()

def search_tool(query: str) -> str:
    return search.run(query)

# --- Simple Tools ---
def calculator_tool(input_str: str) -> str:
    try:
        return str(eval(input_str))
    except:
        return "Invalid math expression."

def summarizer_tool(input_str: str) -> str:
    return f"(Summary) {input_str[:50]}..."

# --- Tool Registry ---
tools = {
    "calculator": calculator_tool,
    "summarizer": summarizer_tool,
    "search": search_tool
}

# --- Use OpenRouter LLM via LangChain ---
llm = ChatOpenAI(
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="gpt-3.5-turbo" ,
     max_tokens=200 # You can change model here
)

# --- LLM Tool Chooser Node ---
def llm_tool_chooser(state: AppState):
    user_input = state["user_input"]
    prompt = f"""You are a smart tool-routing agent.

Choose one of the following tools based on the user input:

- calculator → for math expressions like '12 * 4 + 3'
- summarizer → for summarizing long paragraphs or text
- search → for questions that require real-world knowledge or recent facts

Input: "{user_input}"
Tool:"""
    
    tool = llm.predict(prompt).strip().lower()
    return {"tool_used": tool}

# --- Tool Executor Node ---
def execute_tool(state: AppState):
    tool = state["tool_used"]
    user_input = state["user_input"]
    result = tools.get(tool, lambda x: "Tool not found.")(user_input)
    return {"output": result}

# --- LangGraph Construction ---
builder = StateGraph(AppState)

builder.add_node("chooser", llm_tool_chooser)
builder.add_node("executor", execute_tool)

builder.set_entry_point("chooser")
builder.add_edge("chooser", "executor")
builder.add_edge("executor", END)

graph = builder.compile()

# --- Test Cases ---
if __name__ == "__main__":
    examples = [
        " 25 * (8 - 3)",
        "Summarize this: Artificial intelligence is a branch of computer science...",
        "Who is the CEO of Microsoft?"
    ]

    for q in examples:
        print(f"\n📩 Input: {q}")
        response = graph.invoke({"user_input": q})
        print("🔧 Tool used:", response["tool_used"])
        print("📤 Output:", response["output"])
