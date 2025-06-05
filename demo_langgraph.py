import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from langchain import LLMChain, PromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Setup SerpAPI Search Tool
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
search_tool = Tool(name="Search", func=search.run, description="Web search tool")

# ------------------------ AGENT FUNCTIONS ------------------------

def researcher_node(state, config):
    query = state.get("input")
    print("[researcher_node] input =", query)
    result = search_tool.run(f"Find the latest info on: {query}. Then summarize it.")
    new_state = dict(state)
    new_state["summary"] = result
    return new_state

def extractor_node(state, config):
    summary = state.get("summary")
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
Extract the most relevant keywords or key phrases from the following text.
Return exactly two important terms as a Python list of strings.

Text:
\"\"\"{text}\"\"\"
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    keywords_text = chain.run(summary)

    try:
        keywords = eval(keywords_text)
    except:
        keywords = [keywords_text.strip()]

    new_state = dict(state)
    new_state["keywords"] = keywords
    return new_state

def verifier_node(state, config):
    keywords = state.get("keywords", [])
    explanations = []
    for kw in keywords:
        explanation = search_tool.run(f"Search '{kw}' and explain what it means in the context of AI or recent news.")
        explanations.append((kw, explanation[:300]))

    new_state = dict(state)
    new_state["explanations"] = explanations
    return new_state

def output_node(state, config):
    print("\n--- Research Summary ---")
    print(state.get("summary"))
    print("\n--- Extracted Keywords ---")
    print(state.get("keywords"))
    print("\n--- Keyword Explanations ---")
    for kw, expl in state.get("explanations", []):
        print(f"\n🔑 {kw}:\n{expl}")
    return state

# ------------------------ LANGGRAPH WORKFLOW ------------------------

graph = StateGraph(dict)

graph.add_node("researcher", researcher_node)
graph.add_node("extractor", extractor_node)
graph.add_node("verifier", verifier_node)
graph.add_node("output", output_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "extractor")
graph.add_edge("extractor", "verifier")
graph.add_edge("verifier", "output")
graph.add_edge("output", END)

app = graph.compile()

# ------------------------ RUN ------------------------

user_input = input("Enter a research topic: ").strip()
initial_state = {"input": user_input}
app.invoke(initial_state)
