from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.genai import types

from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import WikipediaQueryRun # A third-party LangChain tool
from langchain_community.utilities import WikipediaAPIWrapper # Utility for Wikipedia tool

GEMINI_MODEL = "gemini-2.0-flash"

# --- Agent 2: Fact Finder Agent (Uses a Built-in Tool: Google Search) ---
# This agent answers general knowledge questions by searching the web.

fact_finder_agent = LlmAgent(
    name="FactFinderAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are a knowledgeable fact-finding assistant. "
        "Use the Google Search tool to find answers to factual questions. "
        "Summarize the information concisely and provide the answer."
    ),
    description="Answers factual questions by performing Google searches.",
    tools=[google_search], # Uses the built-in Google Search tool
    generate_content_config=types.GenerateContentConfig(temperature=0.3) # Balanced for factual summary
)

# --- Agent 3: Wikipedia Summarizer Agent (Uses a Third-Party LangChain Tool) ---
# This agent uses LangChain's Wikipedia tool to fetch and summarize information.

# Initialize the Wikipedia tool from LangChain
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

adk_wikipedia_tool = LangchainTool(tool=wikipedia_tool)

wikipedia_summarizer_agent = LlmAgent(
    name="WikipediaSummarizerAgent",
    model=GEMINI_MODEL,
    instruction=(
        "You are an academic summarizer. "
        "Use the Wikipedia tool to find information on the given topic and then provide a concise summary of the key points. "
        "Focus on factual information and avoid speculation."
    ),
    description="Fetches and summarizes information from Wikipedia using a third-party LangChain tool.",
    tools=[adk_wikipedia_tool], # Uses the third-party Wikipedia tool
    generate_content_config=types.GenerateContentConfig(temperature=0.2) # More factual, less creative
)

root_agent = fact_finder_agent