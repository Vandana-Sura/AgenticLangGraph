from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

import os
from dotenv import load_dotenv

load_dotenv()

# Set environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "TestProject"

from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama3-8b-8192")

# Define the graph state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_tool_graph():
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    tools = [add]

    # Bind tools to the model
    llm_with_tool = llm.bind_tools(tools)

    # Automatically prepend a system message
    def call_llm_model(state: State):
        messages = state["messages"]

        # Insert system prompt (only once at the beginning)
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [
                SystemMessage(
                    content="Only use the `add` tool to add two numbers. Answer all other questions directly without using any tools."
                )
            ] + messages

        result = llm_with_tool.invoke(messages)
        return {"messages": [result]}

    # Create graph
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")

    return builder.compile()

# Export the compiled graph
tool_agent = make_tool_graph()
