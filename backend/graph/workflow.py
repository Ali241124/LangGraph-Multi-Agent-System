
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from backend.state.schemas import AgentState

from backend.agents.supervisor import supervisor_node
from backend.agents.researcher import researcher_node
from backend.agents.analyst import analyst_node
from backend.agents.writer import writer_node
from backend.agents.tool_executor import tool_executor_node


# ==========================================
# 1. WORKFLOW ROUTING
# ==========================================

def route_after_supervisor(state: AgentState):

    next_agent = state.get("next_agent")

    allowed_routes = {
        "researcher",
        "analyst",
        "writer",
        "tool_executor",
        "finish"
    }

    if next_agent not in allowed_routes:

        raise ValueError(
            f"Invalid workflow route: {next_agent}"
        )

    return next_agent


# ==========================================
# 2. BUILD LANGGRAPH WORKFLOW
# ==========================================

def build_workflow(checkpointer=None):

    builder = StateGraph(AgentState)

    # Register all five agents

    builder.add_node(
        "supervisor",
        supervisor_node
    )

    builder.add_node(
        "researcher",
        researcher_node
    )

    builder.add_node(
        "analyst",
        analyst_node
    )

    builder.add_node(
        "writer",
        writer_node
    )

    builder.add_node(
        "tool_executor",
        tool_executor_node
    )

    # Initial entry point

    builder.add_edge(
        START,
        "supervisor"
    )

    # Supervisor decides which agent runs next

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "tool_executor": "tool_executor",
            "finish": END
        }
    )

    # Each agent returns control to supervisor

    builder.add_edge(
        "researcher",
        "supervisor"
    )

    builder.add_edge(
        "analyst",
        "supervisor"
    )

    builder.add_edge(
        "writer",
        "supervisor"
    )

    builder.add_edge(
        "tool_executor",
        "supervisor"
    )

    # Compile the workflow

    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph
