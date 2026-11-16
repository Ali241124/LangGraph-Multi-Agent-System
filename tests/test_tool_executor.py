
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from backend.state.schemas import AgentState
from backend.agents.tool_executor import tool_executor_node


# ==========================================
# 1. BUILD A SMALL TEST GRAPH
# ==========================================

def build_test_graph():

    builder = StateGraph(AgentState)

    builder.add_node(
        "tool_executor",
        tool_executor_node
    )

    builder.add_edge(START, "tool_executor")
    builder.add_edge("tool_executor", END)

    checkpointer = InMemorySaver()

    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph


# ==========================================
# 2. SAMPLE STATE
# ==========================================

def get_initial_state():

    return {
        "task_id": "TASK-001",

        "user_request": (
            "Research Agentic AI applications "
            "and save the report as PDF."
        ),

        "status": "running",

        "report": """
# Agentic AI Applications in Software Engineering

## Executive Summary

Agentic AI can assist software engineering teams
with code generation, testing, and automation.

## Benefits

- Increased productivity.
- Faster development cycles.
- Improved testing support.

## Challenges

- Reliability of AI-generated code.
- Integration complexity.
- Need for human supervision.

## Conclusion

Agentic AI can support software engineering
when implemented with appropriate human oversight.
""",

        "completed_agents": [
            "researcher",
            "analyst",
            "writer"
        ],

        "execution_log": [],

        "errors": []
    }


# ==========================================
# 3. TEST HUMAN APPROVAL
# ==========================================

def test_approval():

    graph = build_test_graph()

    config = {
        "configurable": {
            "thread_id": "approval-test-001"
        }
    }

    print("\n" + "=" * 65)
    print("TOOL EXECUTOR APPROVAL TEST")
    print("=" * 65)

    # First invocation pauses at interrupt()

    paused_result = graph.invoke(
        get_initial_state(),
        config=config
    )

    print("\nWORKFLOW PAUSED")

    interrupts = paused_result.get("__interrupt__", [])

    assert len(interrupts) > 0

    approval_data = interrupts[0].value

    print("Approval Request:")
    print(approval_data)

    print("\nHuman Decision: APPROVED")

    # Resume workflow with approval

    final_result = graph.invoke(
        Command(
            resume={
                "approved": True
            }
        ),
        config=config
    )

    print("\nEXECUTION RESULT")
    print(final_result["execution_result"])

    print("\nSTATE UPDATES")

    print("Approval Status:", final_result["approval_status"])
    print("Workflow Status:", final_result["status"])

    # Validate result

    assert final_result["approval_status"] == "approved"

    assert final_result["status"] == "completed"

    assert "successfully saved" in (
        final_result["execution_result"]
    )

    pdf_path = Path(
        final_result["execution_result"].split(": ", 1)[1]
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0

    print("\nPDF File Exists:", pdf_path.exists())

    print("\nHuman Approval Test: PASSED")


# ==========================================
# 4. TEST HUMAN REJECTION
# ==========================================

def test_rejection():

    graph = build_test_graph()

    config = {
        "configurable": {
            "thread_id": "rejection-test-001"
        }
    }

    print("\n" + "=" * 65)
    print("TOOL EXECUTOR REJECTION TEST")
    print("=" * 65)

    # Pause at approval interrupt

    paused_result = graph.invoke(
        get_initial_state(),
        config=config
    )

    assert len(paused_result.get("__interrupt__", [])) > 0

    print("\nHuman Decision: REJECTED")

    # Resume with rejection

    final_result = graph.invoke(
        Command(
            resume={
                "approved": False
            }
        ),
        config=config
    )

    print("\nEXECUTION RESULT")
    print(final_result["execution_result"])

    print("\nApproval Status:", final_result["approval_status"])

    assert final_result["approval_status"] == "rejected"

    assert final_result["status"] == "completed"

    assert "rejected" in final_result["execution_result"]

    print("\nHuman Rejection Test: PASSED")


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":

    test_approval()

    test_rejection()
