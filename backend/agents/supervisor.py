
import json

from typing import Literal
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from backend.state.schemas import AgentState


# ==========================================
# 1. SUPERVISOR CONFIGURATION
# ==========================================

MODEL_NAME = "qwen2.5:7b"


# ==========================================
# 2. STRUCTURED SUPERVISOR RESPONSE
# ==========================================

class SupervisorDecision(BaseModel):

    plan: list[str] = Field(
        description="A list of steps required to complete the user's task."
    )

    next_agent: Literal[
        "researcher",
        "analyst",
        "writer",
        "tool_executor",
        "finish"
    ] = Field(
        description="The next agent to execute, or finish if all work is complete."
    )

    reasoning: str = Field(
        description="A short explanation of why this agent was selected."
    )


# ==========================================
# 3. SUPERVISOR SYSTEM PROMPT
# ==========================================

SUPERVISOR_PROMPT = """
You are the Supervisor Agent in a multi-agent AI system.

Your responsibility is to coordinate specialized agents.

AVAILABLE AGENTS:

1. researcher:
   Collects relevant information about the user's topic.

2. analyst:
   Analyzes the research findings and identifies insights.

3. writer:
   Creates a structured final report.

4. tool_executor:
   Performs requested external or file-related actions.

5. finish:
   Ends the workflow when all required work is complete.

WORKFLOW RULES:

- Every new task starts with researcher.
- After research, select analyst.
- After analysis, select writer.
- After writing, select tool_executor if the user explicitly
  requests an action such as saving, exporting, or sending a file.
- Otherwise, select finish.
- After tool execution, select finish.
- Never skip a required stage.
- Never select an agent that has already completed its work.

Generate a clear plan of 3 to 5 steps.

Return a structured decision.

Do not execute tools yourself.
"""


# ==========================================
# 4. INITIALIZE LOCAL LLM
# ==========================================

def build_supervisor_chain():

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    supervisor = llm.with_structured_output(
        SupervisorDecision,
        method="json_schema"
    )

    return supervisor


# ==========================================
# 5. DETERMINE ALLOWED NEXT AGENTS
# ==========================================

def get_allowed_next_agents(state: AgentState):

    completed = set(state.get("completed_agents", []))

    if "researcher" not in completed:
        return {"researcher"}

    if "analyst" not in completed:
        return {"analyst"}

    if "writer" not in completed:
        return {"writer"}

    if "tool_executor" not in completed:
        return {"tool_executor", "finish"}

    return {"finish"}


# ==========================================
# 6. SUPERVISOR NODE
# ==========================================

def supervisor_node(state: AgentState):

    supervisor = build_supervisor_chain()

    context = {
        "user_request": state["user_request"],
        "completed_agents": state.get("completed_agents", []),
        "existing_plan": state.get("plan", []),
        "research_available": bool(
            state.get("research_findings")
        ),
        "analysis_available": bool(
            state.get("analysis")
        ),
        "report_available": bool(
            state.get("report")
        ),
        "execution_available": bool(
            state.get("execution_result")
        )
    }

    response = supervisor.invoke([
        ("system", SUPERVISOR_PROMPT),
        ("human", json.dumps(context, indent=2))
    ])

    # ======================================
    # VALIDATE ROUTING DECISION
    # ======================================

    allowed_agents = get_allowed_next_agents(state)

    if response.next_agent not in allowed_agents:

        raise ValueError(
            f"Invalid supervisor decision: {response.next_agent}. "
            f"Allowed decisions: {allowed_agents}"
        )

    # Preserve the original plan after initial planning
    plan = state.get("plan") or response.plan

    # ======================================
    # RETURN STATE UPDATE
    # ======================================

    return {

        "current_agent": "supervisor",

        "next_agent": response.next_agent,

        "plan": plan,

        "status": (
            "completed"
            if response.next_agent == "finish"
            else "running"
        ),

        "execution_log": [
            f"Supervisor selected: {response.next_agent}",
            f"Reason: {response.reasoning}"
        ]
    }
