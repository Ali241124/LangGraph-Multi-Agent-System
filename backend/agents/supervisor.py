
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

    # Stage 1: Research
    if "researcher" not in completed:
        return {"researcher"}

    # Stage 2: Analysis
    if "analyst" not in completed:
        return {"analyst"}

    # Stage 3: Report writing
    if "writer" not in completed:
        return {"writer"}

    # Stage 4: Optional file execution
    if "tool_executor" not in completed:

        user_request = state["user_request"].lower()

        requires_pdf_action = any(
            keyword in user_request
            for keyword in ["pdf", "save", "export"]
        )

        if requires_pdf_action:
            return {"tool_executor"}

        return {"finish"}

    # All required work is complete
    return {"finish"}


# ==========================================
# 6. SUPERVISOR NODE
# ==========================================

def supervisor_node(state: AgentState):

    # Get the allowed next route first.
    allowed_agents = get_allowed_next_agents(state)

    # Build context for the LLM.
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

    # Initialize the LLM.
    supervisor = build_supervisor_chain()

    # Ask the LLM for a structured decision.
    response = supervisor.invoke([
        ("system", SUPERVISOR_PROMPT),
        ("human", json.dumps(context, indent=2))
    ])

    # ======================================
    # VALIDATE AND SELECT ROUTING DECISION
    # ======================================

    # If only one route is allowed, use it directly.
    # This prevents the LLM from selecting an already
    # completed agent.

    if len(allowed_agents) == 1:

        selected_agent = next(iter(allowed_agents))

    # If multiple routes are allowed, validate
    # the LLM's decision against the allowed routes.

    elif response.next_agent in allowed_agents:

        selected_agent = response.next_agent

    else:

        raise ValueError(
            f"Invalid supervisor decision: {response.next_agent}. "
            f"Allowed decisions: {allowed_agents}"
        )

    # Preserve the original plan after initial planning.
    plan = state.get("plan") or response.plan

    # ======================================
    # RETURN STATE UPDATE
    # ======================================

    return {

        "current_agent": "supervisor",

        "next_agent": selected_agent,

        "plan": plan,

        "status": (
            "completed"
            if selected_agent == "finish"
            else "running"
        ),

        "execution_log": [
            f"Supervisor selected: {selected_agent}",
            f"Reason: {response.reasoning}"
        ]
    }
