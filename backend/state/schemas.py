
from typing import TypedDict, Annotated, Literal, NotRequired
import operator


# ==========================================
# 1. AGENT DEFINITIONS
# ==========================================

AgentName = Literal[
    "supervisor",
    "researcher",
    "analyst",
    "writer",
    "tool_executor"
]

RoutingTarget = Literal[
    "researcher",
    "analyst",
    "writer",
    "tool_executor",
    "finish"
]

# ==========================================
# 2. WORKFLOW STATUS
# ==========================================

WorkflowStatus = Literal[
    "pending",
    "running",
    "paused_for_approval",
    "completed",
    "failed"
]


# ==========================================
# 3. APPROVAL STATUS
# ==========================================

ApprovalStatus = Literal[
    "not_requested",
    "pending",
    "approved",
    "rejected"
]


# ==========================================
# 4. APPROVAL REQUEST STRUCTURE
# ==========================================

class ApprovalRequest(TypedDict):
    action: str
    description: str
    requested_by: str


# ==========================================
# 5. MAIN SHARED STATE
# ==========================================

class AgentState(TypedDict):

    # Required initial information
    task_id: str
    user_request: str
    status: WorkflowStatus

    # Fields updated during execution
    current_agent: NotRequired[AgentName]
    next_agent: NotRequired[RoutingTarget]

    plan: NotRequired[list[str]]

    research_findings: NotRequired[str]
    analysis: NotRequired[str]
    report: NotRequired[str]

    approval_request: NotRequired[ApprovalRequest]
    approval_status: NotRequired[ApprovalStatus]

    execution_result: NotRequired[str]

    # Reducer fields: accumulate information
    completed_agents: Annotated[list[str], operator.add]
    execution_log: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
