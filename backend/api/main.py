
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from langgraph.types import Command

from backend.graph.workflow import build_workflow
from backend.graph.checkpointer import get_checkpointer


# ==========================================
# 1. APPLICATION LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting LangGraph API...")

    # Keep the SQLite connection open while
    # the FastAPI application is running.
    with get_checkpointer() as checkpointer:

        app.state.graph = build_workflow(
            checkpointer=checkpointer
        )

        # Store background workflow statuses.
        app.state.workflow_jobs = {}

        print("LangGraph workflow initialized.")
        print("SQLite checkpointer connected.")
        print("Background workflow tracking initialized.")

        yield

    print("LangGraph API shutting down.")


# ==========================================
# 2. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="LangGraph Multi-Agent Research System",
    description=(
        "An AI research automation system "
        "powered by LangGraph, Ollama, and FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# 3. REQUEST SCHEMAS
# ==========================================

class WorkflowRequest(BaseModel):

    user_request: str = Field(
        min_length=10,
        max_length=5000,
        description="The research task submitted by the user."
    )


class ApprovalRequest(BaseModel):

    approved: bool = Field(
        description="Whether the requested action is approved."
    )


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def get_graph():

    graph = getattr(app.state, "graph", None)

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Workflow engine is not initialized."
        )

    return graph


def get_workflow_config(thread_id: str):

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def get_pending_interrupts(snapshot):

    interrupts = []

    for task in snapshot.tasks:

        for interrupt in getattr(task, "interrupts", []):

            interrupts.append(interrupt.value)

    return interrupts

def run_workflow_background(thread_id: str, initial_state: dict):

    graph = get_graph()
    config = get_workflow_config(thread_id)

    # Track the workflow's current execution status.
    app.state.workflow_jobs[thread_id] = "running"

    try:

        result = graph.invoke(
            initial_state,
            config=config
        )

        interrupts = result.get("__interrupt__", [])

        if interrupts:
            app.state.workflow_jobs[thread_id] = (
                "paused_for_approval"
            )
        else:
            app.state.workflow_jobs[thread_id] = (
                result.get("status", "completed")
            )

    except Exception as error:

        app.state.workflow_jobs[thread_id] = "failed"

        print(
            f"Workflow {thread_id} failed: {error}"
        )


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/api/v1/health")
def health_check():

    return {
        "status": "healthy",
        "service": "LangGraph Multi-Agent API",
        "version": "1.0.0"
    }


# ==========================================
# 6. START WORKFLOW
# ==========================================


@app.post("/api/v1/workflows", status_code=202)
def start_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks
):

    graph = get_graph()

    thread_id = str(uuid4())

    initial_state = {

        "task_id": thread_id,

        "user_request": request.user_request,

        "status": "pending",

        "completed_agents": [],

        "execution_log": [],

        "errors": []
    }

    # Register the workflow before scheduling it.
    app.state.workflow_jobs[thread_id] = "queued"

    # Schedule background execution.
    background_tasks.add_task(
        run_workflow_background,
        thread_id,
        initial_state
    )

    # Return immediately instead of waiting for Ollama.
    return {

        "thread_id": thread_id,

        "status": "queued",

        "message": "Workflow accepted and scheduled.",

        "status_endpoint": (
            f"/api/v1/workflows/{thread_id}"
        )
    }


# ==========================================
# 7. GET WORKFLOW STATUS
# ==========================================

@app.get("/api/v1/workflows/{thread_id}")
def get_workflow_status(thread_id: str):

    graph = get_graph()
    config = get_workflow_config(thread_id)

    job_status = app.state.workflow_jobs.get(thread_id)

    snapshot = graph.get_state(config)

    # The background task may not have created its first
    # checkpoint yet. Return its in-memory status meanwhile.
    if not snapshot.values:

        if job_status is not None:

            return {
                "thread_id": thread_id,
                "status": job_status,
                "completed_agents": [],
                "approval_required": False
            }

        raise HTTPException(
            status_code=404,
            detail="Workflow not found."
        )

    values = snapshot.values

    interrupts = get_pending_interrupts(snapshot)

    approval_required = len(interrupts) > 0

    workflow_status = values.get("status", "unknown")

    if approval_required:
        workflow_status = "paused_for_approval"

    elif job_status == "failed":
        workflow_status = "failed"

    elif job_status in {"queued", "running"}:
        workflow_status = job_status

    return {

        "thread_id": thread_id,

        "status": workflow_status,

        "current_agent": values.get("current_agent"),

        "completed_agents": values.get(
            "completed_agents", []
        ),

        "approval_required": approval_required,

        "approval_request": (
            interrupts[0]
            if approval_required
            else None
        ),

        "execution_log": values.get(
            "execution_log", []
        ),

        "errors": values.get("errors", [])
    }

# ==========================================
# 8. HUMAN APPROVAL ENDPOINT
# ==========================================

@app.post("/api/v1/workflows/{thread_id}/approval")
def submit_approval(
    thread_id: str,
    request: ApprovalRequest
):

    graph = get_graph()

    config = get_workflow_config(thread_id)

    # Confirm that the workflow exists and
    # is currently waiting for approval.
    snapshot = graph.get_state(config)

    if not snapshot.values:

        raise HTTPException(
            status_code=404,
            detail="Workflow not found."
        )

    interrupts = get_pending_interrupts(snapshot)

    if not interrupts:

        raise HTTPException(
            status_code=409,
            detail="Workflow is not waiting for approval."
        )

    
    try:

        app.state.workflow_jobs[thread_id] = "running"

        result = graph.invoke(
            Command(
                resume={
                    "approved": request.approved
                }
            ),
            config=config
        )

        interrupts_after_resume = result.get(
            "__interrupt__", []
        )

        if interrupts_after_resume:
            new_status = "paused_for_approval"
        else:
            new_status = result.get("status", "completed")

        app.state.workflow_jobs[thread_id] = new_status

        return {

            "thread_id": thread_id,

            "approval_decision": (
                "approved"
                if request.approved
                else "rejected"
            ),

            "status": new_status,

            "approval_status": result.get(
                "approval_status"
            ),

            "execution_result": result.get(
                "execution_result"
            ),

            "completed_agents": result.get(
                "completed_agents", []
            )
        }

    except Exception as error:

        app.state.workflow_jobs[thread_id] = "failed"

        raise HTTPException(
            status_code=500,
            detail=f"Approval processing failed: {str(error)}"
        )



# ==========================================
# 9. GET GENERATED REPORT
# ==========================================

@app.get("/api/v1/workflows/{thread_id}/report")
def get_workflow_report(thread_id: str):

    graph = get_graph()

    config = get_workflow_config(thread_id)

    snapshot = graph.get_state(config)

    if not snapshot.values:

        raise HTTPException(
            status_code=404,
            detail="Workflow not found."
        )

    values = snapshot.values

    report = values.get("report")

    if not report:

        raise HTTPException(
            status_code=404,
            detail="Report has not been generated yet."
        )

    return {

        "thread_id": thread_id,

        "status": values.get("status"),

        "report": report
    }
