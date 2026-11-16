
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from backend.graph.workflow import build_workflow
from backend.graph.checkpointer import get_checkpointer


def get_initial_state():

    return {
        "task_id": "TASK-CHECKPOINT-001",

        "user_request": (
            "Research Agentic AI applications in software "
            "engineering, analyze its benefits and challenges, "
            "prepare a professional report, and save it as a PDF."
        ),

        "status": "pending",

        "completed_agents": [],
        "execution_log": [],
        "errors": []
    }


def test_checkpoint_persistence():

    print("\n" + "=" * 70)
    print("CHECKPOINT PERSISTENCE AND RECOVERY TEST")
    print("=" * 70)

    # Use the same thread ID before and after reopening SQLite.
    config = {
        "configurable": {
            "thread_id": f"persistent-workflow-{uuid4()}"
        }
    }

    # ==========================================
    # STEP 1: START WORKFLOW
    # ==========================================

    print("\nSTEP 1: STARTING WORKFLOW")

    with get_checkpointer() as checkpointer:

        graph = build_workflow(checkpointer=checkpointer)

        paused_result = graph.invoke(
            get_initial_state(),
            config=config
        )

        interrupts = paused_result.get("__interrupt__", [])

        assert len(interrupts) > 0, (
            "Expected workflow to pause for human approval."
        )

        print("Workflow paused for human approval.")

        print(
            "Agents completed before approval:",
            paused_result["completed_agents"]
        )

    # The first SQLite connection is now closed.
    print("\nSQLite connection closed.")

    # ==========================================
    # STEP 2: REOPEN SQLITE AND RECOVER STATE
    # ==========================================

    print("\nSTEP 2: REOPENING SQLITE DATABASE")

    with get_checkpointer() as checkpointer:

        graph = build_workflow(checkpointer=checkpointer)

        recovered_state = graph.get_state(config)

        assert recovered_state is not None, (
            "Workflow checkpoint was not recovered."
        )

        assert recovered_state.values, (
            "Recovered workflow state is empty."
        )

        print("Checkpoint successfully recovered.")

        recovered_agents = recovered_state.values.get(
            "completed_agents", []
        )

        print("Recovered completed agents:", recovered_agents)

        required_before_approval = {
            "researcher",
            "analyst",
            "writer"
        }

        assert required_before_approval.issubset(
            set(recovered_agents)
        ), "Previously completed agents were not recovered."

        # ==========================================
        # STEP 3: RESUME INTERRUPTED WORKFLOW
        # ==========================================

        print("\nSTEP 3: RESUMING WORKFLOW")

        final_result = graph.invoke(
            Command(resume={"approved": True}),
            config=config
        )

        print("Workflow resumed successfully.")

        # ==========================================
        # STEP 4: VERIFY FINAL STATE
        # ==========================================

        print("\nSTEP 4: VERIFYING FINAL RESULTS")

        completed_agents = set(
            final_result["completed_agents"]
        )

        required_agents = {
            "researcher",
            "analyst",
            "writer",
            "tool_executor"
        }

        assert required_agents.issubset(completed_agents)

        assert final_result["status"] == "completed"

        assert final_result["approval_status"] == "approved"

        assert final_result["next_agent"] == "finish"

        # ==========================================
        # STEP 5: VERIFY PDF
        # ==========================================

        pdf_path = Path(
            "data/reports/research_report.pdf"
        )

        assert pdf_path.exists(), "PDF was not generated."

        assert pdf_path.stat().st_size > 0, (
            "Generated PDF is empty."
        )

        print("\nFINAL WORKFLOW STATUS:")
        print("Completed Agents:", final_result["completed_agents"])
        print("Status:", final_result["status"])
        print("Approval:", final_result["approval_status"])

        print("\nPDF VERIFIED:", pdf_path)
        print("PDF SIZE:", pdf_path.stat().st_size, "bytes")

        print("\n" + "=" * 70)
        print("CHECKPOINT PERSISTENCE TEST: PASSED")
        print("=" * 70)


if __name__ == "__main__":
    test_checkpoint_persistence()
