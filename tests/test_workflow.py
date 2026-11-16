
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from backend.graph.workflow import build_workflow
from backend.graph.checkpointer import get_checkpointer


# ==========================================
# 1. INITIAL WORKFLOW STATE
# ==========================================

def get_initial_state():

    return {
        "task_id": "TASK-FULL-001",

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


# ==========================================
# 2. COMPLETE WORKFLOW TEST
# ==========================================

def test_complete_workflow():

    print("\n" + "=" * 70)
    print("LANGGRAPH MULTI-AGENT WORKFLOW TEST")
    print("=" * 70)

    # Create a unique thread for this test

    config = {
        "configurable": {
            "thread_id": f"workflow-{uuid4()}"
        }
    }

    # Keep SQLite connection open throughout execution

    with get_checkpointer() as checkpointer:

        graph = build_workflow(
            checkpointer=checkpointer
        )

        print("\nSTARTING WORKFLOW...")

        # ----------------------------------
        # STEP 1: START WORKFLOW
        # ----------------------------------

        paused_result = graph.invoke(
            get_initial_state(),
            config=config
        )

        print("\nWORKFLOW EXECUTION PAUSED")

        interrupts = paused_result.get(
            "__interrupt__",
            []
        )

        assert len(interrupts) > 0, (
            "Expected human approval interrupt."
        )

        approval_data = interrupts[0].value

        print("\nHUMAN APPROVAL REQUEST")
        print("-" * 70)

        print(approval_data)

        # ----------------------------------
        # STEP 2: HUMAN APPROVAL
        # ----------------------------------

        print("\nHUMAN DECISION: APPROVED")

        final_result = graph.invoke(
            Command(
                resume={
                    "approved": True
                }
            ),
            config=config
        )

        # ----------------------------------
        # STEP 3: VERIFY ALL AGENTS
        # ----------------------------------

        print("\nWORKFLOW RESULTS")
        print("=" * 70)

        print(
            "Completed Agents:",
            final_result["completed_agents"]
        )

        print(
            "Workflow Status:",
            final_result["status"]
        )

        print(
            "Approval Status:",
            final_result["approval_status"]
        )

        # ----------------------------------
        # STEP 4: DISPLAY EXECUTION LOG
        # ----------------------------------

        print("\nEXECUTION LOG")
        print("-" * 70)

        for log in final_result["execution_log"]:
            print("-", log)

        # ----------------------------------
        # STEP 5: VERIFY GENERATED REPORT
        # ----------------------------------

        report = final_result["report"]

        assert len(report) > 500

        assert "Executive Summary" in report

        assert "Recommendations" in report

        # ----------------------------------
        # STEP 6: VERIFY PDF
        # ----------------------------------

        pdf_path = Path(
            "data/reports/research_report.pdf"
        )

        assert pdf_path.exists()

        assert pdf_path.stat().st_size > 0

        print("\nPDF GENERATED:", pdf_path)
        print("PDF SIZE:", pdf_path.stat().st_size, "bytes")

        # ----------------------------------
        # STEP 7: FINAL ASSERTIONS
        # ----------------------------------

        completed = set(
            final_result["completed_agents"]
        )

        required_agents = {
            "researcher",
            "analyst",
            "writer",
            "tool_executor"
        }

        assert required_agents.issubset(completed)

        assert final_result["status"] == "completed"

        assert final_result["approval_status"] == "approved"

        assert final_result["next_agent"] == "finish"

        print("\n" + "=" * 70)
        print("COMPLETE WORKFLOW TEST: PASSED")
        print("=" * 70)


if __name__ == "__main__":

    test_complete_workflow()
