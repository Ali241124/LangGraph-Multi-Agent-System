
from backend.state.schemas import AgentState


def test_state_updates():

    state: AgentState = {
        "task_id": "TASK-001",
        "user_request": "Prepare an AI research report",
        "status": "running",
        "completed_agents": [],
        "execution_log": [],
        "errors": []
    }

    # Researcher completes its work
    researcher_update = {
        "completed_agents": ["researcher"],
        "execution_log": ["Research completed"]
    }

    state["completed_agents"] += researcher_update["completed_agents"]
    state["execution_log"] += researcher_update["execution_log"]

    # Analyst completes its work
    analyst_update = {
        "completed_agents": ["analyst"],
        "execution_log": ["Analysis completed"]
    }

    state["completed_agents"] += analyst_update["completed_agents"]
    state["execution_log"] += analyst_update["execution_log"]

    print("\nUPDATED STATE")
    print("=" * 50)

    print("Completed Agents:", state["completed_agents"])
    print("Execution Log:", state["execution_log"])

    assert len(state["completed_agents"]) == 2
    assert len(state["execution_log"]) == 2

    print("\nState Update Test: PASSED")


if __name__ == "__main__":
    test_state_updates()
