
from backend.state.schemas import AgentState


def test_state_schema():

    initial_state: AgentState = {
        "task_id": "TASK-001",
        "user_request": (
            "Research Agentic AI applications "
            "in software engineering."
        ),
        "status": "pending",

        "completed_agents": [],
        "execution_log": [],
        "errors": []
    }

    print("\nINITIAL WORKFLOW STATE")
    print("=" * 50)

    for key, value in initial_state.items():
        print(f"{key}: {value}")

    print("=" * 50)

    assert initial_state["task_id"] == "TASK-001"
    assert initial_state["status"] == "pending"
    assert initial_state["completed_agents"] == []

    print("\nState Schema Test: PASSED")


if __name__ == "__main__":
    test_state_schema()
