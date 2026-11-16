
from backend.agents.supervisor import supervisor_node


def test_supervisor():

    # Initial workflow state
    initial_state = {

        "task_id": "TASK-001",

        "user_request": (
            "Research Agentic AI applications "
            "in software engineering and prepare a report."
        ),

        "status": "pending",

        "completed_agents": [],

        "execution_log": [],

        "errors": []
    }

    print("\n" + "=" * 60)
    print("SUPERVISOR AGENT TEST")
    print("=" * 60)

    print("\nUser Request:")
    print(initial_state["user_request"])

    # Execute supervisor
    result = supervisor_node(initial_state)

    print("\nSUPERVISOR PLAN")
    print("-" * 60)

    for index, step in enumerate(result["plan"], start=1):
        print(f"{index}. {step}")

    print("\nSUPERVISOR DECISION")
    print("-" * 60)

    print("Current Agent:", result["current_agent"])
    print("Next Agent:", result["next_agent"])
    print("Workflow Status:", result["status"])

    print("\nEXECUTION LOG")
    print("-" * 60)

    for log in result["execution_log"]:
        print(log)

    # Verify the supervisor decision
    assert result["next_agent"] == "researcher"

    assert result["current_agent"] == "supervisor"

    assert result["status"] == "running"

    assert len(result["plan"]) >= 3

    print("\nSupervisor Agent Test: PASSED")


if __name__ == "__main__":
    test_supervisor()
