
from backend.agents.researcher import researcher_node


def test_researcher():

    initial_state = {
        "task_id": "TASK-001",

        "user_request": (
            "Research Agentic AI applications "
            "in software engineering and prepare a report."
        ),

        "status": "running",

        "plan": [
            "Research Agentic AI applications.",
            "Analyze benefits and challenges.",
            "Prepare a structured report."
        ],

        "completed_agents": [],

        "execution_log": [],

        "errors": []
    }

    print("\n" + "=" * 65)
    print("RESEARCHER AGENT TEST")
    print("=" * 65)

    print("\nUser Request:")
    print(initial_state["user_request"])

    result = researcher_node(initial_state)

    print("\nRESEARCH FINDINGS")
    print("=" * 65)

    print(result["research_findings"])

    print("\nSTATE UPDATES")
    print("=" * 65)

    print("Current Agent:", result["current_agent"])
    print("Status:", result["status"])
    print("Completed Agents:", result["completed_agents"])

    print("\nEXECUTION LOG")
    for log in result["execution_log"]:
        print("-", log)

    # Validate output

    assert result["current_agent"] == "researcher"

    assert result["status"] == "running"

    assert "RESEARCH TOPIC" in result["research_findings"]

    assert len(result["research_findings"]) > 100

    assert "researcher" in result["completed_agents"]

    assert len(result["execution_log"]) >= 2

    print("\nResearcher Agent Test: PASSED")


if __name__ == "__main__":
    test_researcher()
