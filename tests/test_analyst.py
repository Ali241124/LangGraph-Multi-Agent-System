
from backend.agents.analyst import analyst_node


def test_analyst():

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

        "research_findings": """
RESEARCH TOPIC
Agentic AI applications in software engineering

SUMMARY
Agentic AI systems can make decisions and perform
actions to achieve software engineering goals.

KEY CONCEPTS
- Autonomous agents
- Code generation
- Software testing
- Project management

RESEARCH FINDINGS
- AI agents can assist with code generation.
- Agents can automate parts of software testing.
- AI systems can support project management.

BENEFITS
- Increased productivity.
- Faster development cycles.
- Improved software quality.

CHALLENGES
- Reliability of AI-generated code.
- Integration complexity.
- Transparency and explainability.

RESEARCH GAPS
- Long-term impact on development teams.
- User trust in autonomous systems.
""",

        "completed_agents": ["researcher"],

        "execution_log": [],

        "errors": []
    }

    print("\n" + "=" * 65)
    print("ANALYST AGENT TEST")
    print("=" * 65)

    print("\nRunning Analyst Agent...")

    result = analyst_node(initial_state)

    print("\nANALYSIS RESULTS")
    print("=" * 65)

    print(result["analysis"])

    print("\nSTATE UPDATES")
    print("=" * 65)

    print("Current Agent:", result["current_agent"])
    print("Status:", result["status"])
    print("Completed Agents:", result["completed_agents"])

    print("\nEXECUTION LOG")

    for log in result["execution_log"]:
        print("-", log)

    # Validate output

    assert result["current_agent"] == "analyst"

    assert result["status"] == "running"

    assert "KEY INSIGHTS" in result["analysis"]

    assert "RECOMMENDATIONS" in result["analysis"]

    assert "CONCLUSION" in result["analysis"]

    assert len(result["analysis"]) > 100

    assert "analyst" in result["completed_agents"]

    assert len(result["execution_log"]) >= 2

    print("\nAnalyst Agent Test: PASSED")


def test_analyst_without_research():

    state = {
        "task_id": "TASK-002",
        "user_request": "Analyze Agentic AI.",
        "status": "running",
        "completed_agents": [],
        "execution_log": [],
        "errors": []
    }

    try:

        analyst_node(state)

    except ValueError as error:

        assert "Research findings are missing" in str(error)

        print("\nMissing Research Validation: PASSED")

    else:

        raise AssertionError(
            "Analyst should reject missing research findings."
        )


if __name__ == "__main__":

    test_analyst()

    test_analyst_without_research()
