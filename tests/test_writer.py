
from backend.agents.writer import writer_node


def test_writer():

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

RESEARCH FINDINGS
- AI agents can assist with code generation.
- Agents can automate software testing.
- AI systems can support project management.

BENEFITS
- Increased productivity.
- Faster development cycles.
- Improved software quality.

CHALLENGES
- Reliability of AI-generated code.
- Integration complexity.
- Transparency and explainability.
""",

        "analysis": """
KEY INSIGHTS
- AI agents can automate repetitive development tasks.
- Human supervision remains important.

BENEFITS ANALYSIS
- Automation can improve developer productivity.
- Faster feedback can improve development cycles.

CHALLENGES ANALYSIS
- Incorrect generated code requires human review.
- Integration requires appropriate infrastructure.

IMPACT ASSESSMENT
Agentic AI has potential to improve software
engineering workflows when properly supervised.

RECOMMENDATIONS
- Introduce AI agents gradually.
- Maintain human review.
- Evaluate output quality continuously.

CONCLUSION
Agentic AI can support software development
but should complement human engineering expertise.
""",

        "completed_agents": ["researcher", "analyst"],

        "execution_log": [],

        "errors": []
    }

    print("\n" + "=" * 65)
    print("WRITER AGENT TEST")
    print("=" * 65)

    print("\nGenerating research report...")

    result = writer_node(initial_state)

    print("\nGENERATED RESEARCH REPORT")
    print("=" * 65)

    print(result["report"])

    print("\nSTATE UPDATES")
    print("=" * 65)

    print("Current Agent:", result["current_agent"])
    print("Status:", result["status"])
    print("Completed Agents:", result["completed_agents"])

    print("\nEXECUTION LOG")

    for log in result["execution_log"]:
        print("-", log)

    # Validate report

    assert result["current_agent"] == "writer"

    assert result["status"] == "running"

    assert "# " in result["report"]

    assert "Executive Summary" in result["report"]

    assert "Introduction" in result["report"]

    assert "Research Findings" in result["report"]

    assert "Recommendations" in result["report"]

    assert "Conclusion" in result["report"]

    assert len(result["report"]) > 500

    assert "writer" in result["completed_agents"]

    print("\nWriter Agent Test: PASSED")


def test_writer_without_research():

    state = {
        "task_id": "TASK-002",
        "user_request": "Prepare a research report.",
        "status": "running",
        "completed_agents": [],
        "execution_log": [],
        "errors": []
    }

    try:

        writer_node(state)

    except ValueError as error:

        assert "Research findings are missing" in str(error)

        print("\nMissing Research Validation: PASSED")

    else:

        raise AssertionError(
            "Writer should reject missing research findings."
        )


if __name__ == "__main__":

    test_writer()

    test_writer_without_research()
