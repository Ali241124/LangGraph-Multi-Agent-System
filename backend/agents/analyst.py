
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from backend.state.schemas import AgentState


# ==========================================
# 1. ANALYST CONFIGURATION
# ==========================================

MODEL_NAME = "qwen2.5:7b"


# ==========================================
# 2. STRUCTURED ANALYSIS OUTPUT
# ==========================================

class AnalysisResult(BaseModel):

    key_insights: list[str] = Field(
        description="Important insights derived from the research."
    )

    benefits_analysis: list[str] = Field(
        description="Detailed evaluation of the identified benefits."
    )

    challenges_analysis: list[str] = Field(
        description="Detailed evaluation of the identified challenges."
    )

    impact_assessment: str = Field(
        description="Overall assessment of the topic's impact."
    )

    recommendations: list[str] = Field(
        description="Practical recommendations based on the research."
    )

    conclusion: str = Field(
        description="A concise analytical conclusion."
    )


# ==========================================
# 3. ANALYST SYSTEM PROMPT
# ==========================================

ANALYST_PROMPT = """
You are the Analyst Agent in a multi-agent AI system.

Your responsibility is to critically analyze research findings
provided by the Researcher Agent.

YOUR TASKS:

1. Identify the most important insights.
2. Evaluate the potential benefits.
3. Examine the challenges and limitations.
4. Assess the overall impact.
5. Develop practical recommendations.
6. Produce a meaningful analytical conclusion.

ANALYSIS RULES:

- Base your analysis on the provided research findings.
- Do not simply repeat the research.
- Explain why findings are important.
- Identify relationships between benefits and challenges.
- Consider practical implementation requirements.
- Distinguish evidence from assumptions.
- Do not invent statistics, studies, references, or citations.
- Do not introduce unsupported claims.
- Do not write the final report.
- Do not execute external tools.

Return your analysis using the required structured format.
"""


# ==========================================
# 4. INITIALIZE ANALYST MODEL
# ==========================================

def build_analyst_chain():

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    analyst = llm.with_structured_output(
        AnalysisResult,
        method="json_schema"
    )

    return analyst


# ==========================================
# 5. ANALYST NODE
# ==========================================

def analyst_node(state: AgentState):

    # Ensure research is available

    research_findings = state.get("research_findings")

    if not research_findings:
        raise ValueError(
            "Cannot run Analyst Agent: "
            "Research findings are missing."
        )

    analyst = build_analyst_chain()

    # Prepare research context

    context = f"""
USER REQUEST:
{state["user_request"]}

RESEARCH FINDINGS:
{research_findings}

SUPERVISOR PLAN:
{state.get("plan", [])}

Your task is to critically analyze the research findings.
Produce insights, evaluations, recommendations, and a conclusion.
"""

    # Generate structured analysis

    response = analyst.invoke([
        ("system", ANALYST_PROMPT),
        ("human", context)
    ])

    # Format analysis into readable text

    analysis = f"""
KEY INSIGHTS
{chr(10).join('- ' + item for item in response.key_insights)}

BENEFITS ANALYSIS
{chr(10).join('- ' + item for item in response.benefits_analysis)}

CHALLENGES ANALYSIS
{chr(10).join('- ' + item for item in response.challenges_analysis)}

IMPACT ASSESSMENT
{response.impact_assessment}

RECOMMENDATIONS
{chr(10).join('- ' + item for item in response.recommendations)}

CONCLUSION
{response.conclusion}
"""

    # Update shared workflow state

    return {
        "current_agent": "analyst",
        "analysis": analysis,
        "completed_agents": ["analyst"],
        "status": "running",
        "execution_log": [
            "Analyst Agent started analysis.",
            "Analyst Agent completed analysis."
        ]
    }
