
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from backend.state.schemas import AgentState


# ==========================================
# 1. WRITER CONFIGURATION
# ==========================================

MODEL_NAME = "qwen2.5:1.5b"


# ==========================================
# 2. STRUCTURED REPORT OUTPUT
# ==========================================

class ReportResult(BaseModel):

    title: str = Field(
        description="Professional title of the research report."
    )

    executive_summary: str = Field(
        description="Concise overview of the complete report."
    )

    introduction: str = Field(
        description="Introduction and background of the topic."
    )

    research_findings: list[str] = Field(
        description="Important findings from the research."
    )

    benefits_and_applications: list[str] = Field(
        description="Benefits and practical applications."
    )

    challenges_and_limitations: list[str] = Field(
        description="Challenges and limitations."
    )

    analytical_insights: list[str] = Field(
        description="Important insights derived from the analysis."
    )

    recommendations: list[str] = Field(
        description="Practical recommendations."
    )

    conclusion: str = Field(
        description="Final conclusion of the report."
    )


# ==========================================
# 3. WRITER SYSTEM PROMPT
# ==========================================

WRITER_PROMPT = """
You are the Writer Agent in a multi-agent AI system.

Your responsibility is to create a professional research report
using the research findings and analytical results provided.

REPORT STRUCTURE:

1. Executive Summary
2. Introduction
3. Research Findings
4. Benefits and Applications
5. Challenges and Limitations
6. Analytical Insights
7. Recommendations
8. Conclusion

WRITING RULES:

- Write in professional academic language.
- Maintain a logical flow between sections.
- Use the research and analysis provided.
- Preserve important information.
- Avoid unnecessary repetition.
- Do not invent statistics, research papers, references,
  URLs, or citations.
- Do not introduce unsupported claims.
- Clearly explain technical concepts.
- Make the report informative and readable.
- Do not execute external tools.
- Do not save or export files.

Return the completed report using the required
structured format.
"""


# ==========================================
# 4. INITIALIZE WRITER MODEL
# ==========================================

def build_writer_chain():

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    writer = llm.with_structured_output(
        ReportResult,
        method="json_schema"
    )

    return writer


# ==========================================
# 5. FORMAT FINAL REPORT
# ==========================================

def format_report(response: ReportResult) -> str:

    report = f"""
# {response.title}


## 1. Executive Summary

{response.executive_summary}


## 2. Introduction

{response.introduction}


## 3. Research Findings

{chr(10).join('- ' + item for item in response.research_findings)}


## 4. Benefits and Applications

{chr(10).join('- ' + item for item in response.benefits_and_applications)}


## 5. Challenges and Limitations

{chr(10).join('- ' + item for item in response.challenges_and_limitations)}


## 6. Analytical Insights

{chr(10).join('- ' + item for item in response.analytical_insights)}


## 7. Recommendations

{chr(10).join('- ' + item for item in response.recommendations)}


## 8. Conclusion

{response.conclusion}
"""

    return report.strip()


# ==========================================
# 6. WRITER NODE
# ==========================================

def writer_node(state: AgentState):

    research_findings = state.get("research_findings")

    analysis = state.get("analysis")

    # Validate required inputs

    if not research_findings:
        raise ValueError(
            "Cannot write report: Research findings are missing."
        )

    if not analysis:
        raise ValueError(
            "Cannot write report: Analysis is missing."
        )

    writer = build_writer_chain()

    # Prepare input context

    context = f"""
USER REQUEST:
{state["user_request"]}

SUPERVISOR PLAN:
{state.get("plan", [])}

RESEARCH FINDINGS:
{research_findings}

ANALYST RESULTS:
{analysis}

Create a complete professional research report
based on the provided information.
"""

    # Generate report

    response = writer.invoke([
        ("system", WRITER_PROMPT),
        ("human", context)
    ])

    # Format report

    report = format_report(response)

    # Update shared state

    return {
        "current_agent": "writer",
        "report": report,
        "completed_agents": ["writer"],
        "status": "running",
        "execution_log": [
            "Writer Agent started report generation.",
            "Writer Agent completed report generation."
        ]
    }
