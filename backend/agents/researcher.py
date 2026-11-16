
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from backend.state.schemas import AgentState


# ==========================================
# 1. RESEARCHER CONFIGURATION
# ==========================================

MODEL_NAME = "qwen2.5:1.5b"


# ==========================================
# 2. STRUCTURED RESEARCH OUTPUT
# ==========================================

class ResearchResult(BaseModel):

    research_topic: str = Field(
        description="The main topic being researched."
    )

    summary: str = Field(
        description="A concise overview of the research topic."
    )

    key_concepts: list[str] = Field(
        description="Important concepts related to the topic."
    )

    findings: list[str] = Field(
        description="Detailed research findings and observations."
    )

    benefits: list[str] = Field(
        description="Potential benefits and applications."
    )

    challenges: list[str] = Field(
        description="Challenges, risks, and limitations."
    )

    research_gaps: list[str] = Field(
        description="Areas that need further research."
    )


# ==========================================
# 3. RESEARCHER SYSTEM PROMPT
# ==========================================

RESEARCHER_PROMPT = """
You are the Researcher Agent in a multi-agent AI system.

Your responsibility is to research the user's requested topic
and produce structured, detailed research findings.

INSTRUCTIONS:

1. Understand the user's research request.
2. Identify the main research topic.
3. Explain important concepts.
4. Produce meaningful research findings.
5. Identify potential benefits and applications.
6. Identify challenges and limitations.
7. Identify research gaps and unanswered questions.

IMPORTANT RULES:

- Be informative and technically accurate.
- Use clear and professional language.
- Organize findings logically.
- Avoid unnecessary repetition.
- Do not invent statistics, studies, papers, URLs,
  references, or citations.
- You are currently working without live web access.
- Clearly distinguish established concepts from
  potential applications or predictions.
- Do not write the final report.
- Do not perform external actions.

Return your findings using the required structured format.
"""


# ==========================================
# 4. INITIALIZE RESEARCHER MODEL
# ==========================================

def build_researcher_chain():

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    researcher = llm.with_structured_output(
        ResearchResult,
        method="json_schema"
    )

    return researcher


# ==========================================
# 5. RESEARCHER NODE
# ==========================================

def researcher_node(state: AgentState):

    researcher = build_researcher_chain()

    user_request = state["user_request"]

    existing_plan = state.get("plan", [])

    context = f"""
USER REQUEST:
{user_request}

SUPERVISOR PLAN:
{existing_plan}

Your task is to conduct the research stage only.
Return structured research findings.
"""

    response = researcher.invoke([
        ("system", RESEARCHER_PROMPT),
        ("human", context)
    ])

    # Convert structured research into readable text

    research_findings = f"""
RESEARCH TOPIC
{response.research_topic}

SUMMARY
{response.summary}

KEY CONCEPTS
{chr(10).join('- ' + item for item in response.key_concepts)}

RESEARCH FINDINGS
{chr(10).join('- ' + item for item in response.findings)}

BENEFITS
{chr(10).join('- ' + item for item in response.benefits)}

CHALLENGES
{chr(10).join('- ' + item for item in response.challenges)}

RESEARCH GAPS
{chr(10).join('- ' + item for item in response.research_gaps)}
"""

    # Return updates to the shared LangGraph state

    return {
        "current_agent": "researcher",
        "research_findings": research_findings,
        "completed_agents": ["researcher"],
        "status": "running",
        "execution_log": [
            "Researcher Agent started research.",
            "Researcher Agent completed research."
        ]
    }
