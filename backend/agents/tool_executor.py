
from pathlib import Path
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)

from langgraph.types import interrupt

from backend.state.schemas import AgentState


# ==========================================
# 1. CONFIGURATION
# ==========================================

REPORT_DIRECTORY = Path("data/reports")

REPORT_PATH = REPORT_DIRECTORY / "research_report.pdf"


# ==========================================
# 2. GENERATE PDF
# ==========================================

def generate_pdf(report: str) -> str:

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = styles["BodyText"]

    story = []

    for line in report.splitlines():

        line = line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):

            story.append(
                Paragraph(
                    escape(line[2:]),
                    title_style
                )
            )

        elif line.startswith("## "):

            story.append(
                Paragraph(
                    escape(line[3:]),
                    heading_style
                )
            )

        elif line.startswith("- "):

            story.append(
                Paragraph(
                    "&bull; " + escape(line[2:]),
                    body_style
                )
            )

        else:

            story.append(
                Paragraph(
                    escape(line),
                    body_style
                )
            )

    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    document.build(story)

    return str(REPORT_PATH)


# ==========================================
# 3. TOOL EXECUTOR NODE
# ==========================================

def tool_executor_node(state: AgentState):

    report = state.get("report")

    if not report:
        raise ValueError(
            "Cannot execute tool: Final report is missing."
        )

    # Define the requested action

    approval_request = {
        "action": "save_report_pdf",
        "description": (
            "Save the generated research report "
            "as a PDF file at data/reports/research_report.pdf."
        ),
        "requested_by": "tool_executor"
    }

    # Pause execution and request human approval

    approval_response = interrupt({
        "message": "Human approval is required.",
        "approval_request": approval_request
    })

    # Validate the human decision

    if not isinstance(approval_response, dict):

        raise ValueError(
            "Invalid approval response. Expected a dictionary."
        )

    approved = approval_response.get("approved")

    if not isinstance(approved, bool):

        raise ValueError(
            "Approval response must contain a boolean 'approved' value."
        )

    # Handle rejection

    if not approved:

        return {
            "current_agent": "tool_executor",
            "approval_request": approval_request,
            "approval_status": "rejected",
            "execution_result": (
                "PDF generation was rejected by the human reviewer."
            ),
            "completed_agents": ["tool_executor"],
            "status": "completed",
            "execution_log": [
                "Tool Executor received human rejection.",
                "PDF generation was cancelled."
            ]
        }

    # Execute the approved action

    if approval_request["action"] == "save_report_pdf":

        saved_path = generate_pdf(report)

    else:

        raise ValueError("Unsupported tool action.")

    # Return final execution result

    return {
        "current_agent": "tool_executor",
        "approval_request": approval_request,
        "approval_status": "approved",
        "execution_result": (
            f"Research report successfully saved at: {saved_path}"
        ),
        "completed_agents": ["tool_executor"],
        "status": "completed",
        "execution_log": [
            "Tool Executor received human approval.",
            f"PDF generated successfully: {saved_path}"
        ]
    }
