from __future__ import annotations

import asyncio
import io
import re
import sys
import uuid
from datetime import datetime
from html import escape
from typing import Any

# ============================================================
# WINDOWS EVENT LOOP FIX
# ============================================================
#
# psycopg AsyncPostgresSaver requires SelectorEventLoop on Windows.
# This MUST happen before asyncio.run() is used.
# ============================================================

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()

from graph import run_graph


# ============================================================
# REPORTLAB
# ============================================================

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# AGENT METADATA
# ============================================================

AGENT_META = {
    "flight_agent": {
        "label": "Flight Agent",
        "icon": "✈️",
        "desc": "Searches flights & fares",
        "color": "#5B8DEF",
    },
    "hotel_agent": {
        "label": "Hotel Agent",
        "icon": "🏨",
        "desc": "Finds stays in budget",
        "color": "#F2A65A",
    },
    "weather_agent": {
        "label": "Weather Agent",
        "icon": "🌤️",
        "desc": "Checks climate & forecast",
        "color": "#4FC3A1",
    },
    "budget_agent": {
        "label": "Budget Agent",
        "icon": "💰",
        "desc": "Validates total spend",
        "color": "#E86A6A",
    },
    "itinerary_agent": {
        "label": "Itinerary Agent",
        "icon": "🗺️",
        "desc": "Builds the day-by-day plan",
        "color": "#B084F0",
    },
}

AGENT_ORDER = list(AGENT_META.keys())

RESULT_KEY_FOR_AGENT = {
    "flight_agent": "flight_results",
    "hotel_agent": "hotel_results",
    "weather_agent": "weather_results",
    "budget_agent": "budget_results",
    "itinerary_agent": "itinerary",
}


# ============================================================
# SESSION STATE
# ============================================================


def initialize_session_state() -> None:

    defaults = {
        "thread_id": f"demo_user_{uuid.uuid4().hex[:8]}",
        "user_query": "",
        "latest_result": None,
        "waiting_for_approval": False,
        "user_id": "demo_user",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# SAFE HTML
# ============================================================


def safe_html(value: Any) -> str:
    """
    Escape arbitrary/LLM-generated content before inserting into
    custom HTML.
    """

    if value is None:
        return ""

    return escape(
        str(value),
        quote=True,
    )


def safe_multiline_html(value: Any) -> str:
    """
    Escape content while preserving line breaks for HTML rendering.
    """

    text = safe_html(value)

    return text.replace(
        "\n",
        "<br>",
    )


def render_html(html: str) -> None:
    """
    Render custom HTML safely.

    Leading indentation is stripped because Streamlit Markdown
    interprets 4+ leading spaces as code blocks.
    """

    lines = html.split("\n")

    dedented = "\n".join(line.lstrip() for line in lines)

    st.markdown(
        dedented,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:

    render_html(
        f"""
<div class="section-label-row">
    <div class="section-label">{safe_html(text)}</div>
    <div class="section-rule"></div>
</div>
"""
    )


# ============================================================
# GENERIC HELPERS
# ============================================================


def as_dict(value: Any) -> dict[str, Any]:

    if isinstance(value, dict):
        return value

    return {}


def as_list(value: Any) -> list[Any]:

    if isinstance(value, list):
        return value

    if value is None:
        return []

    return [value]


def get_critic_verdict(
    result: dict[str, Any],
) -> dict[str, Any]:

    verdict = result.get(
        "critic_verdict",
        {},
    )

    return as_dict(verdict)


def get_critic_scores(
    result: dict[str, Any],
) -> dict[str, Any]:

    verdict = get_critic_verdict(result)

    scores = verdict.get(
        "scores",
        {},
    )

    return as_dict(scores)


def get_iteration(
    result: dict[str, Any],
) -> int:

    try:
        return max(
            0,
            int(
                result.get(
                    "iteration_count",
                    0,
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        return 0


def get_decision(
    result: dict[str, Any],
) -> str:

    verdict = get_critic_verdict(result)

    decision = (
        str(
            verdict.get(
                "decision",
                "",
            )
        )
        .upper()
        .strip()
    )

    if decision:
        return decision

    if verdict.get("passed"):
        return "PASS"

    return "REPLAN"


def get_selected_agents(
    result: dict[str, Any],
) -> list[str]:

    selected = result.get(
        "selected_agents",
        [],
    )

    if not isinstance(
        selected,
        list,
    ):
        return []

    return [agent for agent in AGENT_ORDER if agent in selected]


# ============================================================
# PDF HELPERS
# ============================================================


def _clean_for_pdf(text: Any) -> str:

    if text is None:
        return ""

    text = str(text)

    # Remove HTML tags.
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    # Basic HTML entities.
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )

    return text.strip()


def _pdf_escape(text: Any) -> str:

    return escape(
        _clean_for_pdf(text),
        quote=False,
    )


def _markdown_to_flowables(
    text: str,
    styles: dict[str, ParagraphStyle],
):

    flowables = []

    text = _clean_for_pdf(text)

    if not text:
        flowables.append(
            Paragraph(
                "<i>No content available.</i>",
                styles["PlanBody"],
            )
        )

        return flowables

    lines = text.split("\n")

    bullet_buffer: list[str] = []

    def flush_bullets():

        if not bullet_buffer:
            return

        for bullet in bullet_buffer:
            safe_bullet = _pdf_escape(bullet)

            flowables.append(
                Paragraph(
                    f"&bull;&nbsp;&nbsp;{safe_bullet}",
                    styles["PlanBullet"],
                )
            )

        bullet_buffer.clear()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush_bullets()

            flowables.append(
                Spacer(
                    1,
                    4,
                )
            )

            continue

        # ----------------------------------------------------
        # Extract heading level before escaping.
        # ----------------------------------------------------

        if line.startswith("### "):
            flush_bullets()

            flowables.append(
                Paragraph(
                    _pdf_escape(line[4:]),
                    styles["PlanH3"],
                )
            )

            continue

        if line.startswith("## "):
            flush_bullets()

            flowables.append(
                Paragraph(
                    _pdf_escape(line[3:]),
                    styles["PlanH2"],
                )
            )

            continue

        if line.startswith("# "):
            flush_bullets()

            flowables.append(
                Paragraph(
                    _pdf_escape(line[2:]),
                    styles["PlanH1"],
                )
            )

            continue

        # ----------------------------------------------------
        # Bullet
        # ----------------------------------------------------

        if line.startswith(
            (
                "- ",
                "* ",
            )
        ):
            bullet_buffer.append(line[2:])

            continue

        # ----------------------------------------------------
        # Numbered list
        # ----------------------------------------------------

        if re.match(
            r"^\d+\.\s",
            line,
        ):
            flush_bullets()

            flowables.append(
                Paragraph(
                    _pdf_escape(line),
                    styles["PlanBullet"],
                )
            )

            continue

        # ----------------------------------------------------
        # Normal paragraph
        # ----------------------------------------------------

        flush_bullets()

        flowables.append(
            Paragraph(
                _pdf_escape(line),
                styles["PlanBody"],
            )
        )

    flush_bullets()

    return flowables


def build_travel_plan_pdf(
    result: dict,
    user_id: str,
    thread_id: str,
    user_query: str,
) -> bytes:

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title="Travel Plan",
    )

    base = getSampleStyleSheet()

    styles = {
        "TitleMain": ParagraphStyle(
            "TitleMain",
            parent=base["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1F2430"),
            spaceAfter=4,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#6B7080"),
            spaceAfter=14,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2E3A59"),
            spaceBefore=16,
            spaceAfter=8,
        ),
        "PlanH1": ParagraphStyle(
            "PlanH1",
            parent=base["Heading1"],
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#1F2430"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "PlanH2": ParagraphStyle(
            "PlanH2",
            parent=base["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#2E3A59"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "PlanH3": ParagraphStyle(
            "PlanH3",
            parent=base["Heading3"],
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#3B4664"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "PlanBody": ParagraphStyle(
            "PlanBody",
            parent=base["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#2A2E38"),
            spaceAfter=4,
        ),
        "PlanBullet": ParagraphStyle(
            "PlanBullet",
            parent=base["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#2A2E38"),
            leftIndent=14,
            spaceAfter=3,
        ),
        "MetaLabel": ParagraphStyle(
            "MetaLabel",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#8B90A0"),
        ),
    }

    story = []

    verdict = get_critic_verdict(result)

    decision = get_decision(result)

    quality_score = verdict.get("quality_score")

    confidence = verdict.get("confidence")

    iteration = get_iteration(result)

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "Your Travel Plan",
            styles["TitleMain"],
        )
    )

    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    metadata = (
        f"Generated {generated_at}"
        f" &nbsp;|&nbsp; "
        f"User: {_pdf_escape(user_id)}"
        f" &nbsp;|&nbsp; "
        f"Thread: {_pdf_escape(thread_id)}"
    )

    story.append(
        Paragraph(
            metadata,
            styles["Subtitle"],
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            color=colors.HexColor("#D8DCE6"),
            thickness=1,
        )
    )

    story.append(
        Spacer(
            1,
            10,
        )
    )

    # ========================================================
    # QUALITY STATUS
    # ========================================================

    story.append(
        Paragraph(
            "Quality Gate",
            styles["SectionHeader"],
        )
    )

    quality_text = (
        f"Decision: {decision}"
        f" &nbsp;|&nbsp; "
        f"Quality: {quality_score if quality_score is not None else 'N/A'}"
        f" &nbsp;|&nbsp; "
        f"Confidence: {confidence if confidence is not None else 'N/A'}"
        f" &nbsp;|&nbsp; "
        f"Iteration: {iteration}"
    )

    story.append(
        Paragraph(
            quality_text,
            styles["PlanBody"],
        )
    )

    # ========================================================
    # REQUEST
    # ========================================================

    story.append(
        Paragraph(
            "Trip Request",
            styles["SectionHeader"],
        )
    )

    story.append(
        Paragraph(
            _pdf_escape(user_query) or "—",
            styles["PlanBody"],
        )
    )

    # ========================================================
    # AGENTS
    # ========================================================

    selected_agents = get_selected_agents(result)

    if selected_agents:
        story.append(
            Paragraph(
                "Agents Involved",
                styles["SectionHeader"],
            )
        )

        rows = [
            [
                "Agent",
                "Role",
            ]
        ]

        for key in AGENT_ORDER:
            if key not in selected_agents:
                continue

            meta = AGENT_META[key]

            rows.append(
                [
                    _pdf_escape(meta["label"]),
                    _pdf_escape(meta["desc"]),
                ]
            )

        table = Table(
            rows,
            colWidths=[
                2.0 * inch,
                4.3 * inch,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#EEF1F8"),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#2E3A59"),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        9.5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#E2E5EC"),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                ]
            )
        )

        story.append(table)

    # ========================================================
    # ITINERARY
    # ========================================================

    itinerary = result.get(
        "itinerary",
        "",
    )

    if itinerary:
        story.append(
            Paragraph(
                "Itinerary",
                styles["SectionHeader"],
            )
        )

        story.extend(
            _markdown_to_flowables(
                itinerary,
                styles,
            )
        )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    final_response = result.get(
        "final_response",
        "",
    )

    if final_response:
        story.append(
            Paragraph(
                "Final Travel Plan",
                styles["SectionHeader"],
            )
        )

        story.extend(
            _markdown_to_flowables(
                final_response,
                styles,
            )
        )

    # ========================================================
    # FOOTER
    # ========================================================

    story.append(
        Spacer(
            1,
            16,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            color=colors.HexColor("#D8DCE6"),
            thickness=1,
        )
    )

    story.append(
        Spacer(
            1,
            6,
        )
    )

    story.append(
        Paragraph(
            "Generated by Multi-Agent Travel Planner — "
            "Supervisor + Quality Gate + Human-in-the-Loop",
            styles["MetaLabel"],
        )
    )

    document.build(story)

    pdf_bytes = buffer.getvalue()

    buffer.close()

    return pdf_bytes


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0A0C11;
    --panel: #12151D;
    --panel-2: #161A24;
    --hairline: rgba(201,162,75,0.16);
    --hairline-strong: rgba(201,162,75,0.32);
    --brass: #C9A24B;
    --brass-dim: rgba(201,162,75,0.55);
    --coral: #E2542D;
    --parchment: #F4EEE1;
    --parchment-ink: #2B2416;
    --parchment-line: rgba(43,36,22,0.14);
    --slate: #838C9E;
    --text: #ECEEF3;
}

html, body, .stApp, .stApp p, .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div:not(:has([data-testid="stIconMaterial"])),
[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]) {
    font-family: 'Inter', -apple-system, sans-serif;
}

[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

.stApp {
    background:
        radial-gradient(
            circle at 12% -6%,
            rgba(201,162,75,0.07) 0%,
            transparent 40%
        ),
        repeating-linear-gradient(
            120deg,
            rgba(255,255,255,0.012) 0px,
            rgba(255,255,255,0.012) 1px,
            transparent 1px,
            transparent 64px
        ),
        var(--ink);
}

[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--hairline);
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    color: var(--slate);
    transition: color 0.15s ease;
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
    color: var(--brass);
}

/* ============================================================
   HERO
   ============================================================ */

.hero-wrap {
    padding: 30px 34px 24px 34px;
    border-radius: 4px;
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-top: 3px solid var(--brass);
    margin-bottom: 30px;
}

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--brass);
    margin-bottom: 12px;
}

.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.5rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 10px;
}

.hero-sub {
    color: var(--slate);
    font-size: 0.98rem;
    line-height: 1.6;
    margin-bottom: 20px;
    max-width: 720px;
}

.ticket-row {
    display: flex;
    gap: 0;
    border-top: 1px dashed var(--hairline-strong);
    padding-top: 14px;
    flex-wrap: wrap;
}

.ticket-field {
    padding-right: 22px;
    margin-right: 22px;
    border-right: 1px dashed var(--hairline-strong);
}

.ticket-field:last-child {
    border-right: none;
}

.ticket-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--slate);
    margin-bottom: 3px;
}

.ticket-value {
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--text);
}

/* ============================================================
   SECTION
   ============================================================ */

.section-label-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 30px 0 14px 0;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--brass);
    white-space: nowrap;
}

.section-rule {
    flex: 1;
    border-top: 1px dashed var(--hairline-strong);
}

/* ============================================================
   AGENT CARDS
   ============================================================ */

.agent-card {
    border-radius: 3px;
    padding: 0;
    border: 1px solid var(--hairline);
    background: var(--panel);
    min-height: 175px;
    overflow: hidden;
    position: relative;
}

.agent-card.active {
    border-color: var(--hairline-strong);
    border-left: 3px solid var(--accent);
}

.agent-card.inactive {
    opacity: 0.4;
}

.agent-card.replanned {
    border-color: rgba(226,84,45,0.55);
    border-left: 3px solid var(--coral);
}

.agent-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    background: var(--panel-2);
}

.agent-gate {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    color: var(--slate);
}

.agent-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 2px;
}

.agent-status.on {
    background: rgba(201,162,75,0.14);
    color: var(--brass);
    border: 1px solid var(--hairline-strong);
}

.agent-status.off {
    background: transparent;
    color: var(--slate);
    border: 1px solid rgba(255,255,255,0.06);
}

.agent-status.warn {
    background: rgba(226,84,45,0.12);
    color: #F08A6D;
    border: 1px solid rgba(226,84,45,0.3);
}

.agent-card-body {
    padding: 14px;
}

.agent-icon {
    font-size: 1.35rem;
    display: block;
    margin-bottom: 8px;
}

.agent-name {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--text);
}

.agent-desc {
    font-size: 0.78rem;
    color: var(--slate);
    margin-top: 4px;
    line-height: 1.4;
}

/* ============================================================
   PANELS
   ============================================================ */

.panel {
    border-radius: 3px;
    border: 1px solid var(--hairline);
    background: var(--panel);
    padding-bottom: 16px;
    margin-bottom: 18px;
    overflow: hidden;
}

.panel-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--text);
    padding: 13px 18px;
    background: var(--panel-2);
    border-bottom: 1px solid var(--hairline);
    margin-bottom: 14px;
}

/* ============================================================
   REASONING
   ============================================================ */

.reasoning-box {
    border: 1px solid var(--hairline);
    border-left: 3px solid var(--brass);
    background: var(--panel);
    padding: 16px 20px;
    border-radius: 0 3px 3px 0;
}

.reasoning-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--brass);
    margin-bottom: 8px;
}

.reasoning-text {
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ============================================================
   QUALITY GATE
   ============================================================ */

.quality-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.quality-card {
    background: var(--panel);
    border: 1px solid var(--hairline);
    padding: 14px;
    border-radius: 3px;
}

.quality-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--slate);
}

.quality-value {
    margin-top: 6px;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
}

.quality-pass {
    color: #63D4A6 !important;
}

.quality-review {
    color: var(--brass) !important;
}

.quality-replan {
    color: #F0785D !important;
}

.quality-neutral {
    color: var(--slate) !important;
}

.dimension-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px dashed var(--hairline);
}

.dimension-name {
    font-size: 0.82rem;
    color: var(--text);
}

.dimension-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--brass);
}

.warning-box {
    background: rgba(226,84,45,0.08);
    border: 1px solid rgba(226,84,45,0.25);
    border-left: 3px solid var(--coral);
    padding: 13px 16px;
    border-radius: 3px;
    margin-bottom: 12px;
}

.info-box {
    background: rgba(201,162,75,0.07);
    border: 1px solid var(--hairline);
    border-left: 3px solid var(--brass);
    padding: 13px 16px;
    border-radius: 3px;
}

/* ============================================================
   PARCHMENT
   ============================================================ */

.parchment {
    border-radius: 4px;
    padding: 26px 30px;
    background: var(--parchment);
    border: 1px solid var(--parchment-line);
    box-shadow: 0 14px 34px -20px rgba(0,0,0,0.55);
    color: var(--parchment-ink);
}

.parchment-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--parchment-line);
    padding-bottom: 10px;
    margin-bottom: 16px;
}

.parchment-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(43,36,22,0.55);
}

.parchment-stamp {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    color: var(--brass);
    border: 1px solid var(--brass);
    padding: 3px 10px;
    border-radius: 999px;
}

.parchment .stMarkdown,
.parchment p,
.parchment li {
    color: var(--parchment-ink) !important;
}

/* ============================================================
   NATIVE WIDGETS
   ============================================================ */

.stButton > button,
.stDownloadButton > button {
    background: var(--panel-2);
    border: 1px solid var(--hairline-strong);
    color: var(--text);
    border-radius: 3px;
    font-weight: 600;
    font-size: 0.85rem;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--brass);
    color: var(--brass);
}

.stButton > button[kind="primary"] {
    background: var(--coral);
    border-color: var(--coral);
    color: #fff;
}

[data-testid="stChatInput"] {
    border: 1px solid var(--hairline-strong);
    border-radius: 4px;
    background: var(--panel);
}

hr {
    border-top: 1px dashed var(--hairline-strong) !important;
}

@media (max-width: 900px) {
    .quality-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .hero-title {
        font-size: 2rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(
        """
<div style="font-family:'IBM Plex Mono', monospace;
            font-size:0.68rem;
            letter-spacing:0.14em;
            text-transform:uppercase;
            color:#C9A24B;
            margin-bottom:2px;">
    Session
</div>

<div style="font-family:'Fraunces', serif;
            font-size:1.2rem;
            font-weight:600;
            color:#ECEEF3;
            margin-bottom:14px;">
    Passenger Details
</div>
"""
    )

    user_id = st.text_input(
        "User ID",
        value=st.session_state.get(
            "user_id",
            "demo_user",
        ),
    )

    st.session_state.user_id = user_id

    if st.button(
        "➕ New Thread",
        use_container_width=True,
    ):
        st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"

        st.session_state.latest_result = None
        st.session_state.user_query = ""
        st.session_state.waiting_for_approval = False

        st.rerun()

    render_html(
        f"""
<div style="font-family:'IBM Plex Mono', monospace;
            font-size:0.72rem;
            color:#838C9E;
            margin-top:10px;">
    THREAD&nbsp;&nbsp;
    <span style="color:#ECEEF3;">
        {safe_html(st.session_state.thread_id)}
    </span>
</div>
"""
    )

    st.divider()

    render_html(
        """
<div style="font-family:'IBM Plex Mono', monospace;
            font-size:0.68rem;
            letter-spacing:0.14em;
            text-transform:uppercase;
            color:#C9A24B;
            margin-bottom:2px;">
    Roster
</div>

<div style="font-family:'Fraunces', serif;
            font-size:1.2rem;
            font-weight:600;
            color:#ECEEF3;
            margin-bottom:14px;">
    Specialist Agents
</div>
"""
    )

    for index, key in enumerate(AGENT_ORDER):
        meta = AGENT_META[key]

        render_html(
            f"""
<div style="display:flex;
            align-items:center;
            gap:10px;
            padding:8px 0;
            border-bottom:1px dashed rgba(201,162,75,0.16);">

    <div style="font-family:'IBM Plex Mono', monospace;
                font-size:0.62rem;
                color:#838C9E;
                width:24px;">
        0{index + 1}
    </div>

    <div style="font-size:1.05rem;">
        {meta["icon"]}
    </div>

    <div>
        <div style="font-weight:600;
                    font-size:0.85rem;
                    color:#ECEEF3;">
            {safe_html(meta["label"])}
        </div>

        <div style="font-size:0.72rem;
                    color:#838C9E;">
            {safe_html(meta["desc"])}
        </div>
    </div>

</div>
"""
        )


# ============================================================
# HERO
# ============================================================

render_html(
    """
<div class="hero-wrap">

    <div class="hero-eyebrow">
        Itinerary Desk · Supervisor-Routed Planning
    </div>

    <div class="hero-title">
        Multi-Agent Travel Planner
    </div>

    <div class="hero-sub">
        Describe your trip. A supervisor routes the request to the
        specialists it needs, generates a plan, evaluates it through
        deterministic and semantic quality gates, and requests human
        approval before finalization.
    </div>

    <div class="ticket-row">

        <div class="ticket-field">
            <div class="ticket-label">Routing</div>
            <div class="ticket-value">
                Supervisor-directed
            </div>
        </div>

        <div class="ticket-field">
            <div class="ticket-label">Quality</div>
            <div class="ticket-value">
                Critic + deterministic gates
            </div>
        </div>

        <div class="ticket-field">
            <div class="ticket-label">Recovery</div>
            <div class="ticket-value">
                Targeted replanning
            </div>
        </div>

        <div class="ticket-field">
            <div class="ticket-label">Review</div>
            <div class="ticket-value">
                Human-in-the-loop
            </div>
        </div>

        <div class="ticket-field">
            <div class="ticket-label">Output</div>
            <div class="ticket-value">
                Plan + PDF
            </div>
        </div>

    </div>

</div>
"""
)


# ============================================================
# LANGGRAPH CONFIG
# ============================================================

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
    }
}


# ============================================================
# CHAT INPUT
# ============================================================

new_query = st.chat_input(
    placeholder=(
        "Plan a 7-day Japan trip under Rs. 2 lakh. "
        "I prefer budget hotels and no overnight flights."
    )
)


# ============================================================
# NEW TRIP
# ============================================================

if new_query:
    st.session_state.user_query = new_query
    st.session_state.waiting_for_approval = False

    input_state = {
        "messages": [HumanMessage(content=new_query)],
        "user_id": user_id,
        "user_query": new_query,
        # Compatibility fields for current state/agents.
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0,
        # New execution state defaults.
        "iteration_count": 0,
        "critic_verdict": {},
        "unresolved_violations": [],
        "is_replan": False,
    }

    with st.spinner("🧠 Supervisor → specialists → itinerary → quality gate..."):
        try:
            result = asyncio.run(
                run_graph(
                    input_state,
                    config,
                )
            )

        except Exception as exc:
            st.error("Something went wrong while planning your trip.")

            st.exception(exc)

            result = None

    if result is not None:
        st.session_state.latest_result = result

        st.session_state.waiting_for_approval = "__interrupt__" in result


# ============================================================
# CURRENT RESULT
# ============================================================

result = st.session_state.get("latest_result")


# ============================================================
# SUPERVISOR PLAN
# ============================================================

if result:
    selected_agents = get_selected_agents(result)

    section_label("Supervisor Plan")

    reasoning = result.get(
        "supervisor_reasoning",
        "Supervisor completed routing.",
    )

    render_html(
        f"""
<div class="reasoning-box">

    <div class="reasoning-eyebrow">
        Routing Note
    </div>

    <div class="reasoning-text">
        {safe_multiline_html(reasoning)}
    </div>

</div>
"""
    )

    # ========================================================
    # SELECTED AGENTS
    # ========================================================

    section_label("Agent Execution")

    cols = st.columns(len(AGENT_ORDER))

    current_iteration = get_iteration(result)

    is_replan = bool(
        result.get(
            "is_replan",
            False,
        )
    )

    for col, index, key in zip(
        cols,
        range(len(AGENT_ORDER)),
        AGENT_ORDER,
    ):
        meta = AGENT_META[key]

        is_selected = key in selected_agents

        content = result.get(
            RESULT_KEY_FOR_AGENT[key],
            "",
        )

        has_output = bool(content)

        if not is_selected:
            status_text = "SKIPPED"
            status_class = "off"
            card_class = "inactive"

        elif has_output and is_replan:
            status_text = "REPLANNED"
            status_class = "warn"
            card_class = "replanned"

        elif has_output:
            status_text = "COMPLETED"
            status_class = "on"
            card_class = "active"

        else:
            status_text = "SELECTED"
            status_class = "on"
            card_class = "active"

        bar_widths = [
            3,
            1,
            2,
            1,
            3,
            1,
            1,
            2,
            3,
            1,
            2,
            1,
        ]

        barcode_bars = "".join(
            f'<span style="width:{w}px;height:{6 + (i % 3) * 3}px;"></span>'
            for i, w in enumerate(bar_widths)
        )

        with col:
            render_html(
                f"""
<div class="agent-card {card_class}"
     style="--accent:{meta["color"]};">

    <div class="agent-card-top">

        <span class="agent-gate">
            GATE 0{index + 1}
        </span>

        <span class="agent-status {status_class}">
            {status_text}
        </span>

    </div>

    <div class="agent-card-body">

        <span class="agent-icon">
            {meta["icon"]}
        </span>

        <div class="agent-name">
            {safe_html(meta["label"])}
        </div>

        <div class="agent-desc">
            {safe_html(meta["desc"])}
        </div>

    </div>

    <div class="agent-perf"></div>

    <div class="agent-barcode">
        {barcode_bars}
    </div>

</div>
"""
            )

    # ========================================================
    # QUALITY GATE
    # ========================================================

    section_label("Quality Gate")

    verdict = get_critic_verdict(result)

    decision = get_decision(result)

    quality_score = verdict.get("quality_score")

    confidence = verdict.get("confidence")

    degraded_mode = bool(
        verdict.get(
            "degraded_mode",
            False,
        )
    )

    deterministic_passed = verdict.get("deterministic_checks_passed")

    llm_passed = verdict.get("llm_check_passed")

    decision_class = {
        "PASS": "quality-pass",
        "REVIEW": "quality-review",
        "REPLAN": "quality-replan",
    }.get(
        decision,
        "quality-neutral",
    )

    score_display = (
        f"{quality_score:.0f}"
        if isinstance(
            quality_score,
            (int, float),
        )
        else "—"
    )

    confidence_display = (
        f"{confidence:.0%}"
        if isinstance(
            confidence,
            (int, float),
        )
        else "—"
    )

    render_html(
        f"""
<div class="quality-grid">

    <div class="quality-card">
        <div class="quality-label">
            Decision
        </div>
        <div class="quality-value {decision_class}">
            {safe_html(decision)}
        </div>
    </div>

    <div class="quality-card">
        <div class="quality-label">
            Quality Score
        </div>
        <div class="quality-value">
            {score_display}
        </div>
    </div>

    <div class="quality-card">
        <div class="quality-label">
            Confidence
        </div>
        <div class="quality-value">
            {confidence_display}
        </div>
    </div>

    <div class="quality-card">
        <div class="quality-label">
            Iteration
        </div>
        <div class="quality-value">
            {current_iteration}/{3}
        </div>
    </div>

</div>
"""
    )

    # ========================================================
    # QUALITY FLAGS
    # ========================================================

    if degraded_mode:
        render_html(
            """
<div class="warning-box">
    <strong>⚠ Degraded quality evaluation</strong><br>
    The semantic critic could not complete normally. The plan should
    receive human review before being treated as trustworthy.
</div>
"""
        )

    if deterministic_passed is False:
        render_html(
            """
<div class="warning-box">
    <strong>⚠ Deterministic checks failed</strong><br>
    One or more hard validation rules failed. Review the violations
    before approving the plan.
</div>
"""
        )

    if llm_passed is False:
        render_html(
            """
<div class="warning-box">
    <strong>⚠ Semantic critic rejected the plan</strong><br>
    The LLM quality gate identified issues that require review or
    targeted replanning.
</div>
"""
        )

    # ========================================================
    # DIMENSION SCORES
    # ========================================================

    scores = get_critic_scores(result)

    if scores:
        score_col, detail_col = st.columns([1.1, 1])

        with score_col:
            dimensions = [
                (
                    "constraint",
                    "Constraint",
                ),
                (
                    "budget",
                    "Budget",
                ),
                (
                    "routing",
                    "Routing",
                ),
                (
                    "itinerary",
                    "Itinerary",
                ),
                (
                    "evidence",
                    "Evidence",
                ),
                (
                    "safety",
                    "Safety",
                ),
            ]

            render_html(
                """
<div class="panel">
    <div class="panel-title">
        Quality Dimensions
    </div>
"""
            )

            for key, label in dimensions:
                value = scores.get(key)

                if isinstance(
                    value,
                    (int, float),
                ):
                    value_display = f"{value:.0f}/100"

                else:
                    value_display = "—"

                render_html(
                    f"""
<div class="dimension-row">

    <span class="dimension-name">
        {safe_html(label)}
    </span>

    <span class="dimension-score">
        {safe_html(value_display)}
    </span>

</div>
"""
                )

            render_html(
                """
</div>
"""
            )

        with detail_col:
            violations = as_list(verdict.get("violations"))

            suggestions = as_list(verdict.get("suggestions"))

            if violations:
                render_html(
                    """
<div class="panel">
    <div class="panel-title">
        Violations
    </div>
"""
                )

                for violation in violations:
                    render_html(
                        f"""
<div style="padding:7px 18px;
            color:#F08A6D;
            font-size:0.82rem;
            line-height:1.45;">
    • {safe_html(violation)}
</div>
"""
                    )

                render_html(
                    """
</div>
"""
                )

            if suggestions:
                render_html(
                    """
<div class="panel">
    <div class="panel-title">
        Critic Suggestions
    </div>
"""
                )

                for suggestion in suggestions:
                    render_html(
                        f"""
<div style="padding:7px 18px;
            color:#ECEEF3;
            font-size:0.82rem;
            line-height:1.45;">
    • {safe_html(suggestion)}
</div>
"""
                    )

                render_html(
                    """
</div>
"""
                )

    # ========================================================
    # CRITIC REASONING
    # ========================================================

    critic_reasoning = verdict.get(
        "reasoning",
        "",
    )

    if critic_reasoning:
        render_html(
            f"""
<div class="info-box">

    <strong>Critic assessment</strong><br><br>

    <span style="color:#ECEEF3;
                 font-size:0.84rem;
                 line-height:1.55;">
        {safe_multiline_html(critic_reasoning)}
    </span>

</div>
"""
        )

    # ========================================================
    # RESPONSIBLE AGENTS
    # ========================================================

    responsible_agents = [
        agent
        for agent in as_list(verdict.get("responsible_agents"))
        if agent in AGENT_META
    ]

    if responsible_agents:
        responsible_labels = [
            AGENT_META[agent]["label"] for agent in responsible_agents
        ]

        render_html(
            f"""
<div style="margin-top:10px;
            color:#838C9E;
            font-size:0.76rem;">
    Replanning responsibility:
    <strong style="color:#C9A24B;">
        {safe_html(", ".join(responsible_labels))}
    </strong>
</div>
"""
        )

    # ========================================================
    # AGENT OUTPUTS
    # ========================================================

    section_label("Agent Outputs")

    output_agents = [
        agent
        for agent in AGENT_ORDER
        if agent != "itinerary_agent" and agent in selected_agents
    ]

    if output_agents:
        out_cols = st.columns(2)

        for index, key in enumerate(output_agents):
            meta = AGENT_META[key]

            content = result.get(
                RESULT_KEY_FOR_AGENT[key],
                "",
            )

            if content:
                body = safe_multiline_html(content)

            else:
                body = (
                    '<span style="color:#838C9E;'
                    'font-size:0.85rem;">'
                    "No output yet."
                    "</span>"
                )

            with out_cols[index % 2]:
                render_html(
                    f"""
<div class="panel">

    <div class="panel-title">
        {meta["icon"]}
        {safe_html(meta["label"])}
    </div>

    <div style="padding:0 18px;
                color:var(--text);
                font-size:0.9rem;
                line-height:1.6;
                overflow-wrap:anywhere;">

        {body}

    </div>

</div>
"""
                )

    else:
        st.caption("No specialist agents were selected.")

    # ========================================================
    # DRAFT ITINERARY
    # ========================================================

    section_label("Draft Itinerary")

    draft = ""

    if "__interrupt__" in result:
        interrupts = result.get(
            "__interrupt__",
            [],
        )

        if interrupts:
            interrupt_value = interrupts[0].value

            if isinstance(
                interrupt_value,
                dict,
            ):
                draft = interrupt_value.get(
                    "draft_itinerary",
                    "",
                )

    else:
        draft = result.get(
            "itinerary",
            "",
        )

    if draft:
        stamp = (
            "AWAITING APPROVAL"
            if st.session_state.get(
                "waiting_for_approval",
                False,
            )
            else "DRAFT"
        )

        render_html(
            f"""
<div class="parchment">

    <div class="parchment-header">

        <span class="parchment-title">
            Working Draft
        </span>

        <span class="parchment-stamp">
            {safe_html(stamp)}
        </span>

    </div>

    <div style="white-space:pre-wrap;
                line-height:1.65;
                color:var(--parchment-ink);">
        {safe_multiline_html(draft)}
    </div>

</div>
"""
        )

    else:
        st.info("The itinerary has not been generated yet.")


# ============================================================
# HUMAN APPROVAL
# ============================================================

if st.session_state.get(
    "waiting_for_approval",
    False,
):
    st.divider()

    section_label("Human Approval")

    decision = get_decision(result or {})

    if decision == "REVIEW":
        st.warning("The quality gate recommends human review before finalization.")

    elif decision == "REPLAN":
        st.warning("The planner has requested another planning iteration.")

    else:
        st.info("The planner is waiting for your approval.")

    approved = st.radio(
        "Approve this draft?",
        [
            "Yes",
            "No, revise it",
        ],
        horizontal=True,
    )

    feedback = st.text_area(
        "Feedback",
        placeholder=("Example: Reduce hotel cost and add more cultural activities."),
        disabled=approved == "Yes",
    )

    if st.button(
        "✅ Submit Approval",
        type="primary",
    ):
        with st.spinner("Finalizing the travel plan..."):
            try:
                resume_command = Command(
                    resume={
                        "approved": (approved == "Yes"),
                        "feedback": feedback,
                    }
                )

                final_result = asyncio.run(
                    run_graph(
                        resume_command,
                        config,
                    )
                )

            except Exception as exc:
                st.error("Something went wrong while finalizing the plan.")

                st.exception(exc)

                final_result = None

        if final_result is not None:
            st.session_state.latest_result = final_result

            st.session_state.waiting_for_approval = "__interrupt__" in final_result

            st.rerun()


# ============================================================
# FINAL RESPONSE
# ============================================================

final_result = st.session_state.get("latest_result")

if final_result and final_result.get("final_response"):
    st.divider()

    section_label("Final Travel Plan")

    final_response = final_result.get(
        "final_response",
        "",
    )

    final_verdict = get_critic_verdict(final_result)

    final_decision = get_decision(final_result)

    human_approved = final_result.get(
        "approved",
        False,
    )

    if human_approved:
        final_stamp = "APPROVED"

    elif final_decision == "REVIEW":
        final_stamp = "REVIEWED"

    else:
        final_stamp = "PLAN"

    render_html(
        f"""
<div class="parchment">

    <div class="parchment-header">

        <span class="parchment-title">
            Final Travel Plan
            &nbsp;·&nbsp;
            {safe_html(st.session_state.thread_id)}
        </span>

        <span class="parchment-stamp">
            {safe_html(final_stamp)}
        </span>

    </div>

    <div style="white-space:pre-wrap;
                line-height:1.65;
                color:var(--parchment-ink);
                overflow-wrap:anywhere;">

        {safe_multiline_html(final_response)}

    </div>

</div>
"""
    )

    # ========================================================
    # FINAL QUALITY SUMMARY
    # ========================================================

    final_quality = final_verdict.get("quality_score")

    final_confidence = final_verdict.get("confidence")

    final_iteration = get_iteration(final_result)

    summary_parts = [
        f"Decision: {final_decision}",
        (
            f"Quality: {final_quality:.0f}/100"
            if isinstance(
                final_quality,
                (int, float),
            )
            else "Quality: N/A"
        ),
        (
            f"Confidence: {final_confidence:.0%}"
            if isinstance(
                final_confidence,
                (int, float),
            )
            else "Confidence: N/A"
        ),
        f"Iterations: {final_iteration}",
    ]

    render_html(
        f"""
<div style="margin-top:12px;
            color:#838C9E;
            font-family:'IBM Plex Mono',monospace;
            font-size:0.68rem;
            letter-spacing:0.04em;">
    {safe_html(" · ".join(summary_parts))}
</div>
"""
    )

    # ========================================================
    # PDF
    # ========================================================

    pdf_bytes = build_travel_plan_pdf(
        result=final_result,
        user_id=user_id,
        thread_id=st.session_state.thread_id,
        user_query=st.session_state.get(
            "user_query",
            "",
        ),
    )

    dl_col, _ = st.columns([1, 4])

    with dl_col:
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=(f"travel_plan_{st.session_state.thread_id}.pdf"),
            mime="application/pdf",
            use_container_width=True,
        )
