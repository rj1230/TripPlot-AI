from __future__ import annotations

import asyncio
import io
import re
import sys
import uuid
from datetime import datetime
from html import escape

# ============================================================
# WINDOWS EVENT LOOP FIX
# ============================================================
#
# psycopg's async mode (used by AsyncPostgresSaver in graph.py) only
# supports SelectorEventLoop. On Windows, asyncio.run() defaults to
# ProactorEventLoop, which raises:
#
#   InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run
#   in async mode.
#
# This must be set before ANY asyncio.run() call happens anywhere in
# the process, so it's done here at the very top of the entrypoint,
# before other imports that might touch the event loop.
# ============================================================

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import run_graph


# ============================================================
# REPORTLAB
# ============================================================

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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


def initialize_session_state():

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"demo_user_{uuid.uuid4().hex[:8]}"

    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None

    if "waiting_for_approval" not in st.session_state:
        st.session_state.waiting_for_approval = False


initialize_session_state()


# ============================================================
# SAFE HTML HELPERS
# ============================================================


def safe_html(value) -> str:

    if value is None:
        return ""

    return escape(str(value))


def render_html(html: str):
    """
    Render HTML using Streamlit markdown.

    IMPORTANT:
    unsafe_allow_html=True is required for our custom cards.

    FIX:
    Markdown treats any line that starts with 4+ leading spaces
    as a preformatted code block. Our f-string HTML templates are
    written with nested/indented lines for readability, which was
    causing Streamlit's markdown parser to render the raw tags as
    literal text instead of parsing them as HTML. We strip leading
    whitespace from every line before handing it to st.markdown.
    Indentation carries no meaning in HTML, so this is always safe.
    """

    lines = html.split("\n")
    dedented = "\n".join(line.lstrip() for line in lines)

    st.markdown(
        dedented,
        unsafe_allow_html=True,
    )


def section_label(text: str):
    """Render a section heading with the flight-path dashed rule."""

    render_html(
        f"""
<div class="section-label-row">
    <div class="section-label">{safe_html(text)}</div>
    <div class="section-rule"></div>
</div>
"""
    )


# ============================================================
# PDF HELPERS
# ============================================================


def _clean_for_pdf(text: str) -> str:

    if not text:
        return ""

    text = re.sub(
        r"<[^>]+>",
        "",
        str(text),
    )

    return text.strip()


def _markdown_to_flowables(
    text: str,
    styles,
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

    bullet_buffer = []

    def flush_bullets():

        if bullet_buffer:
            for bullet in bullet_buffer:
                flowables.append(
                    Paragraph(
                        f"&bull;&nbsp;&nbsp;{bullet}",
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

        line = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            line,
        )

        line = re.sub(
            r"__(.+?)__",
            r"<b>\1</b>",
            line,
        )

        if line.startswith("### "):
            flush_bullets()

            flowables.append(
                Paragraph(
                    line[4:],
                    styles["PlanH3"],
                )
            )

        elif line.startswith("## "):
            flush_bullets()

            flowables.append(
                Paragraph(
                    line[3:],
                    styles["PlanH2"],
                )
            )

        elif line.startswith("# "):
            flush_bullets()

            flowables.append(
                Paragraph(
                    line[2:],
                    styles["PlanH1"],
                )
            )

        elif line.startswith(
            (
                "- ",
                "* ",
            )
        ):
            bullet_buffer.append(line[2:])

        elif re.match(
            r"^\d+\.\s",
            line,
        ):
            flush_bullets()

            flowables.append(
                Paragraph(
                    line,
                    styles["PlanBullet"],
                )
            )

        else:
            flush_bullets()

            flowables.append(
                Paragraph(
                    line,
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

    story.append(
        Paragraph(
            "Your Travel Plan",
            styles["TitleMain"],
        )
    )

    story.append(
        Paragraph(
            (
                f"Generated "
                f"{datetime.now().strftime('%d %b %Y, %I:%M %p')} "
                f"&nbsp;|&nbsp; User: {safe_html(user_id)} "
                f"&nbsp;|&nbsp; Thread: {safe_html(thread_id)}"
            ),
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
            _clean_for_pdf(user_query) or "—",
            styles["PlanBody"],
        )
    )

    # ========================================================
    # AGENTS
    # ========================================================

    selected_agents = result.get(
        "selected_agents",
        [],
    )

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
            if key in selected_agents:
                meta = AGENT_META[key]

                rows.append(
                    [
                        meta["label"],
                        meta["desc"],
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
            (
                "Generated by Multi-Agent Travel Planner — "
                "Supervisor + Guardrails + Human-in-the-Loop"
            ),
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

/* ------------------------------------------------------------
   TYPOGRAPHY — scoped to actual text elements only.
   Previously this used a wildcard on [data-testid="stSidebar"] *
   and .stApp span / .stApp div, which also caught Streamlit's
   built-in Material Symbols icon elements (the sidebar collapse
   arrow, etc). Those icons work by ligature: the literal text
   "keyboard_double_arrow_right" is shown as a glyph ONLY when
   rendered in the Material Symbols font. Overriding font-family
   on those elements broke the ligature and printed the raw text
   instead of the arrow icon (visible as "ouble_arrow_right" in
   the top-left of the app). Fix: don't touch stIconMaterial.
------------------------------------------------------------ */

html, body, .stApp, .stApp p, .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div:not(:has([data-testid="stIconMaterial"])),
[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]) {
    font-family: 'Inter', -apple-system, sans-serif;
}

[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Theme the sidebar collapse/expand control to match the brass
   accent instead of leaving Streamlit's default grey. */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    color: var(--slate);
    transition: color 0.15s ease;
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
    color: var(--brass);
}

.stApp {
    background:
        radial-gradient(circle at 12% -6%, rgba(201,162,75,0.07) 0%, transparent 40%),
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

/* ---------------------------------------------------------- */
/* HERO — departure board                                      */
/* ---------------------------------------------------------- */

.hero-wrap {
    padding: 30px 34px 24px 34px;
    border-radius: 4px;
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-top: 3px solid var(--brass);
    margin-bottom: 30px;
    position: relative;
}

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--brass);
    margin: 0 0 12px 0;
}

.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.5rem;
    font-weight: 600;
    font-optical-sizing: auto;
    color: var(--text);
    margin: 0 0 10px 0;
    letter-spacing: -0.01em;
}

.hero-sub {
    color: var(--slate);
    font-size: 0.98rem;
    line-height: 1.6;
    margin: 0 0 20px 0;
    max-width: 640px;
}

.ticket-row {
    display: flex;
    gap: 0;
    border-top: 1px dashed var(--hairline-strong);
    padding-top: 14px;
    flex-wrap: wrap;
}

.ticket-field {
    padding: 0 22px 0 0;
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

/* ---------------------------------------------------------- */
/* SECTION LABELS — flight-path divider                        */
/* ---------------------------------------------------------- */

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
    height: 0;
    border-top: 1px dashed var(--hairline-strong);
}

/* ---------------------------------------------------------- */
/* AGENT CARDS — boarding-pass stubs                            */
/* ---------------------------------------------------------- */

.agent-card {
    border-radius: 3px;
    padding: 0;
    border: 1px solid var(--hairline);
    background: var(--panel);
    min-height: 175px;
    overflow: hidden;
    position: relative;
    transition: opacity 0.15s ease;
}

.agent-card.active {
    border-color: var(--hairline-strong);
    border-left: 3px solid var(--accent);
}

.agent-card.inactive {
    opacity: 0.4;
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
    margin: 0;
}

.agent-desc {
    font-size: 0.78rem;
    color: var(--slate);
    margin: 4px 0 0 0;
    line-height: 1.4;
}

.agent-perf {
    position: relative;
    height: 1px;
    border-top: 1px dashed var(--hairline);
    margin: 0 14px;
}

.agent-perf::before, .agent-perf::after {
    content: "";
    position: absolute;
    top: -6px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--ink);
    border: 1px solid var(--hairline);
}

.agent-perf::before { left: -20px; }
.agent-perf::after { right: -20px; }

.agent-barcode {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 16px;
    padding: 10px 14px 12px 14px;
    opacity: 0.5;
}

.agent-barcode span {
    display: block;
    width: 2px;
    background: var(--slate);
}

/* ---------------------------------------------------------- */
/* PANELS                                                       */
/* ---------------------------------------------------------- */

.panel {
    border-radius: 3px;
    border: 1px solid var(--hairline);
    background: var(--panel);
    padding: 0 0 16px 0;
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
    margin: 0 0 14px 0;
}

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

/* ---------------------------------------------------------- */
/* PARCHMENT — itinerary & final plan                           */
/* ---------------------------------------------------------- */

.parchment {
    border-radius: 4px;
    padding: 26px 30px;
    background: var(--parchment);
    border: 1px solid var(--parchment-line);
    box-shadow: 0 14px 34px -20px rgba(0,0,0,0.55);
    color: var(--parchment-ink);
    position: relative;
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
    transform: rotate(-2deg);
}

.parchment .stMarkdown, .parchment p, .parchment li {
    color: var(--parchment-ink) !important;
}

/* ---------------------------------------------------------- */
/* NATIVE WIDGETS                                                */
/* ---------------------------------------------------------- */

.stButton > button, .stDownloadButton > button {
    background: var(--panel-2);
    border: 1px solid var(--hairline-strong);
    color: var(--text);
    border-radius: 3px;
    font-weight: 600;
    font-size: 0.85rem;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--brass);
    color: var(--brass);
}

.stButton > button[kind="primary"] {
    background: var(--coral);
    border-color: var(--coral);
    color: #fff;
}

.stButton > button[kind="primary"]:hover {
    background: #c94a27;
    border-color: #c94a27;
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
<div style="font-family:'IBM Plex Mono', monospace; font-size:0.68rem;
            letter-spacing:0.14em; text-transform:uppercase; color:#C9A24B;
            margin-bottom:2px;">
    Session
</div>
<div style="font-family:'Fraunces', serif; font-size:1.2rem; font-weight:600;
            color:#ECEEF3; margin-bottom:14px;">
    Passenger Details
</div>
"""
    )

    user_id = st.text_input(
        "User ID",
        value="demo_user",
    )

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
<div style="font-family:'IBM Plex Mono', monospace; font-size:0.72rem;
            color:#838C9E; margin-top:10px;">
    THREAD&nbsp;&nbsp;<span style="color:#ECEEF3;">{safe_html(st.session_state.thread_id)}</span>
</div>
"""
    )

    st.divider()

    render_html(
        """
<div style="font-family:'IBM Plex Mono', monospace; font-size:0.68rem;
            letter-spacing:0.14em; text-transform:uppercase; color:#C9A24B;
            margin-bottom:2px;">
    Roster
</div>
<div style="font-family:'Fraunces', serif; font-size:1.2rem; font-weight:600;
            color:#ECEEF3; margin-bottom:14px;">
    Specialist Agents
</div>
"""
    )

    for index, key in enumerate(AGENT_ORDER):
        meta = AGENT_META[key]

        render_html(
            f"""
<div style="display:flex; align-items:center; gap:10px; padding:8px 0;
            border-bottom:1px dashed rgba(201,162,75,0.16);">
    <div style="font-family:'IBM Plex Mono', monospace; font-size:0.62rem;
                color:#838C9E; width:24px;">0{index + 1}</div>
    <div style="font-size:1.05rem;">{meta["icon"]}</div>
    <div>
        <div style="font-weight:600; font-size:0.85rem; color:#ECEEF3;">{safe_html(meta["label"])}</div>
        <div style="font-size:0.72rem; color:#838C9E;">{safe_html(meta["desc"])}</div>
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
        Itinerary Desk &nbsp;·&nbsp; Supervisor-Routed Planning
    </div>

    <div class="hero-title">
        Multi-Agent Travel Planner
    </div>

    <div class="hero-sub">
        Describe your trip. A supervisor agent reads the brief, routes it to
        the specialists it needs, drafts a plan, and holds it for your
        approval before anything is finalized.
    </div>

    <div class="ticket-row">

        <div class="ticket-field">
            <div class="ticket-label">Routing</div>
            <div class="ticket-value">Supervisor-directed</div>
        </div>

        <div class="ticket-field">
            <div class="ticket-label">Review</div>
            <div class="ticket-value">Human-in-the-loop</div>
        </div>

        <div class="ticket-field">
            <div class="ticket-label">Roster</div>
            <div class="ticket-value">5 specialist agents</div>
        </div>

        <div class="ticket-field">
            <div class="ticket-label">Output</div>
            <div class="ticket-value">Plan + downloadable PDF</div>
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

    input_state = {
        "messages": [HumanMessage(content=new_query)],
        "user_id": user_id,
        "user_query": new_query,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0,
    }

    with st.spinner("🧠 Supervisor and specialist agents are planning..."):
        try:
            # IMPORTANT:
            #
            # run_graph() compiles the StateGraph first.
            #
            # We are NOT doing:
            #
            # build_graph().ainvoke(...)
            #
            # because StateGraph has no ainvoke().

            result = asyncio.run(
                run_graph(
                    input_state,
                    config,
                )
            )

        except Exception as exc:
            st.error("Something went wrong while planning your trip:")

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
    selected_agents = result.get(
        "selected_agents",
        [],
    )

    section_label("Supervisor Plan")

    reasoning = result.get(
        "supervisor_reasoning",
        "Supervisor completed routing.",
    )

    render_html(
        f"""
<div class="reasoning-box">
    <div class="reasoning-eyebrow">Routing Note</div>
    <div class="reasoning-text">{safe_html(reasoning)}</div>
</div>
"""
    )

    # ========================================================
    # SELECTED AGENTS
    # ========================================================

    section_label("Selected Agents")

    cols = st.columns(len(AGENT_ORDER))

    for col, index, key in zip(
        cols,
        range(len(AGENT_ORDER)),
        AGENT_ORDER,
    ):
        meta = AGENT_META[key]

        is_active = key in selected_agents

        state_class = "active" if is_active else "inactive"

        status_class = "on" if is_active else "off"

        status_text = "SELECTED" if is_active else "SKIPPED"

        # Decorative barcode bars — purely visual, width varies per agent
        # so cards don't look identically stamped.
        bar_widths = [3, 1, 2, 1, 3, 1, 1, 2, 3, 1, 2, 1]

        barcode_bars = "".join(
            f'<span style="width:{w}px; height:{6 + (i % 3) * 3}px;"></span>'
            for i, w in enumerate(bar_widths)
        )

        with col:
            render_html(
                f"""
<div class="agent-card {state_class}"
     style="--accent:{meta["color"]};">

    <div class="agent-card-top">
        <span class="agent-gate">GATE 0{index + 1}</span>
        <span class="agent-status {status_class}">{status_text}</span>
    </div>

    <div class="agent-card-body">
        <span class="agent-icon">{meta["icon"]}</span>
        <div class="agent-name">{safe_html(meta["label"])}</div>
        <div class="agent-desc">{safe_html(meta["desc"])}</div>
    </div>

    <div class="agent-perf"></div>

    <div class="agent-barcode">{barcode_bars}</div>

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

            body = (
                content
                if content
                else '<span style="color:#838C9E; font-size:0.85rem;">No output yet.</span>'
            )

            with out_cols[index % 2]:
                render_html(
                    f"""
<div class="panel">
    <div class="panel-title">{meta["icon"]} {safe_html(meta["label"])}</div>
    <div style="padding: 0 18px; color: var(--text); font-size: 0.9rem; line-height: 1.6;">

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
        render_html(
            f"""
<div class="parchment">
    <div class="parchment-header">
        <span class="parchment-title">Working Draft &nbsp;·&nbsp; Awaiting Approval</span>
        <span class="parchment-stamp">DRAFT</span>
    </div>

{draft}

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
        with st.spinner("Creating final response..."):
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
                st.error("Something went wrong while finalizing the plan:")

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

    render_html(
        f"""
<div class="parchment">
    <div class="parchment-header">
        <span class="parchment-title">Confirmed Itinerary &nbsp;·&nbsp; {safe_html(st.session_state.thread_id)}</span>
        <span class="parchment-stamp">APPROVED</span>
    </div>

{final_response}

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
