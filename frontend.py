from dotenv import load_dotenv

load_dotenv()
import os

import io
import re
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Agent metadata — single source of truth for icons / labels / colors
# ──────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────
# PDF generation
# ──────────────────────────────────────────────────────────────────────────
def _clean_for_pdf(text: str) -> str:
    """Strip raw HTML and convert light markdown (**bold**, # headers, - bullets)
    into a plain string list of (kind, content) blocks reportlab can render."""
    if not text:
        return ""
    # Drop any stray HTML tags that may have leaked from the Streamlit markdown panels
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _markdown_to_flowables(text: str, styles):
    """Very small markdown -> reportlab Paragraph converter.
    Handles #/##/### headers, **bold**, and -/* bullet lists — enough for
    typical LLM-generated itinerary/plan text."""
    flowables = []
    text = _clean_for_pdf(text)
    if not text:
        flowables.append(Paragraph("<i>No content available.</i>", styles["PlanBody"]))
        return flowables

    lines = text.split("\n")
    bullet_buffer = []

    def flush_bullets():
        if bullet_buffer:
            for b in bullet_buffer:
                flowables.append(
                    Paragraph(f"&bull;&nbsp;&nbsp;{b}", styles["PlanBullet"])
                )
            bullet_buffer.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_bullets()
            flowables.append(Spacer(1, 4))
            continue

        # Bold: **text** -> <b>text</b>
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"__(.+?)__", r"<b>\1</b>", line)
        # Italic: *text* -> <i>text</i>
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", line)

        if line.startswith("### "):
            flush_bullets()
            flowables.append(Paragraph(line[4:], styles["PlanH3"]))
        elif line.startswith("## "):
            flush_bullets()
            flowables.append(Paragraph(line[3:], styles["PlanH2"]))
        elif line.startswith("# "):
            flush_bullets()
            flowables.append(Paragraph(line[2:], styles["PlanH1"]))
        elif line.startswith(("- ", "* ")):
            bullet_buffer.append(line[2:])
        elif re.match(r"^\d+\.\s", line):
            flush_bullets()
            flowables.append(Paragraph(line, styles["PlanBullet"]))
        else:
            flush_bullets()
            flowables.append(Paragraph(line, styles["PlanBody"]))

    flush_bullets()
    return flowables


def build_travel_plan_pdf(
    result: dict, user_id: str, thread_id: str, user_query: str
) -> bytes:
    """Builds a formatted PDF of the trip plan (query, agent summary, itinerary,
    final plan) and returns it as raw bytes for st.download_button."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
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

    # ---- Title block ----
    story.append(Paragraph("Your Travel Plan", styles["TitleMain"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')} &nbsp;|&nbsp; "
            f"User: {user_id} &nbsp;|&nbsp; Thread: {thread_id}",
            styles["Subtitle"],
        )
    )
    story.append(
        HRFlowable(width="100%", color=colors.HexColor("#D8DCE6"), thickness=1)
    )
    story.append(Spacer(1, 10))

    # ---- Trip request ----
    story.append(Paragraph("Trip Request", styles["SectionHeader"]))
    story.append(Paragraph(_clean_for_pdf(user_query) or "—", styles["PlanBody"]))

    # ---- Agents involved ----
    selected_agents = result.get("selected_agents", [])
    if selected_agents:
        story.append(Paragraph("Agents Involved", styles["SectionHeader"]))
        rows = [["Agent", "Role"]]
        for key in AGENT_ORDER:
            if key in selected_agents:
                meta = AGENT_META[key]
                # Emoji glyphs aren't in ReportLab's default fonts, so use plain text here
                rows.append([meta["label"], meta["desc"]])
        tbl = Table(rows, colWidths=[2.0 * inch, 4.3 * inch])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF1F8")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2E3A59")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E5EC")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(tbl)

    # ---- Draft itinerary ----
    itinerary_text = result.get("itinerary", "")
    if itinerary_text:
        story.append(Paragraph("Itinerary", styles["SectionHeader"]))
        story.extend(_markdown_to_flowables(itinerary_text, styles))

    # ---- Final plan ----
    final_text = result.get("final_response", "")
    if final_text:
        story.append(Paragraph("Final Travel Plan", styles["SectionHeader"]))
        story.extend(_markdown_to_flowables(final_text, styles))

    story.append(Spacer(1, 16))
    story.append(
        HRFlowable(width="100%", color=colors.HexColor("#D8DCE6"), thickness=1)
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Generated by Multi-Agent Travel Planner — Supervisor + Guardrails + Human-in-the-Loop",
            styles["MetaLabel"],
        )
    )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ──────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #12141c 0%, #0a0b10 55%, #0a0b10 100%);
    }

    /* ---- Hero header ---- */
    .hero-wrap {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(91,141,239,0.14), rgba(176,132,240,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 28px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F5F6FA;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #9AA0AE;
        font-size: 0.98rem;
        margin: 0;
    }
    .pill-row { display:flex; gap:8px; margin-top:14px; flex-wrap: wrap; }
    .pill {
        font-size: 0.72rem; font-weight: 600; letter-spacing:.02em;
        padding: 5px 12px; border-radius: 999px;
        background: rgba(255,255,255,0.06);
        color: #C7CBD6; border: 1px solid rgba(255,255,255,0.08);
    }

    /* ---- Section labels ---- */
    .section-label {
        font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #7C8290; margin: 4px 0 12px 2px;
    }

    /* ---- Agent selector cards ---- */
    .agent-card {
        border-radius: 14px;
        padding: 16px 16px 14px 16px;
        border: 1px solid rgba(255,255,255,0.07);
        background: #14161e;
        height: 100%;
        transition: all .15s ease;
        position: relative;
    }
    .agent-card.active {
        border-color: var(--accent);
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
        box-shadow: 0 0 0 1px var(--accent) inset, 0 8px 20px -12px var(--accent);
    }
    .agent-card.inactive { opacity: 0.42; filter: grayscale(0.35); }
    .agent-icon {
        font-size: 1.5rem; display:inline-block; margin-bottom: 8px;
    }
    .agent-name { font-weight: 700; font-size: 0.95rem; color: #F0F1F5; margin: 0; }
    .agent-desc { font-size: 0.78rem; color: #8B90A0; margin: 2px 0 10px 0; }
    .agent-status {
        font-size: 0.68rem; font-weight: 700; letter-spacing: .04em;
        padding: 3px 9px; border-radius: 999px; display:inline-block;
    }
    .agent-status.on { background: rgba(79,195,161,0.15); color: #6FE3BF; }
    .agent-status.off { background: rgba(255,255,255,0.05); color: #6B7080; }

    /* ---- Generic panel card ---- */
    .panel {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
        background: #14161e;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .panel-title {
        font-weight: 700; font-size: 0.95rem; color: #F0F1F5;
        display:flex; align-items:center; gap:8px; margin-bottom: 10px;
    }

    .reasoning-box {
        border-left: 3px solid #5B8DEF;
        background: rgba(91,141,239,0.06);
        padding: 12px 16px; border-radius: 0 10px 10px 0;
        color: #C7CBD6; font-size: 0.9rem; line-height: 1.55;
    }

    .final-plan {
        border-radius: 16px;
        padding: 22px 24px;
        background: linear-gradient(135deg, rgba(79,195,161,0.10), rgba(91,141,239,0.06));
        border: 1px solid rgba(79,195,161,0.25);
    }

    div[data-testid="stStatusWidget"] { display:none; }

    /* ---- Download button ---- */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #4FC3A1, #5B8DEF);
        color: #0A0B10;
        font-weight: 700;
        border: none;
        border-radius: 10px;
    }
    div[data-testid="stDownloadButton"] button:hover {
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — session controls
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Session")
    user_id = st.text_input("User ID", value="demo_user")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"

    if st.button("➕ New Thread", use_container_width=True):
        st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        st.session_state.pop("waiting_for_approval", None)
        st.session_state.pop("latest_result", None)
        st.rerun()

    st.caption(f"Thread: `{st.session_state.thread_id}`")

    st.divider()
    st.markdown("### 🤖 Agent Roster")
    for key in AGENT_ORDER:
        meta = AGENT_META[key]
        st.markdown(f"{meta['icon']} **{meta['label']}** — {meta['desc']}")

# ──────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-wrap">
        <p class="hero-title">Real-World Multi-Agent Travel Planner</p>
        <p class="hero-sub">Describe your trip — a supervisor agent routes the request to specialist agents, drafts a plan, and waits for your approval.</p>
        <div class="pill-row">
            <span class="pill">🧠 Supervisor-routed</span>
            <span class="pill">🔁 Human-in-the-loop</span>
            <span class="pill">🧩 5 specialist agents</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Request input
# ──────────────────────────────────────────────────────────────────────────
query = st.text_area(
    "Travel request",
    placeholder="Plan a 7-day Japan trip under Rs. 2 lakh. I prefer budget hotels and no overnight flights.",
    height=110,
)

col_btn, col_spacer = st.columns([1, 4])
with col_btn:
    submit = st.button("🚀 Create Draft Plan", type="primary", use_container_width=True)

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if submit:
    if not query.strip():
        st.warning("Enter a travel request first.")
    else:
        st.session_state.user_query = query
        with st.spinner("Agents are planning..."):
            result = app.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "user_id": user_id,
                    "user_query": query,
                    "flight_results": "",
                    "hotel_results": "",
                    "weather_results": "",
                    "budget_results": "",
                    "itinerary": "",
                    "final_response": "",
                    "llm_calls": 0,
                },
                config=config,
            )

        st.session_state.latest_result = result
        st.session_state.waiting_for_approval = "__interrupt__" in result

result = st.session_state.get("latest_result")

# ──────────────────────────────────────────────────────────────────────────
# Supervisor plan + agent selector
# ──────────────────────────────────────────────────────────────────────────
if result:
    selected_agents = result.get("selected_agents", [])

    st.markdown('<p class="section-label">Supervisor Plan</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="reasoning-box">{result.get("supervisor_reasoning", "")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-label" style="margin-top:22px;">Selected Agents</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(AGENT_ORDER))
    for col, key in zip(cols, AGENT_ORDER):
        meta = AGENT_META[key]
        is_active = key in selected_agents
        state_class = "active" if is_active else "inactive"
        status_class = "on" if is_active else "off"
        status_text = "SELECTED" if is_active else "SKIPPED"
        with col:
            st.markdown(
                f"""
                <div class="agent-card {state_class}" style="--accent:{meta["color"]};">
                    <span class="agent-icon">{meta["icon"]}</span>
                    <p class="agent-name">{meta["label"]}</p>
                    <p class="agent-desc">{meta["desc"]}</p>
                    <span class="agent-status {status_class}">{status_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Agent outputs, only for agents that actually ran ----
    st.markdown(
        '<p class="section-label" style="margin-top:26px;">Agent Outputs</p>',
        unsafe_allow_html=True,
    )
    output_agents = [
        a for a in AGENT_ORDER if a != "itinerary_agent" and a in selected_agents
    ]
    if output_agents:
        out_cols = st.columns(2)
        for i, key in enumerate(output_agents):
            meta = AGENT_META[key]
            content = result.get(RESULT_KEY_FOR_AGENT[key], "") or "_No output yet._"
            with out_cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-title">{meta["icon"]} {meta["label"]}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(content)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("No specialist agents were selected for this request.")

    # ---- Draft itinerary ----
    st.markdown(
        '<p class="section-label" style="margin-top:6px;">Draft Itinerary</p>',
        unsafe_allow_html=True,
    )
    if "__interrupt__" in result:
        draft = result["__interrupt__"][0].value.get("draft_itinerary", "")
    else:
        draft = result.get("itinerary", "")
    st.markdown(f'<div class="panel">{draft}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Human approval
# ──────────────────────────────────────────────────────────────────────────
if st.session_state.get("waiting_for_approval"):
    st.divider()
    st.markdown('<p class="section-label">Human Approval</p>', unsafe_allow_html=True)

    approved = st.radio(
        "Approve this draft?", ["Yes", "No, revise it"], horizontal=True
    )
    feedback = st.text_area("Feedback", disabled=approved == "Yes")

    if st.button("✅ Submit Approval", type="primary"):
        with st.spinner("Creating final response..."):
            final_result = app.invoke(
                Command(
                    resume={
                        "approved": approved == "Yes",
                        "feedback": feedback,
                    }
                ),
                config=config,
            )
        st.session_state.latest_result = final_result
        st.session_state.waiting_for_approval = False
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Final plan
# ──────────────────────────────────────────────────────────────────────────
final_result = st.session_state.get("latest_result")
if final_result and final_result.get("final_response"):
    st.divider()
    st.markdown(
        '<p class="section-label">Final Travel Plan</p>', unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="final-plan">{final_result["final_response"]}</div>',
        unsafe_allow_html=True,
    )

    # ---- PDF download ----
    pdf_bytes = build_travel_plan_pdf(
        result=final_result,
        user_id=user_id,
        thread_id=st.session_state.thread_id,
        user_query=st.session_state.get("user_query", query),
    )
    dl_col, _ = st.columns([1, 4])
    with dl_col:
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=f"travel_plan_{st.session_state.thread_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
