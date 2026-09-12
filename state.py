from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage


# ============================================================
# REUSABLE TYPES
# ============================================================


class EvidenceItem(TypedDict, total=False):
    """
    Evidence returned by an MCP/search/tool call.

    Keeping evidence separate from LLM-written prose allows the
    critic to distinguish grounded information from synthesis.
    """

    source: str
    title: str
    url: str
    snippet: str
    retrieved_at: str
    confidence: float


class AgentResult(TypedDict, total=False):
    """
    Standard result envelope for specialist agents.

    This allows future agents to return:
        - human-readable output
        - structured data
        - evidence
        - confidence
        - warnings
        - execution metadata
    """

    status: str
    summary: str
    data: dict[str, Any]
    evidence: list[EvidenceItem]
    warnings: list[str]
    assumptions: list[str]
    confidence: float
    source: str


class QualityScore(TypedDict, total=False):
    """
    Structured quality dimensions produced by the critic.
    """

    overall: float
    constraint_satisfaction: float
    evidence_grounding: float
    budget_consistency: float
    itinerary_coherence: float
    safety: float


class CriticVerdict(TypedDict, total=False):
    """
    Structured self-correction decision.
    """

    passed: bool
    decision: str

    overall_score: float

    violations: list[str]
    warnings: list[str]

    responsible_agents: list[str]

    quality: QualityScore

    reasoning: str


# ============================================================
# TRAVEL STATE
# ============================================================


class TravelState(TypedDict, total=False):
    # ========================================================
    # CONVERSATION
    # ========================================================

    messages: Annotated[
        list[AnyMessage],
        operator.add,
    ]

    user_id: str
    user_query: str

    # ========================================================
    # PLANNING / SUPERVISOR
    # ========================================================

    trip_constraints: dict[str, Any]

    selected_agents: list[str]

    supervisor_reasoning: str

    # ========================================================
    # SPECIALIST OUTPUTS
    # ========================================================

    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str

    # ========================================================
    # STRUCTURED SPECIALIST DATA
    #
    # These fields allow the system to evolve from:
    #
    #     "agent writes prose"
    #
    # into:
    #
    #     "agent produces validated data + prose + evidence"
    # ========================================================

    flight_analysis: dict[str, Any]
    hotel_analysis: dict[str, Any]
    weather_analysis: dict[str, Any]

    budget_analysis: dict[str, Any]

    # ========================================================
    # EVIDENCE / PROVENANCE
    #
    # Critical for production-grade Agentic RAG-style grounding.
    # ========================================================

    flight_evidence: list[EvidenceItem]
    hotel_evidence: list[EvidenceItem]
    weather_evidence: list[EvidenceItem]
    budget_evidence: list[EvidenceItem]

    # ========================================================
    # AGENT QUALITY
    # ========================================================

    agent_confidence: dict[str, float]

    agent_warnings: Annotated[
        list[str],
        operator.add,
    ]

    agent_errors: Annotated[
        list[str],
        operator.add,
    ]

    # ========================================================
    # FINAL ITINERARY
    # ========================================================

    itinerary: str

    approval_request: str

    # ========================================================
    # HUMAN APPROVAL
    # ========================================================

    human_feedback: str

    approved: bool

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    final_response: str

    # ========================================================
    # OBSERVABILITY
    # ========================================================

    # IMPORTANT:
    #
    # This should NOT be incremented directly by multiple parallel
    # agents as:
    #
    #     llm_calls = state["llm_calls"] + 1
    #
    # because concurrent writes can overwrite each other.
    #
    # Use llm_call_events instead.
    llm_call_events: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    tool_call_events: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    trace_id: str

    # ========================================================
    # REPLANNING
    # ========================================================

    iteration_count: int

    critic_verdict: CriticVerdict

    unresolved_violations: list[str]

    # True only while supervisor is entering a targeted replan.
    is_replan: bool

    # ========================================================
    # EXECUTION STATUS
    # ========================================================

    execution_status: str

    degraded_mode: bool

    # ========================================================
    # FINAL QUALITY
    # ========================================================

    quality_score: QualityScore
