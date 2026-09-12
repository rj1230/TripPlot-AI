from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

REPLANNABLE_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

VALID_RESPONSIBLE_AGENTS = REPLANNABLE_AGENTS

MAX_VIOLATIONS = 20
MAX_SUGGESTIONS = 20

QUALITY_PASS_THRESHOLD = 75
QUALITY_REVIEW_THRESHOLD = 60


# ============================================================
# CRITIC VERDICT
# ============================================================


@dataclass
class CriticVerdict:
    passed: bool

    decision: str = "REVIEW"

    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    responsible_agents: list[str] = field(default_factory=list)

    # Overall quality
    quality_score: float = 0.0
    confidence: float = 0.0

    # Dimension scores
    constraint_score: float = 0.0
    budget_score: float = 0.0
    routing_score: float = 0.0
    itinerary_score: float = 0.0
    evidence_score: float = 0.0
    safety_score: float = 0.0

    # Execution metadata
    deterministic_checks_passed: bool = False
    llm_check_passed: bool = False
    degraded_mode: bool = False

    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "violations": self.violations,
            "suggestions": self.suggestions,
            "responsible_agents": self.responsible_agents,
            "quality_score": round(self.quality_score, 2),
            "confidence": round(self.confidence, 2),
            "scores": {
                "constraint": round(self.constraint_score, 2),
                "budget": round(self.budget_score, 2),
                "routing": round(self.routing_score, 2),
                "itinerary": round(self.itinerary_score, 2),
                "evidence": round(self.evidence_score, 2),
                "safety": round(self.safety_score, 2),
            },
            "deterministic_checks_passed": self.deterministic_checks_passed,
            "llm_check_passed": self.llm_check_passed,
            "degraded_mode": self.degraded_mode,
            "reasoning": self.reasoning,
        }


# ============================================================
# HELPERS
# ============================================================


def _safe_float(value: Any) -> float | None:
    """
    Convert common numeric/currency representations to float.

    Examples:
        100
        "100"
        "₹1,50,000"
        "INR 150000"
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.]", "", value)

        if not cleaned or cleaned == ".":
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


def _normalise_agent_names(
    agents: list[str] | None,
) -> list[str]:
    if not agents:
        return []

    result: list[str] = []

    for agent in agents:
        if not isinstance(agent, str):
            continue

        agent = agent.strip()

        if agent in VALID_RESPONSIBLE_AGENTS:
            result.append(agent)

    return list(dict.fromkeys(result))


def _normalise_list(
    value: Any,
    limit: int = 20,
) -> list[str]:

    if value is None:
        return []

    if not isinstance(value, list):
        value = [value]

    result = []

    for item in value[:limit]:
        text = str(item).strip()

        if text:
            result.append(text)

    return result


def _clamp_score(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    return max(0.0, min(100.0, score))


def _parse_date(value: Any) -> date | None:
    if not value:
        return None

    text = str(value).strip()

    formats = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    )

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort JSON extraction.

    This remains as a compatibility fallback.

    The preferred path is structured LLM output when the configured
    model supports it.
    """

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Critic LLM did not return a JSON object.")

    raw = text[start : end + 1]

    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Critic response must be a JSON object.")

    return data


# ============================================================
# DETERMINISTIC CHECKS
# ============================================================


def rule_based_checks(
    plan: dict[str, Any],
    constraints: dict[str, Any],
) -> CriticVerdict:

    violations: list[str] = []
    suggestions: list[str] = []
    responsible: list[str] = []

    constraint_score = 100.0
    budget_score = 100.0
    routing_score = 100.0
    itinerary_score = 100.0
    evidence_score = 100.0
    safety_score = 100.0

    # ========================================================
    # REQUIRED OUTPUTS
    # ========================================================

    flight = plan.get("flight_results")
    hotel = plan.get("hotel_results")
    itinerary = plan.get("itinerary")

    if not flight or not str(flight).strip():
        violations.append("Missing flight planning information.")
        responsible.append("flight_agent")
        constraint_score -= 20
        evidence_score -= 10

    if not hotel or not str(hotel).strip():
        violations.append("Missing hotel/accommodation planning information.")
        responsible.append("hotel_agent")
        constraint_score -= 20
        evidence_score -= 10

    if not itinerary or not str(itinerary).strip():
        violations.append("Missing itinerary.")
        responsible.append("itinerary_agent")
        itinerary_score = 0

    # ========================================================
    # BUDGET
    # ========================================================

    budget_value = constraints.get("budget")
    budget = _safe_float(budget_value)

    budget_analysis = plan.get("budget_analysis") or {}

    total_cost = _safe_float(budget_analysis.get("total_cost"))

    if total_cost is None:
        total_cost = _safe_float(plan.get("total_cost"))

    if budget is not None and total_cost is not None:
        if total_cost > budget:
            violations.append(
                f"Budget exceeded: estimated total cost "
                f"{total_cost:.2f} > budget {budget:.2f}."
            )

            suggestions.append(
                "Reduce flight, hotel, activity, or transportation costs."
            )

            responsible.extend(
                [
                    "budget_agent",
                    "itinerary_agent",
                ]
            )

            budget_score = max(
                0,
                100 - ((total_cost - budget) / budget) * 100,
            )

        else:
            # Reward reasonable headroom.
            ratio = total_cost / budget

            if ratio >= 0.95:
                suggestions.append(
                    "Budget is very close to the user's maximum; "
                    "consider keeping a contingency buffer."
                )

                budget_score = 85

    elif budget is not None and total_cost is None:
        suggestions.append(
            "Budget was provided, but a reliable total-cost estimate "
            "could not be verified."
        )

        budget_score = 65
        evidence_score -= 15

    # ========================================================
    # BUDGET CURRENCY
    # ========================================================

    requested_currency = constraints.get("currency")

    estimated_currency = budget_analysis.get("currency")

    if (
        requested_currency
        and estimated_currency
        and str(requested_currency).upper() != str(estimated_currency).upper()
    ):
        violations.append(
            "Budget currency is inconsistent with the requested currency."
        )

        responsible.append("budget_agent")
        budget_score -= 20

    # ========================================================
    # DATES
    # ========================================================

    dates = constraints.get("dates") or {}

    start = _parse_date(dates.get("start"))
    end = _parse_date(dates.get("end"))

    if start and end:
        if end < start:
            violations.append("Trip end date precedes trip start date.")

            responsible.append("itinerary_agent")
            constraint_score -= 30

        elif start == end:
            suggestions.append("The trip spans one calendar day.")

    # ========================================================
    # DURATION
    # ========================================================

    duration = constraints.get("duration")

    if duration is not None:
        try:
            duration_number = int(duration)

            if duration_number <= 0:
                violations.append("Trip duration must be greater than zero.")

                responsible.append("itinerary_agent")
                constraint_score -= 25

        except (TypeError, ValueError):
            suggestions.append(
                "Trip duration could not be deterministically validated."
            )

            constraint_score -= 5

    # ========================================================
    # SAFETY / BOOKING CLAIMS
    # ========================================================

    combined_text = " ".join(
        str(plan.get(key, ""))
        for key in (
            "flight_results",
            "hotel_results",
            "weather_results",
            "budget_results",
            "itinerary",
        )
    ).lower()

    forbidden_confirmations = (
        "booking confirmed",
        "ticket confirmed",
        "hotel booked",
        "reservation confirmed",
        "guaranteed availability",
    )

    for phrase in forbidden_confirmations:
        if phrase in combined_text:
            violations.append(
                "Plan contains an unsupported booking/availability claim."
            )

            safety_score -= 30
            evidence_score -= 20
            responsible.append("itinerary_agent")

            break

    # ========================================================
    # WEATHER
    # ========================================================

    weather = plan.get("weather_results")

    if constraints.get("destination") and not weather:
        suggestions.append(
            "Weather information is unavailable; "
            "consider adding destination-specific weather context."
        )

        evidence_score -= 5

    # ========================================================
    # FINAL DETERMINISTIC SCORE
    # ========================================================

    responsible = _normalise_agent_names(responsible)

    violations = list(dict.fromkeys(violations))[:MAX_VIOLATIONS]
    suggestions = list(dict.fromkeys(suggestions))[:MAX_SUGGESTIONS]

    passed = not violations

    quality_score = (
        constraint_score
        + budget_score
        + routing_score
        + itinerary_score
        + evidence_score
        + safety_score
    ) / 6

    return CriticVerdict(
        passed=passed,
        decision="PASS" if passed else "REPLAN",
        violations=violations,
        suggestions=suggestions,
        responsible_agents=responsible,
        quality_score=quality_score,
        confidence=0.95,
        constraint_score=constraint_score,
        budget_score=budget_score,
        routing_score=routing_score,
        itinerary_score=itinerary_score,
        evidence_score=evidence_score,
        safety_score=safety_score,
        deterministic_checks_passed=passed,
        llm_check_passed=False,
        degraded_mode=False,
        reasoning=(
            "Deterministic validation completed."
            if passed
            else "Deterministic validation found hard violations."
        ),
    )


# ============================================================
# LLM FUZZY CHECK
# ============================================================


def llm_fuzzy_check(
    plan: dict[str, Any],
    constraints: dict[str, Any],
    llm,
) -> CriticVerdict:

    prompt = f"""
You are the final quality-control critic for a production travel
planning system.

Evaluate the proposed plan against the user's constraints.

CONSTRAINTS:
{json.dumps(constraints, indent=2, default=str)}

PLAN:
{json.dumps(plan, indent=2, default=str)}

Evaluate these dimensions from 0 to 100:

1. constraint_score
   - destination
   - dates
   - duration
   - preferences
   - explicit user requirements

2. budget_score
   - budget consistency
   - reasonable estimates
   - currency consistency
   - contingency

3. routing_score
   - geographic coherence
   - unnecessary backtracking
   - realistic transportation

4. itinerary_score
   - daily pacing
   - practical sequencing
   - sufficient rest
   - realistic activities

5. evidence_score
   - whether claims are supported by available tool/agent information
   - no fabricated live availability
   - no fabricated confirmed bookings

6. safety_score
   - no unsupported guarantees
   - no dangerous or clearly impractical recommendations
   - no misleading certainty

Rules:

- Be strict but reasonable.
- Estimates are acceptable when clearly presented as estimates.
- Never treat an estimate as confirmed availability.
- Do not invent facts that are absent from the plan.
- Do not mark a plan as failed for minor stylistic imperfections.
- If there is a meaningful problem, identify it.
- Responsible agents must contain only valid graph agent names.
- Prefer the smallest set of responsible agents necessary to fix the issue.

Allowed agents:

[
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
]

Return ONLY valid JSON:

{{
    "passed": true,
    "violations": [],
    "suggestions": [],
    "responsible_agents": [],
    "constraint_score": 0,
    "budget_score": 0,
    "routing_score": 0,
    "itinerary_score": 0,
    "evidence_score": 0,
    "safety_score": 0,
    "confidence": 0,
    "reasoning": ""
}}
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict production travel-plan critic. Return JSON only."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    content = response.content

    if not isinstance(content, str):
        content = str(content)

    data = _extract_json_object(content)

    passed = data.get("passed")

    if not isinstance(passed, bool):
        raise ValueError("Critic verdict 'passed' must be boolean.")

    violations = _normalise_list(data.get("violations"))

    suggestions = _normalise_list(data.get("suggestions"))

    responsible = _normalise_agent_names(data.get("responsible_agents"))

    constraint_score = _clamp_score(
        data.get("constraint_score"),
        75,
    )

    budget_score = _clamp_score(
        data.get("budget_score"),
        75,
    )

    routing_score = _clamp_score(
        data.get("routing_score"),
        75,
    )

    itinerary_score = _clamp_score(
        data.get("itinerary_score"),
        75,
    )

    evidence_score = _clamp_score(
        data.get("evidence_score"),
        75,
    )

    safety_score = _clamp_score(
        data.get("safety_score"),
        90,
    )

    confidence = _clamp_score(
        data.get("confidence"),
        70,
    )

    quality_score = (
        constraint_score
        + budget_score
        + routing_score
        + itinerary_score
        + evidence_score
        + safety_score
    ) / 6

    # A failed verdict cannot become PASS merely because
    # the numerical score happens to be high.
    if violations:
        passed = False

    if passed and quality_score >= QUALITY_PASS_THRESHOLD:
        decision = "PASS"

    elif quality_score >= QUALITY_REVIEW_THRESHOLD:
        decision = "REVIEW"

    else:
        decision = "REPLAN"

    return CriticVerdict(
        passed=passed and decision == "PASS",
        decision=decision,
        violations=violations,
        suggestions=suggestions,
        responsible_agents=responsible,
        quality_score=quality_score,
        confidence=confidence,
        constraint_score=constraint_score,
        budget_score=budget_score,
        routing_score=routing_score,
        itinerary_score=itinerary_score,
        evidence_score=evidence_score,
        safety_score=safety_score,
        deterministic_checks_passed=True,
        llm_check_passed=True,
        degraded_mode=False,
        reasoning=str(
            data.get(
                "reasoning",
                "",
            )
        ),
    )


# ============================================================
# MAIN CRITIC
# ============================================================


def run_critic(
    plan: dict[str, Any],
    constraints: dict[str, Any],
    llm,
) -> CriticVerdict:

    # --------------------------------------------------------
    # STEP 1 — deterministic validation
    # --------------------------------------------------------

    rule_verdict = rule_based_checks(
        plan,
        constraints,
    )

    if not rule_verdict.passed:
        logger.warning(
            "Deterministic critic failure: %s",
            rule_verdict.violations,
        )

        return rule_verdict

    # --------------------------------------------------------
    # STEP 2 — semantic LLM validation
    # --------------------------------------------------------

    try:
        llm_verdict = llm_fuzzy_check(
            plan,
            constraints,
            llm,
        )

        return llm_verdict

    except Exception as exc:
        logger.exception(
            "LLM critic failed: %s",
            exc,
        )

        # ----------------------------------------------------
        # FAIL SAFE
        # ----------------------------------------------------
        #
        # Never allow critic failure to silently pass the plan.
        #

        return CriticVerdict(
            passed=False,
            decision="REVIEW",
            violations=["Critic validation failed unexpectedly."],
            suggestions=["Review the itinerary manually before approval."],
            responsible_agents=["itinerary_agent"],
            quality_score=50.0,
            confidence=0.20,
            constraint_score=50.0,
            budget_score=50.0,
            routing_score=50.0,
            itinerary_score=50.0,
            evidence_score=30.0,
            safety_score=70.0,
            deterministic_checks_passed=True,
            llm_check_passed=False,
            degraded_mode=True,
            reasoning=(
                "The semantic critic was unavailable. "
                "The plan was not automatically approved."
            ),
        )


# ============================================================
# GRAPH NODE
# ============================================================


def critic_node(
    state: dict[str, Any],
    llm,
) -> dict[str, Any]:

    plan = {
        "flight_results": state.get(
            "flight_results",
            "",
        ),
        "hotel_results": state.get(
            "hotel_results",
            "",
        ),
        "weather_results": state.get(
            "weather_results",
            "",
        ),
        "budget_results": state.get(
            "budget_results",
            "",
        ),
        "budget_analysis": state.get(
            "budget_analysis",
            {},
        ),
        "itinerary": state.get(
            "itinerary",
            "",
        ),
    }

    constraints = state.get(
        "trip_constraints",
        {},
    )

    verdict = run_critic(
        plan,
        constraints,
        llm,
    )

    # IMPORTANT:
    # The critic owns the iteration counter.
    #
    # Do NOT increment iteration_count again inside graph.py.
    iteration_count = (
        int(
            state.get(
                "iteration_count",
                0,
            )
        )
        + 1
    )

    verdict_dict = verdict.to_dict()

    logger.info(
        "Critic completed | decision=%s | score=%.2f | confidence=%.2f | iteration=%d",
        verdict.decision,
        verdict.quality_score,
        verdict.confidence,
        iteration_count,
    )

    return {
        "critic_verdict": verdict_dict,
        "iteration_count": iteration_count,
    }
