from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


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

VALID_RESPONSIBLE_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}


# ============================================================
# VERDICT
# ============================================================


@dataclass
class CriticVerdict:
    passed: bool
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    responsible_agents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "suggestions": self.suggestions,
            "responsible_agents": self.responsible_agents,
        }


# ============================================================
# HELPERS
# ============================================================


def _safe_float(value: Any) -> float | None:
    """
    Convert a value to float safely.

    Supports:
        100
        100.5
        "100"
        "100.50"
        "₹1,50,000"
        "INR 150000"

    Returns None when conversion is impossible (e.g. free-text like
    "under 2 lakhs" — the deterministic check simply skips in that case
    rather than crashing).
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Strip currency symbols, commas, and surrounding whitespace/words,
        # keeping only digits and a single decimal point.
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
    """
    Keep only valid graph agent names.
    """
    if not agents:
        return []

    result = []

    for agent in agents:
        if agent in VALID_RESPONSIBLE_AGENTS:
            result.append(agent)

    return list(dict.fromkeys(result))


# ============================================================
# DETERMINISTIC CHECKS
# ============================================================


def rule_based_checks(
    plan: dict[str, Any],
    constraints: dict[str, Any],
) -> CriticVerdict:
    """
    Cheap deterministic checks.

    These checks should run before any LLM critic call.
    """

    violations: list[str] = []
    suggestions: list[str] = []
    responsible: list[str] = []

    # --------------------------------------------------------
    # Budget
    # --------------------------------------------------------

    budget_value = constraints.get("budget")

    budget = _safe_float(budget_value)

    budget_analysis = plan.get("budget_analysis") or {}

    total_cost = _safe_float(budget_analysis.get("total_cost"))

    if total_cost is None:
        total_cost = _safe_float(plan.get("total_cost"))

    if budget is not None and total_cost is not None and total_cost > budget:
        violations.append(
            f"Budget exceeded: estimated total cost "
            f"{total_cost:.2f} > budget {budget:.2f}"
        )

        suggestions.append("Reduce flight, hotel, activity, or transportation costs.")

        responsible.extend(
            [
                "budget_agent",
                "itinerary_agent",
            ]
        )

    # --------------------------------------------------------
    # Required flight
    # --------------------------------------------------------

    flight = plan.get("flight_results")

    if not flight or not str(flight).strip():
        violations.append("Missing flight planning information.")

        responsible.append("flight_agent")

    # --------------------------------------------------------
    # Required hotel
    # --------------------------------------------------------

    hotel = plan.get("hotel_results")

    if not hotel or not str(hotel).strip():
        violations.append("Missing hotel/accommodation planning information.")

        responsible.append("hotel_agent")

    # --------------------------------------------------------
    # Itinerary
    # --------------------------------------------------------

    itinerary = plan.get("itinerary")

    if not itinerary or not str(itinerary).strip():
        violations.append("Missing itinerary.")

        responsible.append("itinerary_agent")

    # --------------------------------------------------------
    # Date consistency
    # --------------------------------------------------------

    dates = constraints.get("dates") or {}

    start = dates.get("start")
    end = dates.get("end")

    if start and end:
        try:
            if str(start) > str(end):
                violations.append("Trip end date precedes trip start date.")

                responsible.append("itinerary_agent")

        except Exception:
            pass

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = constraints.get("duration")

    if duration is not None:
        try:
            duration_number = int(duration)

            if duration_number <= 0:
                violations.append("Trip duration must be greater than zero.")

                responsible.append("itinerary_agent")

        except (TypeError, ValueError):
            pass

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    responsible = list(dict.fromkeys(responsible))

    return CriticVerdict(
        passed=not violations,
        violations=violations,
        suggestions=suggestions,
        responsible_agents=responsible,
    )


# ============================================================
# LLM FUZZY CHECK
# ============================================================


def llm_fuzzy_check(
    plan: dict[str, Any],
    constraints: dict[str, Any],
    llm,
) -> CriticVerdict:
    """
    LLM-based fuzzy validation.

    Used for:
        - geographic sense
        - pacing
        - preference fit
        - routing quality
        - practical travel logic
    """

    prompt = f"""
You are a strict travel-plan quality critic.

Evaluate the proposed travel plan against the user's constraints.

CONSTRAINTS:
{json.dumps(constraints, indent=2, default=str)}

PLAN:
{json.dumps(plan, indent=2, default=str)}

Check:

1. Geographic routing makes sense.
2. Daily pacing is realistic.
3. User preferences are respected.
4. Destination, origin and duration are consistent.
5. Flight recommendations make sense.
6. Hotel recommendations make sense.
7. Weather considerations are relevant.
8. Budget assumptions are reasonable.
9. No important travel requirement was ignored.
10. The itinerary does not contain obvious contradictions.

IMPORTANT:

- Do not claim live booking availability unless explicitly provided.
- Do not invent confirmed prices.
- Estimates are acceptable when clearly identified as estimates.
- Only identify a violation when there is a meaningful problem.
- If the plan is acceptable, return passed=true.
- responsible_agents must contain only graph agent names.

Return ONLY valid JSON.

Schema:

{{
    "passed": true,
    "violations": [],
    "suggestions": [],
    "responsible_agents": []
}}

Allowed responsible_agents:

[
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
]
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=("You are a strict travel-plan critic. Return JSON only.")
            ),
            HumanMessage(content=prompt),
        ]
    )

    content = response.content

    if not isinstance(content, str):
        content = str(content)

    # --------------------------------------------------------
    # Extract JSON
    # --------------------------------------------------------

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Critic LLM did not return a JSON object.")

    data = json.loads(content[start : end + 1])

    passed = data.get("passed")

    if not isinstance(passed, bool):
        raise ValueError("Critic verdict 'passed' must be boolean.")

    violations = data.get(
        "violations",
        [],
    )

    suggestions = data.get(
        "suggestions",
        [],
    )

    responsible = data.get(
        "responsible_agents",
        [],
    )

    if not isinstance(violations, list):
        violations = [str(violations)]

    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)]

    if not isinstance(responsible, list):
        responsible = [str(responsible)]

    responsible = _normalise_agent_names(responsible)

    return CriticVerdict(
        passed=passed,
        violations=[str(x) for x in violations],
        suggestions=[str(x) for x in suggestions],
        responsible_agents=responsible,
    )


# ============================================================
# MAIN CRITIC
# ============================================================


def run_critic(
    plan: dict[str, Any],
    constraints: dict[str, Any],
    llm,
) -> CriticVerdict:
    """
    Execute deterministic checks first.

    If deterministic checks fail, return immediately.

    If they pass, perform the fuzzy LLM evaluation.
    """

    rule_verdict = rule_based_checks(
        plan,
        constraints,
    )

    # --------------------------------------------------------
    # Hard failure
    # --------------------------------------------------------

    if not rule_verdict.passed:
        return rule_verdict

    # --------------------------------------------------------
    # Fuzzy LLM evaluation
    # --------------------------------------------------------

    try:
        llm_verdict = llm_fuzzy_check(
            plan,
            constraints,
            llm,
        )

        return llm_verdict

    except Exception as exc:
        # IMPORTANT:
        # Do not silently pass an invalid critic response.
        #
        # We fail safely toward itinerary review instead.
        return CriticVerdict(
            passed=False,
            violations=["Critic validation failed unexpectedly."],
            suggestions=["Review the itinerary manually before approval."],
            responsible_agents=["itinerary_agent"],
        )


# ============================================================
# GRAPH NODE
# ============================================================


def critic_node(
    state: dict[str, Any],
    llm,
) -> dict[str, Any]:
    """
    LangGraph critic node.
    """

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

    iteration_count = (
        state.get(
            "iteration_count",
            0,
        )
        + 1
    )

    print("\n========== CRITIC ==========")
    print(
        json.dumps(
            verdict.to_dict(),
            indent=2,
            default=str,
        )
    )
    print(
        "Iteration:",
        iteration_count,
    )
    print("============================\n")

    return {
        "critic_verdict": verdict.to_dict(),
        "iteration_count": iteration_count,
    }
