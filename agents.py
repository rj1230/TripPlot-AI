from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from config import get_llm
from mcp_client import (
    current_weather,
    forecast,
    list_airlines,
    list_airports,
    tavily_search,
)
from state import TravelState


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# CONFIGURATION
# ============================================================

MAX_LLM_RETRIES = 2
MAX_TOOL_RETRIES = 2

TOOL_TIMEOUT_SECONDS = 25
LLM_TIMEOUT_SECONDS = 45

MAX_PROMPT_CHARS = 7000
MAX_SPECIALIST_OUTPUT_CHARS = 1800

DEFAULT_CURRENCY = "INR"

ALLOWED_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

REPLANNABLE = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
}


# ============================================================
# LLM
# ============================================================

llm = get_llm()

T = TypeVar("T", bound=BaseModel)


# ============================================================
# PYDANTIC CONTRACTS
# ============================================================


class TripConstraints(BaseModel):
    """Normalized travel requirements extracted by the supervisor."""

    destination: str = ""
    origin: str = ""
    duration: str = ""
    budget: str = ""
    travel_style: str = ""
    special_preferences: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    """Structured supervisor routing decision."""

    selected_agents: list[str] = Field(default_factory=list)
    trip_constraints: TripConstraints = Field(default_factory=TripConstraints)
    reasoning: str = ""


class FlightAnalysis(BaseModel):
    """Structured flight planning output."""

    recommended_departure_airport: str = ""
    recommended_arrival_airport: str = ""
    airlines: list[str] = Field(default_factory=list)
    approximate_duration: str = ""
    estimated_fare_range: str = ""
    direct_available: bool | None = None
    peak_season_warnings: list[str] = Field(default_factory=list)
    booking_advice: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HotelAnalysis(BaseModel):
    """Structured accommodation planning output."""

    recommended_areas: list[str] = Field(default_factory=list)
    area_reasons: dict[str, str] = Field(default_factory=dict)
    hotel_suggestions: list[str] = Field(default_factory=list)
    approximate_price_ranges: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    booking_advice: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class BudgetCategories(BaseModel):
    flights: float = 0.0
    hotel: float = 0.0
    food_and_transport: float = 0.0
    activities: float = 0.0


class BudgetAnalysis(BaseModel):
    """Structured budget result consumed by critic.py."""

    total_cost: float = 0.0
    currency: str = DEFAULT_CURRENCY
    categories: BudgetCategories = Field(default_factory=BudgetCategories)
    risk_areas: list[str] = Field(default_factory=list)
    money_saving_suggestions: list[str] = Field(default_factory=list)
    feasible: bool = False
    narrative: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ItineraryAnalysis(BaseModel):
    """Structured quality metadata for the generated itinerary."""

    overview: str = ""
    days: list[str] = Field(default_factory=list)
    transportation: list[str] = Field(default_factory=list)
    accommodation: list[str] = Field(default_factory=list)
    food: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    daily_spending: list[str] = Field(default_factory=list)
    weather_considerations: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    budget_summary: str = ""
    assumptions: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Deterministic validation result."""

    passed: bool
    warnings: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


# ============================================================
# GENERAL HELPERS
# ============================================================


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: Any, max_chars: int = 1500) -> str:
    """
    Keep downstream prompts bounded.

    This is deliberately conservative because several nodes combine
    multiple specialist outputs into one LLM request.
    """

    if text is None:
        return ""

    if isinstance(text, (dict, list, tuple)):
        try:
            text = json.dumps(
                text,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            text = str(text)

    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated]"


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)

        if number != number:
            return default

        if number == float("inf") or number == float("-inf"):
            return default

        return number

    except (TypeError, ValueError):
        return default


def _normalize_confidence(value: Any) -> float:
    confidence = _safe_float(value, 0.0)

    if confidence > 1:
        confidence = confidence / 100

    return max(0.0, min(1.0, confidence))


def _increment_llm_calls(state: TravelState, amount: int = 1) -> int:
    return int(state.get("llm_calls", 0) or 0) + amount


def _message(text: str) -> list[AIMessage]:
    return [AIMessage(content=text)]


# ============================================================
# LLM HELPERS
# ============================================================


def _llm_text(
    system: str,
    prompt: str,
    *,
    retries: int = MAX_LLM_RETRIES,
) -> str:
    """
    Synchronous LLM invocation with bounded retries.

    Important:
    We don't silently swallow failures. The final exception is
    propagated so LangGraph/checkpointing can handle it correctly.
    """

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=_truncate(prompt, MAX_PROMPT_CHARS)),
                ]
            )

            content = getattr(response, "content", "")

            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)

            content = str(content).strip()

            if not content:
                raise ValueError("LLM returned an empty response.")

            return content

        except Exception as exc:
            last_error = exc

            logger.warning(
                "LLM sync call failed: attempt=%s/%s error=%s",
                attempt + 1,
                retries + 1,
                exc,
            )

            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"LLM invocation failed after {retries + 1} attempts."
    ) from last_error


async def _llm_text_async(
    system: str,
    prompt: str,
    *,
    retries: int = MAX_LLM_RETRIES,
) -> str:
    """Async LLM invocation with timeout and bounded retries."""

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(
                    [
                        SystemMessage(content=system),
                        HumanMessage(
                            content=_truncate(
                                prompt,
                                MAX_PROMPT_CHARS,
                            )
                        ),
                    ]
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )

            content = getattr(response, "content", "")

            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)

            content = str(content).strip()

            if not content:
                raise ValueError("LLM returned an empty response.")

            return content

        except Exception as exc:
            last_error = exc

            logger.warning(
                "LLM async call failed: attempt=%s/%s error=%s",
                attempt + 1,
                retries + 1,
                exc,
            )

            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"Async LLM invocation failed after {retries + 1} attempts."
    ) from last_error


def _structured_llm(
    model: type[T],
    system: str,
    prompt: str,
    *,
    retries: int = MAX_LLM_RETRIES,
) -> T:
    """
    Preferred structured-output path.

    Falls back to conservative JSON extraction only when the configured
    LLM/provider doesn't support native structured output.
    """

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            structured = llm.with_structured_output(model)

            result = structured.invoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(
                        content=_truncate(
                            prompt,
                            MAX_PROMPT_CHARS,
                        )
                    ),
                ]
            )

            if isinstance(result, model):
                return result

            return model.model_validate(result)

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Structured LLM call failed: attempt=%s/%s error=%s",
                attempt + 1,
                retries + 1,
                exc,
            )

            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    # --------------------------------------------------------
    # Compatibility fallback.
    #
    # Some model wrappers/providers don't implement
    # with_structured_output(). We still validate the final object
    # through Pydantic rather than returning arbitrary JSON.
    # --------------------------------------------------------

    try:
        raw = _llm_text(
            system + "\nReturn ONLY valid JSON.",
            prompt,
            retries=1,
        )

        payload = _extract_json_object(raw)

        return model.model_validate(payload)

    except Exception as fallback_error:
        raise RuntimeError(
            f"Structured LLM generation failed for {model.__name__}."
        ) from (fallback_error or last_error)


async def _structured_llm_async(
    model: type[T],
    system: str,
    prompt: str,
    *,
    retries: int = MAX_LLM_RETRIES,
) -> T:
    """
    Async structured-output helper.

    Uses native structured output first. Falls back to async text +
    Pydantic validation if necessary.
    """

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            structured = llm.with_structured_output(model)

            result = await asyncio.wait_for(
                structured.ainvoke(
                    [
                        SystemMessage(content=system),
                        HumanMessage(
                            content=_truncate(
                                prompt,
                                MAX_PROMPT_CHARS,
                            )
                        ),
                    ]
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )

            if isinstance(result, model):
                return result

            return model.model_validate(result)

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Async structured LLM call failed: attempt=%s/%s error=%s",
                attempt + 1,
                retries + 1,
                exc,
            )

            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))

    try:
        raw = await _llm_text_async(
            system + "\nReturn ONLY valid JSON.",
            prompt,
            retries=1,
        )

        payload = _extract_json_object(raw)

        return model.model_validate(payload)

    except Exception as fallback_error:
        raise RuntimeError(
            f"Async structured LLM generation failed for {model.__name__}."
        ) from (fallback_error or last_error)


# ============================================================
# JSON COMPATIBILITY HELPER
# ============================================================


def _extract_json_object(text: str) -> dict:
    """
    Compatibility parser.

    This is intentionally NOT the primary structured-output mechanism.
    Native Pydantic structured output is attempted first.
    """

    if not text:
        raise ValueError("Empty LLM response.")

    text = str(text).strip()

    # Remove fenced JSON if present.
    if "```" in text:
        text = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")

    payload = text[start : end + 1]

    parsed = json.loads(payload)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")

    return parsed


# Backward-compatible alias.
def _json_from_llm(text: str) -> dict:
    return _extract_json_object(text)


# ============================================================
# MCP RESPONSE PARSING HELPERS
# ============================================================


def _extract_mcp_text(payload: Any) -> str:
    """
    MCP tool results commonly come back as a list of content blocks:

        [{"type": "text", "text": "...json or plain text...", "id": "..."}]

    This pulls out and concatenates the actual text content instead of
    letting the raw block structure leak into prompts/output.
    """

    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    if isinstance(payload, list):
        parts = []

        for item in payload:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, dict):
                # Unknown block shape; keep something rather than nothing.
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    if isinstance(payload, dict) and "text" in payload:
        return str(payload["text"])

    return str(payload)


def _parse_mcp_json(payload: Any) -> dict | list | None:
    """
    Extract the text content from an MCP tool response and parse it as
    JSON. Returns None (never raises) if parsing fails, so callers can
    degrade gracefully instead of crashing on a malformed tool response.
    """

    text = _extract_mcp_text(payload).strip()

    if not text:
        return None

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    # Some MCP servers double-encode: the "text" field is itself a
    # JSON string containing JSON. Try one more decode if we got a str.
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            return None

    return parsed


def _format_temp(value: Any) -> str:
    number = _safe_float(value, default=None) if value is not None else None

    if number is None:
        return "N/A"

    return f"{number:.1f}°C"


# ============================================================
# MCP / TOOL HELPERS
# ============================================================


async def _tool_call_async(
    tool,
    *args,
    tool_name: str | None = None,
    retries: int = MAX_TOOL_RETRIES,
    timeout: int = TOOL_TIMEOUT_SECONDS,
    **kwargs,
):
    """
    Reliable async wrapper around MCP tools.

    Handles:
      - timeout
      - transient failures
      - bounded retries
      - useful logging

    It deliberately raises after exhausting retries instead of
    pretending that a failed external tool succeeded.
    """

    name = tool_name or getattr(
        tool,
        "__name__",
        "unknown_tool",
    )

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(
                tool(*args, **kwargs),
                timeout=timeout,
            )

            return result

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Tool failed: tool=%s attempt=%s/%s error=%s",
                name,
                attempt + 1,
                retries + 1,
                exc,
            )

            if attempt < retries:
                await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(
        f"Tool '{name}' failed after {retries + 1} attempts."
    ) from last_error


# ============================================================
# SUPERVISOR AGENT
# ============================================================


def supervisor_agent(state: TravelState):
    """
    Production supervisor.

    Responsibilities:
      1. Normalize the user request.
      2. Select only valid specialist agents.
      3. Extract structured travel constraints.
      4. Prevent arbitrary agent routing.
      5. Preserve targeted replan decisions.
    """

    if state.get("is_replan"):
        selected = [
            agent
            for agent in state.get("selected_agents", [])
            if agent in ALLOWED_AGENTS
        ]

        if not selected:
            selected = ["itinerary_agent"]

        logger.info(
            "Supervisor executing targeted replan: %s",
            selected,
        )

        return {
            "selected_agents": selected,
            "is_replan": False,
            "messages": _message("Supervisor routed the targeted replan."),
        }

    query = str(state.get("user_query", "")).strip()

    if not query:
        raise ValueError("supervisor_agent received no user_query.")

    prompt = f"""
You are the supervisor and planner of a production travel-planning
multi-agent system.

Your job is to determine which specialist agents are actually required.

AVAILABLE AGENTS:

flight_agent:
- flights
- airports
- airlines
- route planning
- airfare guidance

hotel_agent:
- hotels
- accommodation
- neighborhoods
- areas to stay

weather_agent:
- current weather
- forecast
- climate
- season
- packing implications

budget_agent:
- cost estimation
- affordability
- budget constraints
- financial feasibility

itinerary_agent:
- final day-by-day travel plan
- practically always required

ROUTING RULES:

1. Never invent agent names.
2. itinerary_agent should normally be selected.
3. Select flight_agent when transportation/flight information is
   relevant.
4. Select hotel_agent when accommodation is relevant.
5. Select weather_agent when weather/season/packing affects the trip.
6. Select budget_agent when a budget is given or cost planning is
   materially useful.
7. Do not select agents merely to increase the number of agents.
8. Extract only information actually supported by the user request.
9. Empty strings are preferable to hallucinated details.

USER REQUEST:
{query}
"""

    decision = _structured_llm(
        SupervisorDecision,
        (
            "You are a strict production routing controller. "
            "Return a validated structured routing decision. "
            "Never invent facts."
        ),
        prompt,
    )

    selected = [agent for agent in decision.selected_agents if agent in ALLOWED_AGENTS]

    # Itinerary is the normal terminal planning component.
    if "itinerary_agent" not in selected:
        selected.append("itinerary_agent")

    # Remove duplicates while preserving order.
    selected = list(dict.fromkeys(selected))

    constraints = decision.trip_constraints.model_dump()

    logger.info(
        "Supervisor selected agents=%s destination=%s",
        selected,
        constraints.get("destination"),
    )

    return {
        "selected_agents": selected,
        "trip_constraints": constraints,
        "supervisor_reasoning": _truncate(
            decision.reasoning,
            1200,
        ),
        "messages": _message("Supervisor created a validated execution plan."),
        "llm_calls": _increment_llm_calls(state),
    }


# ============================================================
# FLIGHT AGENT
# ============================================================


async def flight_agent(state: TravelState):
    """
    Async flight specialist.

    External MCP data is treated as evidence. The LLM is explicitly
    prohibited from inventing live availability.
    """

    query = str(state.get("user_query", ""))

    constraints = (
        state.get(
            "trip_constraints",
            {},
        )
        or {}
    )

    destination = str(
        constraints.get(
            "destination",
            "",
        )
    ).strip()

    origin = str(
        constraints.get(
            "origin",
            "",
        )
    ).strip()

    if not destination:
        return {
            "flight_results": (
                "Flight planning could not be completed because "
                "the destination was not identified."
            ),
            "messages": _message("Flight agent could not identify the destination."),
        }

    logger.info(
        "Flight agent: origin=%s destination=%s",
        origin,
        destination,
    )

    # --------------------------------------------------------
    # MCP DATA
    # --------------------------------------------------------

    airports, airlines = await asyncio.gather(
        _tool_call_async(
            list_airports,
            destination,
            limit=10,
            tool_name="list_airports",
        ),
        _tool_call_async(
            list_airlines,
            "",
            limit=10,
            tool_name="list_airlines",
        ),
    )

    prompt = f"""
Create practical flight guidance using ONLY the supplied evidence.

USER REQUEST:
{_truncate(query, 1600)}

TRIP CONSTRAINTS:
{_truncate(constraints, 1400)}

ORIGIN:
{origin}

DESTINATION:
{destination}

AIRPORT MCP EVIDENCE:
{_truncate(airports, 2200)}

AIRLINE MCP EVIDENCE:
{_truncate(airlines, 1800)}

Requirements:

- Recommend sensible departure/arrival airports when supported.
- List relevant airlines only when supported by the evidence.
- Give an approximate duration only as an estimate.
- Give fare guidance only as an estimate if supported.
- Distinguish direct vs connecting considerations.
- Mention peak-season considerations if relevant.
- Give practical booking advice.
- Explicitly state assumptions.
- NEVER claim live availability.
- NEVER claim a ticket is booked.
- NEVER invent an exact fare.
"""

    analysis = await _structured_llm_async(
        FlightAnalysis,
        (
            "You are a professional flight-planning specialist. "
            "Ground every recommendation in supplied evidence. "
            "Never fabricate live inventory."
        ),
        prompt,
    )

    result = (
        "Flight Planning\n"
        f"- Departure airport: "
        f"{analysis.recommended_departure_airport or 'Not determined'}\n"
        f"- Arrival airport: "
        f"{analysis.recommended_arrival_airport or 'Not determined'}\n"
        f"- Airlines: "
        f"{', '.join(analysis.airlines) or 'Not determined'}\n"
        f"- Approximate duration: "
        f"{analysis.approximate_duration or 'Not available'}\n"
        f"- Estimated fare range: "
        f"{analysis.estimated_fare_range or 'Not available'}\n"
        f"- Direct flight available: "
        f"{analysis.direct_available}\n"
        f"- Peak-season warnings: "
        f"{'; '.join(analysis.peak_season_warnings) or 'None identified'}\n"
        f"- Booking advice: "
        f"{'; '.join(analysis.booking_advice) or 'Compare current fares before booking'}\n"
        f"- Assumptions: "
        f"{'; '.join(analysis.assumptions) or 'None'}\n"
        f"- Confidence: "
        f"{_normalize_confidence(analysis.confidence):.2f}\n"
        f"- Retrieved at: {_utc_now()}\n"
        "\nImportant: This is planning guidance, not a confirmed booking."
    )

    return {
        "flight_results": _truncate(
            result,
            MAX_SPECIALIST_OUTPUT_CHARS,
        ),
        "messages": _message("Flight agent completed using validated MCP evidence."),
        "llm_calls": _increment_llm_calls(state),
    }


# ============================================================
# HOTEL AGENT
# ============================================================


async def hotel_agent(state: TravelState):
    """
    Accommodation specialist.

    Tavily/MCP output is first treated as raw evidence and then
    transformed into structured recommendations.
    """

    user_query = str(state.get("user_query", "")).strip()

    if not user_query:
        raise ValueError("hotel_agent requires user_query.")

    query = (
        "Find reliable hotel, accommodation, and neighborhood "
        f"information for this travel request:\n{user_query}"
    )

    logger.info("Hotel agent executing search.")

    search_result = await _tool_call_async(
        tavily_search,
        query,
        tool_name="tavily_search",
    )

    prompt = f"""
Create practical accommodation guidance from the supplied search
evidence.

USER REQUEST:
{_truncate(user_query, 1800)}

RAW SEARCH EVIDENCE:
{_truncate(search_result, 3200)}

Requirements:

1. Recommend useful neighborhoods/areas.
2. Explain why each area fits the trip.
3. Give 2-3 specific stay suggestions where the evidence supports them.
4. Include approximate price ranges only when supported.
5. Explain budget/convenience trade-offs.
6. Give booking timing advice when supported.
7. Clearly identify assumptions.
8. Do NOT invent hotel availability.
9. Do NOT invent exact prices.
10. Do NOT claim that a reservation exists.
"""

    analysis = await _structured_llm_async(
        HotelAnalysis,
        (
            "You are a professional accommodation specialist. "
            "Use supplied search evidence and avoid unsupported claims."
        ),
        prompt,
    )

    area_lines = []

    for area in analysis.recommended_areas:
        reason = analysis.area_reasons.get(
            area,
            "",
        )

        if reason:
            area_lines.append(f"- {area}: {reason}")
        else:
            area_lines.append(f"- {area}")

    result = (
        "Accommodation Planning\n"
        f"Recommended areas:\n"
        f"{chr(10).join(area_lines) or '- No specific area established'}\n\n"
        "Hotel/stay suggestions:\n"
        f"{chr(10).join('- ' + x for x in analysis.hotel_suggestions) or '- None established'}\n\n"
        "Approximate price ranges:\n"
        f"{chr(10).join('- ' + x for x in analysis.approximate_price_ranges) or '- None established'}\n\n"
        "Trade-offs:\n"
        f"{chr(10).join('- ' + x for x in analysis.tradeoffs) or '- None identified'}\n\n"
        "Booking advice:\n"
        f"{chr(10).join('- ' + x for x in analysis.booking_advice) or '- Compare current availability before booking'}\n\n"
        "Assumptions:\n"
        f"{chr(10).join('- ' + x for x in analysis.assumptions) or '- None'}\n"
        f"\nConfidence: {_normalize_confidence(analysis.confidence):.2f}"
        f"\nRetrieved at: {_utc_now()}"
        "\n\nImportant: Suggestions are not confirmed reservations."
    )

    return {
        "hotel_results": _truncate(
            result,
            MAX_SPECIALIST_OUTPUT_CHARS,
        ),
        "messages": _message("Hotel agent completed using search evidence."),
        "llm_calls": _increment_llm_calls(state),
    }


# ============================================================
# WEATHER AGENT (improved)
# ============================================================


async def weather_agent(state: TravelState):
    """
    Async weather specialist.

    Improvements over the previous version:
      - Parses the MCP content-block/JSON response instead of dumping
        raw tool output into the result shown to the user.
      - Produces a clean, human-readable summary.
      - Also returns a structured `weather_analysis` dict so downstream
        agents (budget, itinerary) can consume real fields instead of
        re-parsing a text blob.
      - Degrades gracefully (with the raw evidence preserved) if the
        MCP payload can't be parsed, instead of silently failing or
        crashing.
      - Catches tool failures locally since weather is non-critical
        evidence; the rest of the plan can proceed without it.
    """

    constraints = (
        state.get(
            "trip_constraints",
            {},
        )
        or {}
    )

    city = str(
        constraints.get(
            "destination",
            "",
        )
    ).strip()

    if not city:
        return {
            "weather_results": (
                "Weather information could not be retrieved because "
                "the destination was not identified."
            ),
            "weather_analysis": {},
            "messages": _message("Weather agent could not identify destination."),
        }

    logger.info(
        "Weather agent: destination=%s",
        city,
    )

    try:
        weather_raw, forecast_raw = await asyncio.gather(
            _tool_call_async(
                current_weather,
                city,
                tool_name="current_weather",
            ),
            _tool_call_async(
                forecast,
                city,
                tool_name="forecast",
            ),
        )
    except Exception as exc:
        logger.error(
            "Weather agent tool call failed: %s",
            exc,
        )

        return {
            "weather_results": (
                f"Weather data for {city} could not be retrieved due "
                "to a tool error. Proceed with general seasonal "
                "assumptions and verify conditions closer to departure."
            ),
            "weather_analysis": {},
            "messages": _message(
                "Weather agent tool call failed; continuing without live weather data."
            ),
        }

    retrieved_at = _utc_now()

    weather_data = _parse_mcp_json(weather_raw)
    forecast_data = _parse_mcp_json(forecast_raw)

    current = weather_data if isinstance(weather_data, dict) else {}

    forecast_days: list[dict] = []

    if isinstance(forecast_data, dict):
        raw_days = forecast_data.get("forecast", [])
        forecast_days = (
            [d for d in raw_days if isinstance(d, dict)]
            if isinstance(raw_days, list)
            else []
        )
    elif isinstance(forecast_data, list):
        forecast_days = [d for d in forecast_data if isinstance(d, dict)]

    parsed_ok = bool(current) or bool(forecast_days)

    weather_analysis: dict[str, Any] = {
        "city": current.get("city", city),
        "temperature_c": (
            _safe_float(current.get("temperature_c"))
            if "temperature_c" in current
            else None
        ),
        "feels_like_c": (
            _safe_float(current.get("feels_like_c"))
            if "feels_like_c" in current
            else None
        ),
        "humidity": current.get("humidity"),
        "condition": str(current.get("condition", "")).strip(),
        "wind_speed": (
            _safe_float(current.get("wind_speed")) if "wind_speed" in current else None
        ),
        "forecast": [
            {
                "datetime": str(day.get("datetime", "")),
                "temperature": _safe_float(day.get("temperature")),
                "weather": str(day.get("weather", "")).strip(),
            }
            for day in forecast_days
        ],
    }

    if parsed_ok:
        lines = [f"Current weather in {weather_analysis['city']}:"]

        if weather_analysis["condition"]:
            lines.append(f"- Condition: {weather_analysis['condition'].capitalize()}")

        if weather_analysis["temperature_c"] is not None:
            feels = weather_analysis["feels_like_c"]
            feels_txt = f" (feels like {feels:.1f}°C)" if feels is not None else ""
            lines.append(
                f"- Temperature: {weather_analysis['temperature_c']:.1f}°C{feels_txt}"
            )

        if weather_analysis["humidity"] is not None:
            lines.append(f"- Humidity: {weather_analysis['humidity']}%")

        if weather_analysis["wind_speed"] is not None:
            lines.append(f"- Wind speed: {weather_analysis['wind_speed']:.1f} m/s")

        if weather_analysis["forecast"]:
            lines.append("\nForecast:")

            for day in weather_analysis["forecast"][:5]:
                date_label = (
                    day["datetime"].split(" ")[0] if day["datetime"] else "Unknown date"
                )
                lines.append(
                    f"- {date_label}: {_format_temp(day['temperature'])}, "
                    f"{day['weather'].capitalize() or 'No data'}"
                )
        else:
            lines.append("\nForecast: not available.")

        result = "\n".join(lines)
    else:
        logger.warning(
            "Weather agent: could not parse MCP payload for city=%s",
            city,
        )

        result = (
            f"Weather data for {city} was returned in an unexpected "
            "format and could not be fully parsed. Raw evidence "
            "(for debugging):\n"
            f"{_truncate(_extract_mcp_text(weather_raw), 500)}\n"
            f"{_truncate(_extract_mcp_text(forecast_raw), 500)}"
        )

    result += f"\n\nRetrieved at: {retrieved_at}"
    result += "\nWeather data is time-sensitive; verify near departure."

    return {
        "weather_results": _truncate(
            result,
            MAX_SPECIALIST_OUTPUT_CHARS,
        ),
        "weather_analysis": weather_analysis,
        "messages": _message("Weather agent completed using current MCP weather data."),
        "llm_calls": _increment_llm_calls(state),
    }


# ============================================================
# BUDGET VALIDATION
# ============================================================


def _validate_budget(
    analysis: BudgetAnalysis,
) -> ValidationResult:
    """
    Deterministic budget arithmetic validation.

    This prevents the LLM from returning a total that doesn't equal
    its own category breakdown.
    """

    categories = analysis.categories

    calculated_total = (
        categories.flights
        + categories.hotel
        + categories.food_and_transport
        + categories.activities
    )

    warnings: list[str] = []
    violations: list[str] = []

    if analysis.total_cost < 0:
        violations.append("Budget total cannot be negative.")

    if any(
        value < 0
        for value in [
            categories.flights,
            categories.hotel,
            categories.food_and_transport,
            categories.activities,
        ]
    ):
        violations.append("Budget categories cannot contain negative values.")

    if analysis.total_cost > 0:
        difference = abs(calculated_total - analysis.total_cost)

        tolerance = max(
            100.0,
            analysis.total_cost * 0.05,
        )

        if difference > tolerance:
            violations.append("Budget total does not match the category breakdown.")

    if not analysis.currency:
        warnings.append("Budget currency was not explicitly established.")

    return ValidationResult(
        passed=not violations,
        warnings=warnings,
        violations=violations,
    )


# ============================================================
# BUDGET AGENT
# ============================================================


def budget_agent(state: TravelState):
    """
    Analyze financial feasibility.

    Important production behavior:
      - structured output first
      - Pydantic validation
      - deterministic arithmetic validation
      - no silent empty-analysis fallback
    """

    constraints = (
        state.get(
            "trip_constraints",
            {},
        )
        or {}
    )

    prompt = f"""
Analyze whether this trip is financially realistic.

USER REQUEST:
{_truncate(state.get("user_query", ""), 1800)}

TRIP CONSTRAINTS:
{_truncate(constraints, 1500)}

FLIGHT EVIDENCE:
{_truncate(state.get("flight_results", ""), 1600)}

HOTEL EVIDENCE:
{_truncate(state.get("hotel_results", ""), 1600)}

WEATHER EVIDENCE:
{_truncate(state.get("weather_results", ""), 900)}

Rules:

- All numbers are estimates unless externally confirmed.
- Use the currency implied by the user's budget.
- If no currency is specified, use INR.
- Never claim a confirmed price.
- Do not invent a booking.
- total_cost must equal the sum of categories.
- If insufficient evidence exists, say so explicitly.
- confidence must reflect evidence quality.
"""

    analysis = _structured_llm(
        BudgetAnalysis,
        (
            "You are a conservative travel budget analyst. "
            "Never fabricate confirmed prices. "
            "Return internally consistent numeric estimates."
        ),
        prompt,
    )

    validation = _validate_budget(analysis)

    # --------------------------------------------------------
    # One targeted repair attempt if arithmetic is inconsistent.
    # --------------------------------------------------------

    if not validation.passed:
        repair_prompt = f"""
Repair the following budget analysis.

Original analysis:
{analysis.model_dump_json(indent=2)}

Validation problems:
{validation.violations}

Make the category totals mathematically consistent.
Do not invent new evidence.

Return a corrected structured budget analysis.
"""

        analysis = _structured_llm(
            BudgetAnalysis,
            (
                "You repair financial calculation consistency. "
                "Do not change unsupported facts."
            ),
            repair_prompt,
        )

        validation = _validate_budget(analysis)

    # --------------------------------------------------------
    # Hard safety fallback.
    # --------------------------------------------------------

    if not validation.passed:
        logger.error(
            "Budget validation failed: %s",
            validation.violations,
        )

        return {
            "budget_results": (
                "Budget analysis could not be validated reliably. "
                "The trip should be treated as requiring manual budget "
                "verification before relying on the estimate."
            ),
            "budget_analysis": {},
            "messages": _message("Budget analysis requires verification."),
            "llm_calls": _increment_llm_calls(
                state,
                2,
            ),
        }

    categories = analysis.categories.model_dump()

    budget_analysis = {
        "total_cost": round(
            _safe_float(analysis.total_cost),
            2,
        ),
        "currency": (analysis.currency or DEFAULT_CURRENCY),
        "categories": categories,
        "risk_areas": analysis.risk_areas,
        "money_saving_suggestions": (analysis.money_saving_suggestions),
        "feasible": bool(analysis.feasible),
        "confidence": _normalize_confidence(analysis.confidence),
    }

    narrative = analysis.narrative.strip()

    if not narrative:
        narrative = (
            "Estimated total: "
            f"{budget_analysis['currency']} "
            f"{budget_analysis['total_cost']:.2f}."
        )

    narrative += "\n\nBudget validation: PASS."

    return {
        "budget_results": _truncate(
            narrative,
            MAX_SPECIALIST_OUTPUT_CHARS,
        ),
        "budget_analysis": budget_analysis,
        "messages": _message("Budget agent completed with deterministic validation."),
        "llm_calls": _increment_llm_calls(
            state,
            1,
        ),
    }


# ============================================================
# REPLAN HELPERS
# ============================================================


def prepare_replan(state: TravelState):
    """
    Route only the agents identified as responsible by the critic.

    This prevents a failed weather check from unnecessarily re-running
    every specialist.
    """

    verdict = (
        state.get(
            "critic_verdict",
            {},
        )
        or {}
    )

    responsible = [
        agent
        for agent in verdict.get(
            "responsible_agents",
            [],
        )
        if agent in REPLANNABLE
    ]

    # Defensive deduplication.
    responsible = list(dict.fromkeys(responsible))

    logger.warning(
        "Preparing targeted replan: %s",
        responsible,
    )

    if not responsible:
        # If no specialist is identified, rebuilding the itinerary is
        # safer than looping through arbitrary agents.
        selected = ["itinerary_agent"]
    else:
        selected = responsible

    return {
        "selected_agents": selected,
        "is_replan": True,
        "unresolved_violations": [],
        "messages": _message("Targeted replan prepared for: " + ", ".join(selected)),
    }


def mark_unresolved(state: TravelState):
    """
    Critic loop exhausted.

    Do not pretend the plan is perfect. Preserve the unresolved
    violations so the human approval stage can see them.
    """

    verdict = (
        state.get(
            "critic_verdict",
            {},
        )
        or {}
    )

    violations = verdict.get(
        "violations",
        [],
    )

    logger.warning(
        "Maximum critic iterations reached. Unresolved violations=%s",
        violations,
    )

    return {
        "unresolved_violations": violations,
        "messages": _message(
            "Maximum correction attempts reached; "
            "unresolved issues were forwarded for human review."
        ),
    }


# ============================================================
# ITINERARY AGENT
# ============================================================


def itinerary_agent(state: TravelState):
    """
    Generate a draft itinerary from specialist evidence.

    The itinerary agent is explicitly instructed to distinguish:
      - confirmed tool facts
      - estimates
      - recommendations
      - assumptions
    """

    constraints = (
        state.get(
            "trip_constraints",
            {},
        )
        or {}
    )

    budget_analysis = (
        state.get(
            "budget_analysis",
            {},
        )
        or {}
    )

    weather_analysis = (
        state.get(
            "weather_analysis",
            {},
        )
        or {}
    )

    prompt = f"""
Create a practical draft travel itinerary.

USER REQUEST:
{_truncate(state.get("user_query", ""), 1800)}

TRIP CONSTRAINTS:
{_truncate(constraints, 1500)}

FLIGHT EVIDENCE:
{_truncate(state.get("flight_results", ""), 1500)}

HOTEL EVIDENCE:
{_truncate(state.get("hotel_results", ""), 1500)}

WEATHER EVIDENCE:
{_truncate(state.get("weather_results", ""), 1000)}

STRUCTURED WEATHER:
{_truncate(weather_analysis, 800)}

BUDGET EVIDENCE:
{_truncate(state.get("budget_results", ""), 1400)}

STRUCTURED BUDGET:
{_truncate(budget_analysis, 1200)}

Produce a realistic itinerary.

Required structure:

1. Trip overview
2. Day-by-day itinerary
3. Flights / transportation
4. Accommodation
5. Food
6. Activities
7. Estimated daily spending
8. Weather considerations
9. Important travel tips
10. Budget summary
11. Assumptions

CRITICAL GROUNDING RULES:

- Do not claim a booking exists.
- Do not claim live flight/hotel availability.
- Do not invent exact prices.
- Clearly label estimates.
- Do not contradict the structured budget.
- Do not introduce destinations not requested unless clearly marked
  as an optional recommendation.
- Keep the itinerary practical rather than filling every hour.
"""

    result = _llm_text(
        (
            "You are an expert itinerary planner. "
            "You synthesize supplied evidence without hallucinating "
            "bookings, prices, or availability."
        ),
        prompt,
    )

    result = _truncate(
        result,
        5000,
    )

    # --------------------------------------------------------
    # Deterministic lightweight sanity checks.
    # --------------------------------------------------------

    warnings: list[str] = []

    lowered = result.lower()

    suspicious_booking_phrases = [
        "your flight is booked",
        "your hotel is booked",
        "reservation confirmed",
        "booking confirmed",
        "ticket has been booked",
    ]

    for phrase in suspicious_booking_phrases:
        if phrase in lowered:
            warnings.append(f"Potential unsupported booking claim: '{phrase}'.")

    if warnings:
        logger.warning(
            "Itinerary grounding warnings: %s",
            warnings,
        )

    approval_request = f"""
Please review this draft travel plan.

DRAFT:
{result}

Before approving, check:

- Does it satisfy the original request?
- Are the dates/duration sensible?
- Are budget estimates consistent?
- Are unsupported booking claims absent?
- Are assumptions clearly identified?
- Are there any safety or practical issues?

If you reject it, provide specific corrections.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": _message("Draft itinerary created for review."),
        "llm_calls": _increment_llm_calls(state),
    }


# ============================================================
# HUMAN APPROVAL AGENT
# ============================================================


def human_approval_agent(state: TravelState):
    """
    Human-in-the-loop approval.

    Unresolved critic violations are explicitly surfaced to the human.
    """

    unresolved = (
        state.get(
            "unresolved_violations",
            [],
        )
        or []
    )

    approval_request = state.get(
        "approval_request",
        "",
    )

    if unresolved:
        approval_request = (
            approval_request
            + "\n\nUNRESOLVED SYSTEM WARNINGS:\n"
            + "\n".join(f"- {item}" for item in unresolved)
        )

    feedback = interrupt(
        {
            "question": (
                "Do you approve this itinerary? Please review any unresolved warnings."
            ),
            "draft_itinerary": state.get(
                "itinerary",
                "",
            ),
            "approval_request": approval_request,
            "unresolved_violations": unresolved,
            "expected_response": {
                "approved": True,
                "feedback": ("Optional feedback for revision"),
            },
        }
    )

    if not isinstance(feedback, dict):
        raise ValueError("Human approval response must be a dictionary.")

    approved = bool(
        feedback.get(
            "approved",
            False,
        )
    )

    human_feedback = str(
        feedback.get(
            "feedback",
            "",
        )
    ).strip()

    logger.info(
        "Human approval completed: approved=%s",
        approved,
    )

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": _message("Human approval step completed."),
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================


def final_response_agent(state: TravelState):
    """
    Produce the final user-facing response.

    Final generation is deliberately grounded in the already-created
    itinerary rather than asking the LLM to rediscover the trip.
    """

    approved = bool(
        state.get(
            "approved",
            False,
        )
    )

    human_feedback = str(
        state.get(
            "human_feedback",
            "",
        )
    )

    user_query = str(
        state.get(
            "user_query",
            "",
        )
    )

    unresolved = (
        state.get(
            "unresolved_violations",
            [],
        )
        or []
    )

    budget_analysis = (
        state.get(
            "budget_analysis",
            {},
        )
        or {}
    )

    if approved:
        mode_instruction = """
The human approved the draft.

Preserve the approved plan unless there is an obvious factual
consistency issue. Produce a polished final response.
"""
    else:
        mode_instruction = """
The human did NOT approve the draft.

Revise the plan according to the human feedback. Do not blindly
preserve rejected elements.
"""

    prompt = f"""
{mode_instruction}

ORIGINAL USER REQUEST:
{_truncate(user_query, 1800)}

TRIP CONSTRAINTS:
{_truncate(state.get("trip_constraints", {}), 1400)}

DRAFT ITINERARY:
{_truncate(state.get("itinerary", ""), 3200)}

BUDGET:
{_truncate(state.get("budget_results", ""), 1300)}

STRUCTURED BUDGET:
{_truncate(budget_analysis, 1200)}

HUMAN FEEDBACK:
{_truncate(human_feedback, 1200)}

UNRESOLVED WARNINGS:
{_truncate(unresolved, 1000)}

Produce the final user-ready travel plan.

Include:

1. Trip overview
2. Day-by-day itinerary
3. Transportation
4. Accommodation
5. Food
6. Activities
7. Budget
8. Weather considerations
9. Important travel tips
10. Assumptions / things to verify

FINAL GROUNDING RULES:

- Never say a flight is booked.
- Never say a hotel is booked.
- Never say availability is confirmed unless the system explicitly
  obtained booking confirmation.
- Never present estimates as guaranteed prices.
- Keep budget numbers consistent with the structured budget.
- Clearly identify assumptions.
- If information is uncertain, say so.
- Do not introduce unsupported facts.
"""

    result = _llm_text(
        (
            "You produce final production-quality travel plans. "
            "You are a grounded synthesis layer, not a booking engine. "
            "Never fabricate reservations or availability."
        ),
        prompt,
    )

    result = _truncate(
        result,
        6500,
    )

    # --------------------------------------------------------
    # Final output guardrail
    # --------------------------------------------------------

    forbidden_claims = [
        "flight is booked",
        "hotel is booked",
        "reservation is confirmed",
        "ticket is confirmed",
        "booking is confirmed",
    ]

    lowered = result.lower()

    detected = [phrase for phrase in forbidden_claims if phrase in lowered]

    if detected:
        logger.warning(
            "Final response contained potentially unsupported claims: %s",
            detected,
        )

        # One corrective pass instead of silently returning unsafe
        # wording.
        repair_prompt = f"""
Rewrite the following final travel response.

Remove or correct these unsupported booking claims:
{detected}

Response:
{result}

Rules:
- Never claim a booking exists.
- Replace unsupported confirmation language with neutral planning
  language such as "consider booking", "recommended", or
  "availability should be checked".
- Preserve the useful travel information.
"""

        result = _llm_text(
            ("You are a strict output-safety editor for travel planning."),
            repair_prompt,
            retries=1,
        )

    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": _increment_llm_calls(state),
    }
