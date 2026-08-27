import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

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
# LLM
# ============================================================

llm = get_llm()


def _truncate(text: Any, max_chars: int = 1500) -> str:
    """
    Cap a piece of text before it's embedded into a downstream LLM
    prompt.

    Several nodes (budget_agent, itinerary_agent, final_response_agent)
    concatenate the outputs of every prior specialist agent into a
    single prompt. Each individual output is already a full LLM-written
    paragraph (or, for hotel_agent, a raw search-API result), so without
    a cap here the combined prompt can silently exceed the model
    provider's tokens-per-minute limit — e.g. Groq's on-demand tier caps
    total request tokens at 8000, and an uncapped itinerary_agent prompt
    can exceed that on its own.

    ~4 characters per token is a reasonable rule of thumb for English
    prose, so max_chars=1500 keeps a single field to roughly 375 tokens.
    """

    if not text:
        return ""

    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated for length]"


def _llm_text(system: str, prompt: str) -> str:
    """
    Synchronous LLM helper for synchronous LangGraph nodes.
    """
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
    )

    return response.content


async def _llm_text_async(system: str, prompt: str) -> str:
    """
    Asynchronous LLM helper for asynchronous LangGraph nodes.
    """
    response = await llm.ainvoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
    )

    return response.content


# ============================================================
# JSON HELPERS
# ============================================================


def _json_from_llm(text: str) -> dict:
    """
    Extract JSON object from an LLM response.
    """

    print("\n========== RAW LLM RESPONSE ==========")
    print(text)
    print("======================================\n")

    try:
        start = text.index("{")
        end = text.rindex("}") + 1

        json_text = text[start:end]

        print("\n========== EXTRACTED JSON ==========")
        print(json_text)
        print("====================================\n")

        return json.loads(json_text)

    except (ValueError, json.JSONDecodeError) as exc:
        print("\n========== JSON PARSING ERROR ==========")
        print(exc)
        print("========================================\n")

        raise ValueError(
            f"Could not extract valid JSON from LLM response:\n{text}"
        ) from exc


# ============================================================
# SUPERVISOR AGENT
# ============================================================


def supervisor_agent(state: TravelState):
    """
    Supervisor decides which specialist agents should execute.
    """

    if state.get("is_replan"):
        # A replan pass already set selected_agents via prepare_replan —
        # trust it and skip re-deriving from user_query, or the critic's
        # verdict gets discarded and the loop never converges.
        print("\n========== SUPERVISOR: REPLAN PASS ==========")
        print("Trusting selected_agents:", state.get("selected_agents"))
        print("===============================================\n")

        return {
            "is_replan": False,  # reset so a future normal pass isn't skipped
            "messages": [
                AIMessage(content="Supervisor routed replan to targeted agents.")
            ],
        }

    # NOTE: use .get() rather than state["user_query"]. If this node is
    # ever re-entered with a state that doesn't carry the original
    # request (e.g. a checkpointer misconfiguration causing a resume to
    # restart from an empty state), we want a clear, catchable message
    # here rather than a bare KeyError deep in the graph.
    query = state.get("user_query", "")

    if not query:
        print("\n========== SUPERVISOR: MISSING user_query ==========")
        print("state keys present:", list(state.keys()))
        print("=====================================================\n")

        raise ValueError(
            "supervisor_agent received a state with no 'user_query'. "
            "This usually means the graph resumed without its checkpointed "
            "state (e.g. the checkpointer was recreated between the initial "
            "call and the resume call). Check that the same checkpointer "
            "instance/connection is used for a given thread_id across calls."
        )

    prompt = f"""
You are the supervisor of a real-world multi-agent travel planning system.

Decide which specialist agents are needed for this user request.

Available agents:

- flight_agent:
  Use when flights, airports, airlines, routes, or airfare guidance
  are needed.

- hotel_agent:
  Use when hotels, stays, neighborhoods, or accommodation
  are needed.

- weather_agent:
  Use when weather, climate, season, packing, or forecast
  information is useful.

- budget_agent:
  Use when budget, affordability, cost, or price constraints
  are mentioned.

- itinerary_agent:
  Almost always needed to produce the final travel plan.

Return ONLY valid JSON.

Use exactly this schema:

{{
  "selected_agents": [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
  ],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only.",
        prompt,
    )

    parsed = _json_from_llm(raw)

    print("\n========== PARSED SUPERVISOR JSON ==========")
    print(json.dumps(parsed, indent=2))
    print("=============================================\n")

    selected = parsed.get("selected_agents", [])
    trip_constraints = parsed.get("trip_constraints", {})
    reasoning = parsed.get("reasoning", "")

    return {
        "selected_agents": selected,
        "trip_constraints": trip_constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent execution plan.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# FLIGHT AGENT
# ============================================================


async def flight_agent(state: TravelState):
    """
    Async flight specialist.

    IMPORTANT:
    Do NOT use asyncio.run() here.
    LangGraph can execute async nodes directly.
    """

    query = state.get("user_query", "")

    constraints = state.get(
        "trip_constraints",
        {},
    )

    destination = constraints.get(
        "destination",
        "",
    )

    origin = constraints.get(
        "origin",
        "",
    )

    print("\n========== FLIGHT AGENT INPUT ==========")
    print("Query:", query)
    print("Origin:", origin)
    print("Destination:", destination)
    print("Constraints:", constraints)
    print("========================================\n")

    # --------------------------------------------------------
    # MCP calls
    # --------------------------------------------------------

    airports = await list_airports(
        destination,
        limit=10,
    )

    airlines = await list_airlines(
        "",
        limit=10,
    )

    print("\n========== AIRPORT MCP DATA ==========")
    print(airports)
    print("======================================\n")

    print("\n========== AIRLINE MCP DATA ==========")
    print(airlines)
    print("======================================\n")

    # --------------------------------------------------------
    # LLM prompt
    # --------------------------------------------------------

    prompt = f"""
Create practical flight guidance for this trip.

User request:
{query}

Trip constraints:
{constraints}

Origin:
{origin}

Destination:
{destination}

Airport MCP data:
{_truncate(airports, max_chars=2000)}

Airline MCP data:
{_truncate(airlines, max_chars=2000)}

Include:

1. Recommended departure airport
2. Recommended arrival airport
3. Relevant airlines
4. Approximate flight duration
5. Estimated fare range
6. Direct vs connecting flight considerations
7. Peak season warnings
8. Booking advice
9. Important assumptions

Do not claim that you have live ticket availability unless the
provided MCP data explicitly contains live availability.
"""

    result = await _llm_text_async(
        "You are a professional flight planning specialist.",
        prompt,
    )

    print("\n========== FLIGHT AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "flight_results": result,
        "messages": [AIMessage(content="Flight agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HOTEL AGENT
# ============================================================


async def hotel_agent(state: TravelState):
    """
    Async hotel/accommodation specialist.
    """

    user_query = state.get("user_query", "")

    query = f"Best hotels, neighborhoods, and areas to stay for: {user_query}"

    print("\n========== HOTEL AGENT INPUT ==========")
    print(query)
    print("=======================================\n")

    # Async Tavily/MCP call
    result = await tavily_search(query)

    print("\n========== HOTEL SEARCH RESULT ==========")
    print(result)
    print("=========================================\n")

    # Tavily can return a large raw payload (multiple pages of content).
    # Cap it here at the source so it doesn't blow the token budget of
    # every downstream prompt (budget_agent, itinerary_agent,
    # final_response_agent) that includes hotel_results verbatim.
    return {
        "hotel_results": _truncate(result, max_chars=2500),
        "messages": [AIMessage(content="Hotel agent completed.")],
        "llm_calls": state.get("llm_calls", 0),
    }


# ============================================================
# WEATHER AGENT
# ============================================================


async def weather_agent(state: TravelState):
    """
    Async weather specialist.
    """

    constraints = state.get(
        "trip_constraints",
        {},
    )

    city = constraints.get(
        "destination",
        "",
    )

    print("\n========== WEATHER AGENT INPUT ==========")
    print("City:", city)
    print("Constraints:", constraints)
    print("=========================================\n")

    if not city:
        return {
            "weather_results": (
                "Weather information could not be retrieved "
                "because the destination was not identified."
            ),
            "messages": [
                AIMessage(content="Weather agent could not identify destination.")
            ],
        }

    # --------------------------------------------------------
    # MCP calls
    # --------------------------------------------------------

    weather_data = await current_weather(city)

    forecast_data = await forecast(city)

    print("\n========== CURRENT WEATHER ==========")
    print(weather_data)
    print("=====================================\n")

    print("\n========== WEATHER FORECAST ==========")
    print(forecast_data)
    print("======================================\n")

    result = f"""
Current weather:
{weather_data}

Forecast:
{forecast_data}
"""

    print("\n========== WEATHER AGENT OUTPUT ==========")
    print(result)
    print("==========================================\n")

    return {
        "weather_results": result,
        "messages": [AIMessage(content="Weather agent completed.")],
    }


# ============================================================
# BUDGET AGENT
# ============================================================


def budget_agent(state: TravelState):
    """
    Analyze whether the planned trip is financially realistic.

    Returns BOTH:
      - budget_results: human-readable prose assessment (consumed by
        itinerary_agent / final_response_agent prompts)
      - budget_analysis: structured numeric breakdown (consumed by
        critic.py's deterministic rule_based_checks, which needs a
        real total_cost to compare against trip_constraints["budget"])
    """

    print("\n========== BUDGET AGENT INPUT ==========")

    print("Trip Constraints:")
    print(
        state.get(
            "trip_constraints",
            {},
        )
    )

    print("\nFlight Results:")
    print(
        state.get(
            "flight_results",
            "",
        )
    )

    print("\nHotel Results:")
    print(
        state.get(
            "hotel_results",
            "",
        )
    )

    print("\nWeather Results:")
    print(
        state.get(
            "weather_results",
            "",
        )
    )

    print("=========================================\n")

    prompt = f"""
Analyze whether this trip plan is realistic for the user's budget.

User request:
{state.get("user_query", "")}

Trip constraints:
{state.get("trip_constraints", {})}

Flight results:
{_truncate(state.get("flight_results", ""))}

Hotel results:
{_truncate(state.get("hotel_results", ""))}

Weather results:
{_truncate(state.get("weather_results", ""), max_chars=800)}

Return ONLY valid JSON using exactly this schema. All cost fields are
your best numeric ESTIMATES in the same currency implied by trip
constraints (assume INR if unspecified) — use plain numbers, no
commas or currency symbols:

{{
  "total_cost": 0,
  "currency": "INR",
  "categories": {{
    "flights": 0,
    "hotel": 0,
    "food_and_transport": 0,
    "activities": 0
  }},
  "risk_areas": [],
  "money_saving_suggestions": [],
  "feasible": true,
  "narrative": ""
}}

"narrative" should be a concise prose assessment covering:
1. Estimated cost categories
2. Flight cost estimate
3. Hotel/accommodation estimate
4. Food and local transportation estimate
5. Activity/sightseeing estimate
6. Risk areas
7. Money-saving suggestions
8. Whether the trip appears feasible

Clearly distinguish estimates from confirmed prices within the narrative.
"""

    raw = _llm_text(
        "You are a practical travel budget analyst. Return strict JSON only.",
        prompt,
    )

    try:
        parsed = _json_from_llm(raw)
    except ValueError:
        # Fail open: keep the raw prose as budget_results, but leave
        # budget_analysis empty so the critic's numeric check simply
        # skips (rather than crashing the graph run).
        print("\n========== BUDGET AGENT: JSON PARSE FAILED, FALLING BACK ==========")
        print("==================================================================\n")

        return {
            "budget_results": raw,
            "budget_analysis": {},
            "messages": [AIMessage(content="Budget agent completed (fallback).")],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    narrative = parsed.get("narrative", "") or raw

    budget_analysis = {
        "total_cost": parsed.get("total_cost"),
        "currency": parsed.get("currency", "INR"),
        "categories": parsed.get("categories", {}),
        "risk_areas": parsed.get("risk_areas", []),
        "money_saving_suggestions": parsed.get("money_saving_suggestions", []),
        "feasible": parsed.get("feasible"),
    }

    print("\n========== BUDGET AGENT OUTPUT ==========")
    print("Narrative:", narrative)
    print("Structured analysis:", json.dumps(budget_analysis, indent=2, default=str))
    print("=========================================\n")

    return {
        "budget_results": narrative,
        "budget_analysis": budget_analysis,
        "messages": [AIMessage(content="Budget agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# REPLAN HELPERS
# ============================================================

REPLANNABLE = {"flight_agent", "hotel_agent", "weather_agent", "budget_agent"}


def prepare_replan(state: TravelState):
    """
    Narrow selected_agents to only what the critic flagged and mark this
    as a replan pass so supervisor_agent doesn't re-derive from user_query.
    """

    verdict = state.get("critic_verdict", {})
    responsible = [a for a in verdict.get("responsible_agents", []) if a in REPLANNABLE]

    print("\n========== PREPARING REPLAN ==========")
    print("Responsible agents:", responsible)
    print("=======================================\n")

    return {
        "selected_agents": responsible or ["itinerary_agent"],
        "is_replan": True,
        "unresolved_violations": [],
        "messages": [AIMessage(content=f"Replanning via: {responsible}")],
    }


def mark_unresolved(state: TravelState):
    """
    Loop maxed out — fail open, forward to human_approval flagged rather
    than blocking the user indefinitely.
    """

    verdict = state.get("critic_verdict", {})

    return {
        "unresolved_violations": verdict.get("violations", []),
        "messages": [
            AIMessage(
                content="Max critic iterations reached; forwarding for human review."
            )
        ],
    }


# ============================================================
# ITINERARY AGENT
# ============================================================


def itinerary_agent(state: TravelState):
    """
    Combine all specialist outputs into a draft itinerary.
    """

    print("\n========== ITINERARY AGENT INPUT ==========")

    print("Trip Constraints:")
    print(
        state.get(
            "trip_constraints",
            {},
        )
    )

    print("\nFlight Results:")
    print(
        state.get(
            "flight_results",
            "",
        )
    )

    print("\nHotel Results:")
    print(
        state.get(
            "hotel_results",
            "",
        )
    )

    print("\nWeather Results:")
    print(
        state.get(
            "weather_results",
            "",
        )
    )

    print("\nBudget Results:")
    print(
        state.get(
            "budget_results",
            "",
        )
    )

    print("===========================================\n")

    prompt = f"""
Create a clear draft travel itinerary.

User request:
{state.get("user_query", "")}

Trip constraints:
{state.get("trip_constraints", {})}

Flight results:
{_truncate(state.get("flight_results", ""))}

Hotel results:
{_truncate(state.get("hotel_results", ""))}

Weather results:
{_truncate(state.get("weather_results", ""), max_chars=800)}

Budget results:
{_truncate(state.get("budget_results", ""))}

Create a practical itinerary.

Structure the answer as:

- Trip overview
- Day-by-day itinerary
- Flights / transportation
- Accommodation recommendations
- Food recommendations
- Activities
- Estimated daily spending
- Weather considerations
- Important travel tips
- Budget summary

Do not invent confirmed bookings.
Clearly identify estimates and recommendations.
"""

    result = _llm_text(
        "You are an expert itinerary planner.",
        prompt,
    )

    print("\n========== ITINERARY OUTPUT ==========")
    print(result)
    print("======================================\n")

    approval_request = f"""
Please review this draft travel plan.

{result}

Reply with approval or feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HUMAN APPROVAL AGENT
# ============================================================


def human_approval_agent(state: TravelState):
    """
    Pause the graph and request human approval.
    """

    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get(
                "itinerary",
                "",
            ),
            "approval_request": state.get(
                "approval_request",
                "",
            ),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            },
        }
    )

    # --------------------------------------------------------
    # Defensive parsing
    # --------------------------------------------------------

    if not isinstance(feedback, dict):
        raise ValueError("Human approval response must be a dictionary.")

    approved = bool(
        feedback.get(
            "approved",
            False,
        )
    )

    human_feedback = feedback.get(
        "feedback",
        "",
    )

    print("\n========== HUMAN APPROVAL ==========")
    print("Approved:", approved)
    print("Feedback:", human_feedback)
    print("====================================\n")

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================


def final_response_agent(state: TravelState):
    """
    Generate the final user-facing travel plan.
    """

    approved = state.get(
        "approved",
        False,
    )

    human_feedback = state.get(
        "human_feedback",
        "",
    )

    user_query = state.get("user_query", "")

    print("\n========== FINAL AGENT INPUT ==========")
    print("Approved:", approved)
    print("Feedback:", human_feedback)
    print("=======================================\n")

    # --------------------------------------------------------
    # Approved itinerary
    # --------------------------------------------------------

    if approved:
        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

Original user request:
{user_query}

Trip constraints:
{state.get("trip_constraints", {})}

Draft itinerary:
{_truncate(state.get("itinerary", ""), max_chars=3000)}

Budget notes:
{_truncate(state.get("budget_results", ""))}

Human feedback:
{human_feedback}

Create a clear, practical, user-ready final travel plan.

Include:

1. Trip overview
2. Day-by-day itinerary
3. Transportation
4. Accommodation
5. Food
6. Activities
7. Budget
8. Weather considerations
9. Travel tips

Do not claim that anything is booked unless the system
actually confirmed a booking.
"""

    # --------------------------------------------------------
    # Rejected itinerary
    # --------------------------------------------------------

    else:
        prompt = f"""
The human did not approve the draft itinerary.

Create a revised travel plan using the human's feedback.

Original user request:
{user_query}

Trip constraints:
{state.get("trip_constraints", {})}

Previous draft:
{_truncate(state.get("itinerary", ""), max_chars=3000)}

Human feedback:
{human_feedback}

Budget notes:
{_truncate(state.get("budget_results", ""))}

Revise the itinerary according to the feedback.

Clearly explain the improved plan and keep it practical.
Do not claim that anything is booked unless the system
actually confirmed a booking.
"""

    result = _llm_text(
        "You produce final user-ready travel plans.",
        prompt,
    )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
