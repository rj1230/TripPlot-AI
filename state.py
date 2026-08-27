from typing import Annotated, Any, TypedDict
import operator

from langchain_core.messages import AnyMessage


class TravelState(TypedDict, total=False):
    # ------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------
    messages: Annotated[list[AnyMessage], operator.add]

    user_id: str
    user_query: str

    # ------------------------------------------------------------
    # Planning / Supervisor
    # ------------------------------------------------------------
    trip_constraints: dict[str, Any]
    selected_agents: list[str]
    supervisor_reasoning: str

    # ------------------------------------------------------------
    # Specialist agent outputs
    # ------------------------------------------------------------
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str

    # ------------------------------------------------------------
    # Structured budget information
    #
    # This allows the critic to perform deterministic budget checks
    # instead of trying to extract numbers from LLM prose.
    # ------------------------------------------------------------
    budget_analysis: dict[str, Any]

    # ------------------------------------------------------------
    # Final itinerary
    # ------------------------------------------------------------
    itinerary: str
    approval_request: str

    # ------------------------------------------------------------
    # Human approval
    # ------------------------------------------------------------
    human_feedback: str
    approved: bool

    # ------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------
    final_response: str

    # ------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------
    llm_calls: int

    # ------------------------------------------------------------
    # Critic / Replanning
    # ------------------------------------------------------------
    iteration_count: int

    critic_verdict: dict[str, Any]

    unresolved_violations: list[str]

    # True only while supervisor is entering a targeted replan.
    is_replan: bool


# DATABASE_URL = postgresql://travel_agent_u9vy_user:D9eSkwswMmwwoZdsPIKWevenbLJdhXrC@dpg-da6jjqv10e5c73bukp70-a.virginia-postgres.render.com/travel_agent_u9vy
