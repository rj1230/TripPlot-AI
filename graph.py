from __future__ import annotations

import asyncio
from typing import Any

import psycopg

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    budget_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    llm,
    mark_unresolved,
    prepare_replan,
    supervisor_agent,
    weather_agent,
)

from config import DATABASE_URL
from critic import critic_node
from state import TravelState


# ============================================================
# CONFIGURATION
# ============================================================

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

ROUTE_MAP = {
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}

CRITIC_ROUTE_MAP = {
    "human_approval": "human_approval",
    "mark_unresolved": "mark_unresolved",
    "prepare_replan": "prepare_replan",
}

MAX_ITERATIONS = 3


# ============================================================
# HELPERS
# ============================================================


def _selected_agents(state: TravelState) -> list[str]:
    """
    Return selected agents in deterministic execution order.

    The supervisor may return agents in any order. We always execute
    them according to AGENT_ORDER.
    """

    selected = state.get("selected_agents", [])

    if not selected:
        return []

    return [agent for agent in AGENT_ORDER if agent in selected]


# ============================================================
# SUPERVISOR ROUTING
# ============================================================


def route_from_supervisor(state: TravelState) -> str:
    """
    Supervisor -> first selected specialist agent.
    """

    selected = _selected_agents(state)

    if selected:
        return selected[0]

    # Always generate an itinerary if supervisor selected nothing.
    return "itinerary_agent"


def route_after_agent(current_agent: str):
    """
    Create routing function for specialist agents.

    Example:

        flight -> hotel -> weather -> budget -> itinerary

    Only selected agents are executed.
    """

    def route(state: TravelState) -> str:
        selected = _selected_agents(state)

        try:
            current_index = AGENT_ORDER.index(current_agent)
        except ValueError:
            return "itinerary_agent"

        # Find the next selected specialist.
        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        # Always finish specialist execution with itinerary.
        return "itinerary_agent"

    return route


# ============================================================
# CRITIC ROUTING
# ============================================================


def route_after_critic(state: TravelState) -> str:
    """
    Critic routing:

        PASS
            -> human_approval

        FAIL + retries remaining
            -> prepare_replan

        FAIL + max retries reached
            -> mark_unresolved
    """

    verdict = state.get("critic_verdict", {})

    passed = bool(verdict.get("passed", False))

    iteration_count = int(state.get("iteration_count", 0))

    # --------------------------------------------------------
    # Critic passed
    # --------------------------------------------------------

    if passed:
        return "human_approval"

    # --------------------------------------------------------
    # Maximum retry count reached
    # --------------------------------------------------------

    if iteration_count >= MAX_ITERATIONS:
        return "mark_unresolved"

    # --------------------------------------------------------
    # Retry / replan
    # --------------------------------------------------------

    return "prepare_replan"


# ============================================================
# BUILD GRAPH
# ============================================================


def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph.

    IMPORTANT:
    This function returns the graph BUILDER.

    Do NOT call:

        build_graph().ainvoke(...)

    Instead use:

        await compile_graph_and_invoke(...)

    or:

        graph = build_graph()
        app = graph.compile(...)
        await app.ainvoke(...)
    """

    graph = StateGraph(TravelState)

    # ========================================================
    # NODES
    # ========================================================

    graph.add_node(
        "supervisor",
        supervisor_agent,
    )

    graph.add_node(
        "flight_agent",
        flight_agent,
    )

    graph.add_node(
        "hotel_agent",
        hotel_agent,
    )

    graph.add_node(
        "weather_agent",
        weather_agent,
    )

    graph.add_node(
        "budget_agent",
        budget_agent,
    )

    graph.add_node(
        "itinerary_agent",
        itinerary_agent,
    )

    graph.add_node(
        "critic",
        lambda state: critic_node(state, llm),
    )

    graph.add_node(
        "prepare_replan",
        prepare_replan,
    )

    graph.add_node(
        "mark_unresolved",
        mark_unresolved,
    )

    graph.add_node(
        "human_approval",
        human_approval_agent,
    )

    graph.add_node(
        "final_response",
        final_response_agent,
    )

    # ========================================================
    # START
    # ========================================================

    graph.add_edge(
        START,
        "supervisor",
    )

    # ========================================================
    # SUPERVISOR -> SPECIALIST
    # ========================================================

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ROUTE_MAP,
    )

    # ========================================================
    # SPECIALIST ROUTING
    # ========================================================

    graph.add_conditional_edges(
        "flight_agent",
        route_after_agent("flight_agent"),
        ROUTE_MAP,
    )

    graph.add_conditional_edges(
        "hotel_agent",
        route_after_agent("hotel_agent"),
        ROUTE_MAP,
    )

    graph.add_conditional_edges(
        "weather_agent",
        route_after_agent("weather_agent"),
        ROUTE_MAP,
    )

    graph.add_conditional_edges(
        "budget_agent",
        route_after_agent("budget_agent"),
        ROUTE_MAP,
    )

    # ========================================================
    # ITINERARY -> CRITIC
    # ========================================================

    graph.add_edge(
        "itinerary_agent",
        "critic",
    )

    # ========================================================
    # CRITIC ROUTING
    # ========================================================

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        CRITIC_ROUTE_MAP,
    )

    # ========================================================
    # REPLAN
    # ========================================================

    graph.add_edge(
        "prepare_replan",
        "supervisor",
    )

    # ========================================================
    # UNRESOLVED -> HUMAN
    # ========================================================

    graph.add_edge(
        "mark_unresolved",
        "human_approval",
    )

    # ========================================================
    # HUMAN APPROVAL -> FINAL
    # ========================================================

    graph.add_edge(
        "human_approval",
        "final_response",
    )

    # ========================================================
    # FINAL -> END
    # ========================================================

    graph.add_edge(
        "final_response",
        END,
    )

    return graph


# ============================================================
# COMPILE
# ============================================================
#
# IMPORTANT — CHECKPOINTER LIFETIME BUG FIX
# ------------------------------------------------------------
# run_graph() is called twice per trip: once for the initial
# request (which pauses at the human_approval interrupt), and
# once again on resume after the user submits their approval.
#
# The interrupt/resume mechanism only works if BOTH calls use
# the SAME checkpointer instance for a given thread_id, because
# that's where LangGraph stores the paused state.
#
# - Postgres (AsyncPostgresSaver): safe to build a new connection
#   per call, because the checkpoint DATA lives in the database,
#   not in the Python connection object. A fresh connection can
#   still read a thread_id's prior checkpoint.
#
# - MemorySaver: stores checkpoints in an in-process Python dict
#   that belongs to that one object. Creating `MemorySaver()`
#   fresh on every call (as before) wiped all prior checkpoints
#   before the resume call could ever see them — so on resume,
#   LangGraph had nothing to continue from, restarted the graph
#   from START with the bare Command(resume=...) as input, and
#   supervisor_agent crashed on state["user_query"] because that
#   input never contained the original request.
#
# Fix: keep a single module-level MemorySaver (and its compiled
# app) alive for the lifetime of the process, and reuse it on
# every call when no DATABASE_URL is configured.
# ============================================================

_memory_checkpointer: MemorySaver | None = None
_memory_app = None
_compile_lock = asyncio.Lock()


async def compile_graph():
    """
    Compile the graph into a runnable LangGraph application.

    If DATABASE_URL exists:
        AsyncPostgresSaver (new connection per call; checkpoint data
        persists in the database itself, so this is safe).

    Otherwise:
        A single, process-wide MemorySaver reused across calls, so
        that the human-approval interrupt/resume cycle actually has
        somewhere to resume from.
    """

    global _memory_checkpointer, _memory_app

    graph = build_graph()

    # --------------------------------------------------------
    # PostgreSQL
    # --------------------------------------------------------

    if DATABASE_URL:
        conn = await psycopg.AsyncConnection.connect(
            DATABASE_URL,
            autocommit=True,
        )

        checkpointer = AsyncPostgresSaver(conn)

        # Create checkpoint tables if necessary.
        await checkpointer.setup()

        return graph.compile(checkpointer=checkpointer), conn

    # --------------------------------------------------------
    # In-memory fallback (singleton — see note above)
    # --------------------------------------------------------

    async with _compile_lock:
        if _memory_app is None:
            _memory_checkpointer = MemorySaver()
            _memory_app = graph.compile(checkpointer=_memory_checkpointer)

    return _memory_app, None


# ============================================================
# RUN GRAPH
# ============================================================


async def run_graph(
    input_data: Any,
    config: dict,
):
    """
    Run the compiled graph asynchronously.

    This is the function Streamlit should call.

    PostgreSQL connections are created and closed per invocation.
    This avoids Streamlit's multiple-event-loop problems. The
    in-memory checkpointer (used when DATABASE_URL is unset) is a
    process-wide singleton reused across calls — see compile_graph().
    """

    conn = None

    try:
        app, conn = await compile_graph()

        result = await app.ainvoke(
            input_data,
            config=config,
        )

        return result

    finally:
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass


# ============================================================
# CLI TEST
# ============================================================


async def _test_graph():

    print("=" * 70)
    print("BUILDING TRAVEL PLANNER GRAPH")
    print("=" * 70)

    graph = build_graph()

    print("StateGraph created successfully.")

    app = None
    conn = None

    try:
        if DATABASE_URL:
            print("Using PostgreSQL checkpointer.")

            conn = await psycopg.AsyncConnection.connect(
                DATABASE_URL,
                autocommit=True,
            )

            checkpointer = AsyncPostgresSaver(conn)

            await checkpointer.setup()

            app = graph.compile(checkpointer=checkpointer)

        else:
            print("DATABASE_URL not configured.")
            print("Using MemorySaver.")

            app = graph.compile(checkpointer=MemorySaver())

        print("Graph compiled successfully.")
        print("Runnable type:", type(app).__name__)
        print("ainvoke available:", hasattr(app, "ainvoke"))

    finally:
        if conn is not None:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(_test_graph())
