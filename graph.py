from __future__ import annotations

import asyncio
import logging
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

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

SPECIALIST_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

REPLANNABLE_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
}

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
# CHECKPOINTER STATE
# ============================================================

_memory_checkpointer: MemorySaver | None = None
_memory_app = None

_compile_lock = asyncio.Lock()


# ============================================================
# HELPERS
# ============================================================


def _selected_agents(state: TravelState) -> list[str]:
    """
    Return valid selected agents in deterministic execution order.

    The supervisor may return agents in arbitrary order, but the
    orchestration layer always executes them in AGENT_ORDER.
    """

    selected = state.get("selected_agents", []) or []

    if not isinstance(selected, list):
        logger.warning(
            "selected_agents is not a list: %r",
            selected,
        )
        return []

    selected_set = {agent for agent in selected if agent in SPECIALIST_AGENTS}

    ordered = [agent for agent in AGENT_ORDER if agent in selected_set]

    logger.info(
        "Validated selected agents: %s",
        ordered,
    )

    return ordered


def _current_iteration(state: TravelState) -> int:
    """
    Safely retrieve the current critic/replan iteration.
    """

    try:
        return max(
            0,
            int(
                state.get(
                    "iteration_count",
                    0,
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        return 0


# ============================================================
# SUPERVISOR ROUTING
# ============================================================


def route_from_supervisor(state: TravelState) -> str:
    """
    Supervisor -> first selected specialist.

    If no valid specialist is selected, safely fall back to
    itinerary generation.
    """

    selected = _selected_agents(state)

    if selected:
        first = selected[0]

        logger.info(
            "Supervisor routing -> %s",
            first,
        )

        return first

    logger.warning(
        "Supervisor selected no valid agents. Falling back to itinerary_agent."
    )

    return "itinerary_agent"


def route_after_agent(current_agent: str):
    """
    Deterministic specialist-agent routing.

    Example:

        flight
          ↓
        hotel
          ↓
        weather
          ↓
        budget
          ↓
        itinerary

    Only agents selected by the supervisor/replanner execute.
    """

    def route(state: TravelState) -> str:
        selected = _selected_agents(state)

        if current_agent not in AGENT_ORDER:
            logger.error(
                "Unknown current agent: %s",
                current_agent,
            )
            return "itinerary_agent"

        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                logger.info(
                    "%s -> %s",
                    current_agent,
                    next_agent,
                )
                return next_agent

        logger.info(
            "%s -> itinerary_agent",
            current_agent,
        )

        return "itinerary_agent"

    return route


# ============================================================
# CRITIC ROUTING
# ============================================================


def route_after_critic(state: TravelState) -> str:
    """
    Route based on the critic's quality decision.

    PASS:
        Human approval.

    REVIEW:
        Human approval with warnings.

    REPLAN:
        Targeted replanning unless the iteration limit is reached.

    IMPORTANT:
        iteration_count is owned by critic.py.
        Do NOT increment it here.
    """

    verdict = state.get("critic_verdict", {}) or {}

    if not isinstance(verdict, dict):
        logger.error(
            "critic_verdict is not a dictionary: %r",
            verdict,
        )

        return "mark_unresolved"

    decision = str(verdict.get("decision", "")).upper().strip()

    passed = bool(verdict.get("passed", False))

    iteration_count = _current_iteration(state)

    logger.info(
        "Critic decision=%s passed=%s iteration=%s/%s",
        decision,
        passed,
        iteration_count,
        MAX_ITERATIONS,
    )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    if decision == "PASS" or (not decision and passed):
        logger.info("Critic PASS -> human approval.")

        return "human_approval"

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------
    #
    # REVIEW means the plan is not clean enough for automatic
    # approval, but it should be presented to the human rather
    # than automatically regenerated.
    # --------------------------------------------------------

    if decision == "REVIEW":
        logger.warning("Critic REVIEW -> human approval.")

        return "human_approval"

    # --------------------------------------------------------
    # MAX ITERATIONS
    # --------------------------------------------------------

    if iteration_count >= MAX_ITERATIONS:
        logger.warning(
            "Critic iteration limit reached: %s/%s",
            iteration_count,
            MAX_ITERATIONS,
        )

        return "mark_unresolved"

    # --------------------------------------------------------
    # REPLAN
    # --------------------------------------------------------

    if decision == "REPLAN":
        logger.warning("Critic REPLAN -> targeted replanning.")

        return "prepare_replan"

    # --------------------------------------------------------
    # LEGACY FALLBACK
    # --------------------------------------------------------
    #
    # If an older critic only returns passed=False, preserve
    # backward compatibility by treating it as a replan request.
    # --------------------------------------------------------

    if not decision and not passed:
        logger.warning("Legacy critic verdict detected. Routing to targeted replan.")

        return "prepare_replan"

    # --------------------------------------------------------
    # UNKNOWN DECISION
    # --------------------------------------------------------

    logger.error(
        "Unknown critic decision=%r. Failing safely to human review.",
        decision,
    )

    return "human_approval"


# ============================================================
# CRITIC WRAPPER
# ============================================================


def critic_agent_node(
    state: TravelState,
):
    """
    Thin orchestration wrapper around critic_node.

    The critic owns quality evaluation and iteration_count.
    """

    logger.info("Critic evaluation started.")

    result = critic_node(
        state,
        llm,
    )

    if not isinstance(result, dict):
        raise TypeError("critic_node must return a dictionary.")

    verdict = result.get(
        "critic_verdict",
        {},
    )

    if isinstance(verdict, dict):
        logger.info(
            "Critic completed: decision=%s quality=%s confidence=%s",
            verdict.get("decision"),
            verdict.get("quality_score"),
            verdict.get("confidence"),
        )

    return result


# ============================================================
# BUILD GRAPH
# ============================================================


def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph.

    The builder is intentionally separate from compilation so it
    can be inspected and tested independently.
    """

    graph = StateGraph(TravelState)

    # ========================================================
    # SPECIALIST NODES
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

    # ========================================================
    # QUALITY CONTROL
    # ========================================================

    graph.add_node(
        "critic",
        critic_agent_node,
    )

    graph.add_node(
        "prepare_replan",
        prepare_replan,
    )

    graph.add_node(
        "mark_unresolved",
        mark_unresolved,
    )

    # ========================================================
    # HUMAN / OUTPUT
    # ========================================================

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
    # SUPERVISOR -> FIRST SPECIALIST
    # ========================================================

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ROUTE_MAP,
    )

    # ========================================================
    # SPECIALIST PIPELINE
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
    # CRITIC -> QUALITY DECISION
    # ========================================================

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        CRITIC_ROUTE_MAP,
    )

    # ========================================================
    # TARGETED REPLAN LOOP
    #
    # critic
    #    ↓
    # prepare_replan
    #    ↓
    # supervisor
    #    ↓
    # selected specialist agents
    #    ↓
    # itinerary
    #    ↓
    # critic
    #
    # IMPORTANT:
    # critic.py owns iteration_count.
    # There is deliberately NO replan_iteration node.
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
    # HUMAN -> FINAL
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

    logger.info("TravelPlanner StateGraph constructed successfully.")

    return graph


# ============================================================
# CHECKPOINTER / COMPILATION
# ============================================================


async def compile_graph():
    """
    Compile the graph into a runnable LangGraph application.

    PostgreSQL:
        A fresh async connection is created for the invocation.

    MemorySaver:
        A process-wide singleton is reused so interrupt/resume
        works across Streamlit requests.
    """

    global _memory_checkpointer
    global _memory_app

    # --------------------------------------------------------
    # PostgreSQL
    # --------------------------------------------------------

    if DATABASE_URL:
        logger.info("Compiling graph with PostgreSQL checkpointer.")

        graph = build_graph()

        conn = await psycopg.AsyncConnection.connect(
            DATABASE_URL,
            autocommit=True,
        )

        checkpointer = AsyncPostgresSaver(conn)

        await checkpointer.setup()

        app = graph.compile(checkpointer=checkpointer)

        return app, conn

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    async with _compile_lock:
        if _memory_app is None:
            logger.warning(
                "DATABASE_URL is not configured. Using process-local MemorySaver."
            )

            _memory_checkpointer = MemorySaver()

            graph = build_graph()

            _memory_app = graph.compile(checkpointer=_memory_checkpointer)

        return (
            _memory_app,
            None,
        )


# ============================================================
# GRAPH INVOCATION
# ============================================================


async def run_graph(
    input_data: Any,
    config: dict,
):
    """
    Primary graph execution entry point.

    A stable thread_id is mandatory because the graph contains
    human interrupt/resume behavior.
    """

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError("config must be a dictionary.")

    configurable = config.get(
        "configurable",
        {},
    )

    if not isinstance(
        configurable,
        dict,
    ):
        raise TypeError("config['configurable'] must be a dictionary.")

    thread_id = configurable.get("thread_id")

    if not thread_id:
        raise ValueError(
            "run_graph requires "
            "config['configurable']['thread_id'] "
            "for checkpointed execution and "
            "human approval."
        )

    conn = None

    try:
        app, conn = await compile_graph()

        logger.info(
            "Invoking TravelPlanner graph: thread_id=%s",
            thread_id,
        )

        result = await app.ainvoke(
            input_data,
            config=config,
        )

        logger.info(
            "TravelPlanner graph completed: thread_id=%s",
            thread_id,
        )

        return result

    finally:
        if conn is not None:
            try:
                await conn.close()

            except Exception as exc:
                logger.warning(
                    "Failed to close PostgreSQL connection: %s",
                    exc,
                )


# ============================================================
# GRAPH INSPECTION
# ============================================================


def get_graph():
    """
    Return a compiled graph using the default in-memory
    compilation path.

    This helper is intended for inspection/testing.

    Production execution should use compile_graph().
    """

    graph = build_graph()

    return graph.compile()


# ============================================================
# CLI GRAPH TEST
# ============================================================


async def _test_graph():

    print("=" * 70)
    print("TRIPPLOT-AI GRAPH VALIDATION")
    print("=" * 70)

    graph = build_graph()

    print("StateGraph created successfully.")

    app = None
    conn = None

    try:
        if DATABASE_URL:
            print("Checkpointer: PostgreSQL")

            conn = await psycopg.AsyncConnection.connect(
                DATABASE_URL,
                autocommit=True,
            )

            checkpointer = AsyncPostgresSaver(conn)

            await checkpointer.setup()

            app = graph.compile(checkpointer=checkpointer)

        else:
            print("Checkpointer: MemorySaver")

            checkpointer = MemorySaver()

            app = graph.compile(checkpointer=checkpointer)

        print("Graph compiled successfully.")

        print(
            "Runnable type:",
            type(app).__name__,
        )

        print(
            "ainvoke available:",
            hasattr(
                app,
                "ainvoke",
            ),
        )

        print("Graph validation: PASS")

    except Exception as exc:
        print("Graph validation: FAIL")

        print(f"Error: {exc}")

        raise

    finally:
        if conn is not None:
            await conn.close()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    asyncio.run(_test_graph())
