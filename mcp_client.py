from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import (
    AVIATION_STACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)

logger = logging.getLogger(__name__)


# ==============================================================
# PROJECT PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
WEATHER_SERVER_SCRIPT = PROJECT_ROOT / "weather_mcp_server.py"


if not WEATHER_SERVER_SCRIPT.exists():
    raise FileNotFoundError(f"Weather MCP server not found: {WEATHER_SERVER_SCRIPT}")


# ==============================================================
# ENVIRONMENT
# ==============================================================


def _merged_env(**overrides: str | None) -> dict[str, str]:
    """
    Preserve the parent process environment and override only the
    variables required by the MCP subprocess.
    """
    env = os.environ.copy()

    for key, value in overrides.items():
        if value:
            env[key] = value

    return env


# ==============================================================
# API KEY VALIDATION
# ==============================================================


def validate_mcp_configuration() -> None:
    """
    Validate required MCP configuration before the application
    attempts to initialize external services.
    """

    missing: list[str] = []

    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")

    if not AVIATION_STACK_API_KEY:
        missing.append("AVIATION_STACK_API_KEY")

    if not OPENWEATHER_API_KEY:
        missing.append("OPENWEATHER_API_KEY")

    if missing:
        raise RuntimeError("Missing required MCP configuration: " + ", ".join(missing))


# ==============================================================
# MCP CLIENT
# ==============================================================

validate_mcp_configuration()

client = MultiServerMCPClient(
    {
        # ----------------------------------------------------------
        # Tavily
        # ----------------------------------------------------------
        "tavily": {
            "transport": "streamable_http",
            "url": (f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"),
        },
        # ----------------------------------------------------------
        # AviationStack
        # ----------------------------------------------------------
        "aviationstack": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "--with",
                "mcp<2",
                "aviationstack-mcp",
            ],
            "env": _merged_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },
        # ----------------------------------------------------------
        # Weather
        # ----------------------------------------------------------
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                str(WEATHER_SERVER_SCRIPT),
            ],
            "env": _merged_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


# ==============================================================
# TOOL CACHE
# ==============================================================

_tools_cache: list[Any] | None = None
_tool_registry: dict[str, Any] | None = None


async def get_tools() -> list[Any]:
    """
    Load MCP tools once and cache them.

    This prevents every tool invocation from requiring a complete
    tool discovery pass.
    """

    global _tools_cache

    if _tools_cache is not None:
        return _tools_cache

    try:
        logger.info("Loading MCP tools...")

        _tools_cache = await client.get_tools()

        logger.info(
            "Loaded %d MCP tools: %s",
            len(_tools_cache),
            ", ".join(tool.name for tool in _tools_cache),
        )

        return _tools_cache

    except Exception:
        logger.exception("Failed to initialize MCP tools")
        raise


async def get_tool_registry() -> dict[str, Any]:
    """
    Return MCP tools indexed by name.
    """

    global _tool_registry

    if _tool_registry is not None:
        return _tool_registry

    tools = await get_tools()

    _tool_registry = {tool.name: tool for tool in tools}

    return _tool_registry


# ==============================================================
# GENERIC TOOL INVOCATION
# ==============================================================


async def call_tool(
    tool_name: str,
    args: dict[str, Any] | None = None,
) -> Any:
    """
    Invoke an MCP tool by name.

    Centralizing invocation here gives the project one place for:
    - logging
    - validation
    - error handling
    - future retries
    - latency tracking
    - tracing
    """

    registry = await get_tool_registry()

    tool = registry.get(tool_name)

    if tool is None:
        available = sorted(registry.keys())

        raise ValueError(
            f"MCP tool '{tool_name}' not found. Available tools: {available}"
        )

    payload = args or {}

    logger.info(
        "MCP tool call: %s | args=%s",
        tool_name,
        payload,
    )

    try:
        result = await tool.ainvoke(payload)

        logger.info(
            "MCP tool completed: %s",
            tool_name,
        )

        return result

    except Exception:
        logger.exception(
            "MCP tool failed: %s | args=%s",
            tool_name,
            payload,
        )
        raise


# ==============================================================
# TAVILY
# ==============================================================


async def tavily_search(query: str) -> Any:
    if not query.strip():
        raise ValueError("Tavily search query cannot be empty")

    return await call_tool(
        "tavily_search",
        {
            "query": query,
        },
    )


# ==============================================================
# AVIATION
# ==============================================================


async def list_airports(
    search: str = "",
    limit: int = 10,
) -> Any:
    limit = max(1, min(limit, 50))

    return await call_tool(
        "list_airports",
        {
            "search": search,
            "limit": limit,
            "offset": 0,
        },
    )


async def list_airlines(
    search: str = "",
    limit: int = 10,
) -> Any:
    limit = max(1, min(limit, 50))

    return await call_tool(
        "list_airlines",
        {
            "search": search,
            "limit": limit,
            "offset": 0,
        },
    )


# ==============================================================
# WEATHER
# ==============================================================


async def current_weather(city: str) -> Any:
    if not city.strip():
        raise ValueError("Weather city cannot be empty")

    return await call_tool(
        "get_current_weather",
        {
            "city": city,
        },
    )


async def forecast(city: str) -> Any:
    if not city.strip():
        raise ValueError("Forecast city cannot be empty")

    return await call_tool(
        "get_forecast",
        {
            "city": city,
        },
    )


# ==============================================================
# HEALTH CHECK
# ==============================================================


async def check_mcp_health() -> dict[str, Any]:
    """
    Lightweight MCP health check.

    Returns available tool names rather than calling expensive
    external APIs.
    """

    try:
        tools = await get_tools()

        return {
            "status": "healthy",
            "tool_count": len(tools),
            "tools": sorted(tool.name for tool in tools),
        }

    except Exception as exc:
        logger.exception("MCP health check failed")

        return {
            "status": "unhealthy",
            "tool_count": 0,
            "tools": [],
            "error": str(exc),
        }


# ==============================================================
# CLI TEST
# ==============================================================

if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        health = await check_mcp_health()

        print("\nMCP HEALTH")
        print("=" * 50)
        print(health)

    asyncio.run(_main())
