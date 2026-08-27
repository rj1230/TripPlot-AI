import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import (
    AVIATION_STACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)

# --------------------------------------------------------------
# Resolve the local weather MCP server relative to THIS project,
# instead of hardcoding a machine-specific absolute path (the old
# "C:\Users\HP\OneDrive\Desktop\..." path only worked on one laptop
# and breaks on any other machine, CI runner, or container).
#
# sys.executable is used for the interpreter so this also works
# correctly inside whatever venv the project is actually launched
# from, on Windows, macOS, or Linux.
# --------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
WEATHER_SERVER_SCRIPT = PROJECT_ROOT / "weather_mcp_server.py"


# Create MCP Client
client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },
        "aviationstack": {
            "transport": "stdio",
            # Uses uvx to run the published aviationstack-mcp package in its own
            # isolated env. The package itself is pinned to the mcp v1 API
            # (mcp.server.fastmcp.FastMCP), which mcp v2 renamed to MCPServer —
            # `uvx` resolves the newest mcp by default and breaks the import,
            # so --with "mcp<2" pins the dependency this specific package needs.
            "command": "uvx",
            "args": ["--with", "mcp<2", "aviationstack-mcp"],
            # IMPORTANT: passing `env` to a subprocess REPLACES the entire
            # environment rather than adding to it. A dict containing only
            # the API key strips PATH (so "uvx" can't even be located on
            # Windows) and system vars the child process needs to start at
            # all. Merge onto a copy of the parent environment instead.
            "env": {**os.environ, "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY},
        },
        "weather": {
            "transport": "stdio",
            # Launch with the interpreter that's actually running this process
            # and a path resolved relative to this file, so it works regardless
            # of machine, OS, or whether this is invoked from a venv.
            "command": sys.executable,
            "args": [str(WEATHER_SERVER_SCRIPT)],
            # Same env-merge fix as aviationstack above.
            "env": {**os.environ, "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY},
        },
    }
)


# Cache tools so we don't load them repeatedly
_tools_cache = None


async def get_tools():
    global _tools_cache

    if _tools_cache is None:
        try:
            _tools_cache = await client.get_tools()

        except Exception as e:
            print("\n========== FULL ERROR ==========")
            print(type(e))
            print(repr(e))

            if hasattr(e, "exceptions"):
                print("\nSUB EXCEPTIONS:")
                for i, sub in enumerate(e.exceptions):
                    print(f"\n--- Exception {i + 1} ---")
                    print(type(sub))
                    print(repr(sub))

            raise

    return _tools_cache


async def call_tool(tool_name: str, args: dict = None):
    tools = await get_tools()

    tool = next(
        (tool for tool in tools if tool.name == tool_name),
        None,
    )

    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    return await tool.ainvoke(args or {})


# ------------------------
# Tavily MCP Tools
# ------------------------


async def tavily_search(query: str):
    return await call_tool("tavily_search", {"query": query})


async def list_airports(search: str = "", limit: int = 10):
    return await call_tool(
        "list_airports", {"search": search, "limit": limit, "offset": 0}
    )


async def list_airlines(search: str = "", limit: int = 10):
    return await call_tool(
        "list_airlines", {"search": search, "limit": limit, "offset": 0}
    )


async def current_weather(city: str):
    return await call_tool("get_current_weather", {"city": city})


async def forecast(city: str):
    return await call_tool("get_forecast", {"city": city})
