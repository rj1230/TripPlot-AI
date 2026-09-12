from __future__ import annotations

import importlib
import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI(
    title="TripPlot-AI Backend",
    description="Backend health and component verification API",
    version="1.0.0",
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def check_import(module_name: str) -> dict[str, Any]:
    start = time.perf_counter()

    try:
        importlib.import_module(module_name)

        return {
            "status": "ok",
            "module": module_name,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    except Exception as exc:
        return {
            "status": "error",
            "module": module_name,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }


def overall_status(checks: dict[str, Any]) -> str:
    for value in checks.values():
        if isinstance(value, dict) and value.get("status") == "error":
            return "degraded"

    return "healthy"


# ---------------------------------------------------------
# Basic routes
# ---------------------------------------------------------


@app.get("/")
async def root():
    return {
        "service": "TripPlot-AI Backend",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "tripplot-ai-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------
# Component health
# ---------------------------------------------------------


@app.get("/health/components")
async def component_health():

    checks = {
        "state": check_import("state"),
        "agents": check_import("agents"),
        "critic": check_import("critic"),
        "graph": check_import("graph"),
        "mcp_client": check_import("mcp_client"),
    }

    status = overall_status(checks)

    return JSONResponse(
        status_code=200 if status == "healthy" else 503,
        content={
            "status": status,
            "service": "tripplot-ai-backend",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": checks,
        },
    )


# ---------------------------------------------------------
# Graph compilation test
# ---------------------------------------------------------


@app.get("/health/graph")
async def graph_health():

    start = time.perf_counter()

    try:
        from graph import compile_graph

        graph = compile_graph()

        return {
            "status": "ok",
            "graph_compiled": True,
            "graph_type": type(graph).__name__,
            "latency_ms": round(
                (time.perf_counter() - start) * 1000,
                2,
            ),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "graph_compiled": False,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round(
                    (time.perf_counter() - start) * 1000,
                    2,
                ),
            },
        )


# ---------------------------------------------------------
# Configuration check
# ---------------------------------------------------------


@app.get("/health/config")
async def config_health():

    keys = {
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
        "PORTKEY_API_KEY": bool(os.getenv("PORTKEY_API_KEY")),
    }

    configured = sum(keys.values())
    total = len(keys)

    return {
        "status": "ok" if configured == total else "warning",
        "configured": f"{configured}/{total}",
        "environment": keys,
    }


# ---------------------------------------------------------
# MCP import test
# ---------------------------------------------------------


@app.get("/health/mcp")
async def mcp_health():

    start = time.perf_counter()

    try:
        from mcp_client import (
            current_weather,
            forecast,
            list_airlines,
            list_airports,
            tavily_search,
        )

        tools = {
            "tavily_search": callable(tavily_search),
            "list_airports": callable(list_airports),
            "list_airlines": callable(list_airlines),
            "current_weather": callable(current_weather),
            "forecast": callable(forecast),
        }

        failed = [name for name, available in tools.items() if not available]

        if failed:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "tools": tools,
                    "failed_tools": failed,
                },
            )

        return {
            "status": "ok",
            "tools": tools,
            "latency_ms": round(
                (time.perf_counter() - start) * 1000,
                2,
            ),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ---------------------------------------------------------
# Critic test
# ---------------------------------------------------------


@app.get("/health/critic")
async def critic_health():

    try:
        from critic import CriticVerdict

        verdict = CriticVerdict(
            passed=True,
            decision="PASS",
            violations=[],
            suggestions=[],
            responsible_agents=[],
            quality_score=1.0,
            confidence=1.0,
            deterministic_checks_passed=True,
            llm_check_passed=True,
            degraded_mode=False,
            reasoning="Backend critic contract test passed.",
        )

        return {
            "status": "ok",
            "critic_contract": True,
            "sample_verdict": verdict.to_dict(),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "critic_contract": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ---------------------------------------------------------
# Full backend diagnostic
# ---------------------------------------------------------


@app.get("/health/full")
async def full_health():

    components = {
        "state": check_import("state"),
        "agents": check_import("agents"),
        "critic": check_import("critic"),
        "graph": check_import("graph"),
        "mcp_client": check_import("mcp_client"),
    }

    checks = {
        "components": components,
    }

    status = overall_status(components)

    return JSONResponse(
        status_code=200 if status == "healthy" else 503,
        content={
            "status": status,
            "service": "TripPlot-AI",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "next_step": (
                "Backend component checks passed."
                if status == "healthy"
                else "One or more backend components require attention."
            ),
        },
    )


# ---------------------------------------------------------
# Local development
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
