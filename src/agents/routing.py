"""Routing stubs for future multi-tool agent extensions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteDecision:
    """Simple route decision placeholder for future tools."""

    route: str
    use_tools: bool


def route_question(question: str) -> RouteDecision:
    """Route every question to the direct tutor path for now."""

    return RouteDecision(route="direct_tutor", use_tools=False)
