"""
utils.py — Shared utilities for Week 7: Core Agent Mechanics

Provides:
  - Tool implementations: wikipedia_search, python_calculator
  - JSON schemas for Claude's native tool_use API
  - dispatch_tool() router
  - get_client() factory

All agent experiments import from here so the tools stay consistent.
"""

import os
import re
import math
import wikipedia
import anthropic
from dotenv import load_dotenv

# Load .env from the project root (agent_project/.env) if present
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Model config ──────────────────────────────────────────────────────────────
# Use haiku for speed + cost efficiency across all experiments.
MODEL = "claude-haiku-4-5-20251001"


# ── Claude client ─────────────────────────────────────────────────────────────

def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client. Raises EnvironmentError if key not set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Export it before running:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


# ── Tool implementations ──────────────────────────────────────────────────────

def wikipedia_search(query: str) -> str:
    """
    Return the first 600 characters of the Wikipedia summary for `query`.

    Design notes:
    - auto_suggest=True lets Wikipedia handle minor typos.
    - DisambiguationError means multiple pages matched; we surface the options
      so the agent can refine the query.
    - Returns a string in all cases (never raises) so the agent loop stays clean.
    """
    try:
        page = wikipedia.page(query, auto_suggest=True)
        summary = page.summary[:600]
        return f"{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        options = ", ".join(e.options[:5])
        return f"Ambiguous query — did you mean one of: {options}?"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'. Try a different query."
    except Exception as exc:
        return f"Search error: {exc}"


def python_calculator(expression: str) -> str:
    """
    Safely evaluate a math expression and return the result as a string.

    Security approach: restrict __builtins__ to {} and only expose math
    functions by name. This prevents code injection while allowing full
    arithmetic and common math functions.

    Supported: +,-,*,/,**,%, math.sqrt, math.log, math.sin, math.cos,
               math.pi, math.e, abs, round, min, max, int, float, pow, divmod
    """
    # Safe namespace: no builtins, only whitelisted math names
    safe_ns = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    safe_ns.update({
        "abs": abs, "round": round, "min": min, "max": max,
        "int": int, "float": float, "pow": pow, "divmod": divmod,
    })
    try:
        result = eval(expression, {"__builtins__": {}}, safe_ns)  # noqa: S307
        # Format nicely: int if no fractional part, else float
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Calculator error: {exc}"


# ── JSON schemas for Claude's native tool_use API ─────────────────────────────
#
# These are passed as the `tools` parameter to client.messages.create().
# Claude uses them to decide when and how to call each tool.

TOOL_SCHEMAS = [
    {
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia and return a concise summary of the most relevant "
            "article. Use this to look up facts, dates, definitions, and "
            "encyclopaedic knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific to avoid disambiguation. "
                        "Example: 'Mount Everest height' not just 'mountain'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "python_calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Use this for all arithmetic: addition, multiplication, exponents, "
            "square roots, logarithms, trigonometry, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A valid Python math expression. Examples: '2**10', "
                        "'sqrt(144)', 'pi * 7**2', 'log(100, 10)'. "
                        "Do NOT include variable assignments or print()."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    """
    Execute a tool by name with the given inputs dict.

    Returns a string result in all cases. Unknown tool names return an error
    string (not an exception) so the agent loop handles them gracefully.
    """
    if name == "wikipedia_search":
        return wikipedia_search(inputs.get("query", ""))
    elif name == "python_calculator":
        return python_calculator(inputs.get("expression", ""))
    else:
        available = [s["name"] for s in TOOL_SCHEMAS]
        return (
            f"Unknown tool '{name}'. "
            f"Available tools: {available}. "
            "Please use one of the defined tools."
        )
