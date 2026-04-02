"""
Week 8 Utility Module
=====================
Shared tools, API client, image helpers.
Extended from Week 7 with vision (base64 image) support.
"""
import os
import base64
import math
import time
from pathlib import Path

import wikipedia
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

# ─── API Client ───────────────────────────────────────────────────────────────

def get_client() -> Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it or add to .env in the project root."
        )
    return Anthropic(api_key=key)


# ─── Tool Implementations ─────────────────────────────────────────────────────

def wikipedia_search(query: str) -> str:
    """Return a concise Wikipedia summary (≤600 chars)."""
    try:
        summary = wikipedia.summary(query, sentences=4, auto_suggest=True)
        return summary[:600]
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation — try one of: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."
    except Exception as e:
        return f"Search error: {e}"


def python_calculator(expression: str) -> str:
    """Safely evaluate a Python math expression."""
    safe_ns: dict = {
        "__builtins__": {},
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "divmod": divmod,
        "sum": sum,
    }
    try:
        result = eval(expression, safe_ns)  # nosec – restricted namespace
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


# ─── Tool Schemas (Claude API format) ─────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia for factual information about people, places, "
            "events, scientific concepts, or historical facts. Returns a concise summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'speed of light', 'Berlin Wall history'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "python_calculator",
        "description": (
            "Evaluate arithmetic and mathematical expressions. "
            "Supports +, -, *, /, **, %, math.sqrt(), math.pi, math.e, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A valid Python math expression, "
                        "e.g. '12 * 7' or 'math.pi * 5**2' or '45 + 32 + 67 + 28'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    """Route a tool call to the correct implementation."""
    if name == "wikipedia_search":
        return wikipedia_search(inputs.get("query", ""))
    elif name == "python_calculator":
        return python_calculator(inputs.get("expression", ""))
    else:
        available = [s["name"] for s in TOOL_SCHEMAS]
        return f"Unknown tool '{name}'. Available: {available}"


# ─── Image Utilities ──────────────────────────────────────────────────────────

def image_to_base64(path: "str | Path") -> str:
    """Read an image file and return its base64 string."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def build_user_message(text: str, image_path: "str | Path | None" = None) -> dict:
    """
    Build an Anthropic API 'user' message dict.

    • If image_path is None  → plain text message.
    • If image_path is given → multimodal message: [image block, text block].

    Token-concatenation architecture (what Claude uses):
      Image → vision encoder → visual tokens
      Text  → tokenizer      → text tokens
      [visual tokens] ++ [text tokens] → single transformer → response

    This differs from latent-fusion (e.g. LLaVA) where the image embedding is
    projected into the LLM's embedding space via a learned linear projector
    before token generation begins.
    """
    if image_path is None:
        return {"role": "user", "content": text}

    img_b64 = image_to_base64(image_path)
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            },
            {"type": "text", "text": text},
        ],
    }
