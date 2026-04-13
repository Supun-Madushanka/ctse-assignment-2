"""LLM configuration for CrewAI.
CrewAI is configured to talk to Ollama via its OpenAI-compatible endpoint
(default: http://localhost:11434/v1).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from crewai import LLM


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_local_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1"}


def get_crewai_llm() -> LLM:
    """Return a CrewAI LLM configured for local Ollama.

    This repository intentionally does NOT support cloud OpenAI usage.

    Environment variables:
    - OLLAMA_MODEL: model name installed in Ollama (default: llama3)
    - OLLAMA_BASE_URL / OPENAI_API_BASE / OPENAI_BASE_URL: OpenAI-compatible base URL
      (default: http://localhost:11434/v1)
    - OPENAI_API_KEY: optional; use "ollama" for local setups

    Raises:
        RuntimeError: if the base URL is not local (localhost/127.0.0.1).
    """

    model = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:11434/v1"
    )
    api_key = os.getenv("OPENAI_API_KEY") or "ollama"

    if base_url and not _looks_like_local_url(base_url):
        raise RuntimeError(
            "This project is configured for local Ollama only. "
            f"Refusing non-local base URL: {base_url}. "
            "Set OLLAMA_BASE_URL/OPENAI_API_BASE to http://localhost:11434/v1."
        )

    return LLM(model=model, base_url=base_url, api_key=api_key, provider="openai")


def is_using_local_llm() -> bool:
    """Return True (this project is Ollama-only by design)."""
    return True


def crewai_tool_calls_enabled() -> bool:
        """Whether we should attempt CrewAI tool-calling executions.

        Why this exists:
        - Some local Ollama models (e.g. gemma3:1b) reject requests that include
            the OpenAI-style `tools` payload, causing noisy 400 errors.
        - This MAS already includes deterministic fallbacks that call Python tools
            directly, so we can skip the CrewAI kickoff in those cases.

        Behavior:
        - Cloud/OpenAI usage: enabled by default.
        - Local/Ollama usage: disabled by default; enable explicitly with
            `CREWAI_LOCAL_TOOL_CALLS=true`.
        - Always disable with `DISABLE_CREWAI=true`.
        - Always enable with `FORCE_CREWAI=true`.
        """

        if _is_truthy(os.getenv("DISABLE_CREWAI")):
                return False

        if _is_truthy(os.getenv("FORCE_CREWAI")):
                return True

        if is_using_local_llm():
                return _is_truthy(os.getenv("CREWAI_LOCAL_TOOL_CALLS"))

        return True
