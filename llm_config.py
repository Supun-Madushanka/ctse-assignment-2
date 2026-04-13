"""LLM configuration helpers.

This project is designed to work with local LLMs via Ollama by default.
CrewAI is explicitly configured with a local OpenAI-compatible base URL,
so it won't fall back to a cloud model name like 'gpt-4.1-mini'.
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
    """Return a CrewAI LLM configured for either Ollama (local) or OpenAI.

    Rules:
    - If USE_LOCAL_LLM/USE_OLLAMA is true OR OLLAMA_MODEL is set OR no OPENAI_API_KEY
      is set, default to Ollama at http://localhost:11434/v1.
    - Otherwise, assume the user wants their cloud OpenAI config.
    """

    use_local = (
        _is_truthy(os.getenv("USE_LOCAL_LLM"))
        or _is_truthy(os.getenv("USE_OLLAMA"))
        or bool(os.getenv("OLLAMA_MODEL"))
        or not bool(os.getenv("OPENAI_API_KEY"))
    )

    if use_local:
        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:11434/v1"
        )
        api_key = os.getenv("OPENAI_API_KEY") or "ollama"

        # If a user accidentally points at a non-local base URL while intending
        # local execution, keep it safe by forcing localhost.
        if base_url and not _looks_like_local_url(base_url):
            base_url = "http://localhost:11434/v1"

        return LLM(model=model, base_url=base_url, api_key=api_key, provider="openai")

    # Cloud/OpenAI path: rely on the user's environment/provider config.
    # We intentionally do not guess a model here.
    model = os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
    return LLM(model=model, api_key=os.getenv("OPENAI_API_KEY"), provider="openai")


def is_using_local_llm() -> bool:
        """Return True when this project is configured to use Ollama/local LLMs."""
        return (
                _is_truthy(os.getenv("USE_LOCAL_LLM"))
                or _is_truthy(os.getenv("USE_OLLAMA"))
                or bool(os.getenv("OLLAMA_MODEL"))
                or not bool(os.getenv("OPENAI_API_KEY"))
        )


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
