"""
Structured JSON observability logger for the News Summarizer MAS.

Records every agent's input, tool call, output, and status to a
persistent JSON log file. This satisfies the LLMOps/AgentOps &
Observability requirement of the assignment.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from state.shared_state import LogEntry, NewsState

LOG_DIR = "outputs"
LOG_FILE = os.path.join(LOG_DIR, "agent_trace.json")

def _get_timestamp() -> str:
    """
    Returns the current UTC time as an ISO 8601 formatted string.

    Returns:
        str: Timestamp string e.g. '2026-04-13T10:23:01Z'.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _truncate(value: Any, max_chars: int = 200) -> str:
    """
    Converts a value to string and truncates it for safe log previews.

    Args:
        value (Any): The value to preview.
        max_chars (int): Maximum number of characters to keep.

    Returns:
        str: Truncated string representation.
    """
    text = str(value)
    return text[:max_chars] + "..." if len(text) > max_chars else text

def log_agent_run(
    state: NewsState,
    agent_name: str,
    input_data: dict[str, Any],
    tool_called: str,
    tool_output: Any,
    output_data: Any,
    status: str = "success"
) -> None:
    """
    Logs a single agent run to the JSON trace file and appends the entry
    to the shared state's logs list.

    Args:
        state (NewsState): The current shared state (mutated in-place to add log).
        agent_name (str): Name of the agent that ran (e.g. 'NewsFetcher').
        input_data (dict[str, Any]): A dict summarizing what the agent received.
        tool_called (str): Name of the tool the agent invoked.
        tool_output (Any): Raw output returned by the tool.
        output_data (Any): The agent's final output passed to the next agent.
        status (str): 'success' or 'error'. Defaults to 'success'.

    Returns:
        None
    """
    entry: LogEntry = {
        "timestamp": _get_timestamp(),
        "agent": agent_name,
        "input_summary": _truncate(input_data),
        "tool_called": tool_called,
        "tool_output_summary": _truncate(tool_output),
        "output_preview": _truncate(output_data),
        "status": status
    }

    state["logs"].append(entry)

    os.makedirs(LOG_DIR, exist_ok=True)

    existing_logs: list[LogEntry] = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                existing_logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_logs = []

    existing_logs.append(entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_logs, f, indent=2, ensure_ascii=False)

    print(f"[LOGGER] {entry['timestamp']} | {agent_name} | {status} | tool: {tool_called}")

def print_trace_summary(state: NewsState) -> None:
    """
    Prints a human-readable summary of all logged agent runs from the
    current state to the console.

    Args:
        state (NewsState): The shared state containing all log entries.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("AGENT TRACE SUMMARY")
    print("=" * 60)
    for i, entry in enumerate(state["logs"], 1):
        print(f"\n[{i}] Agent     : {entry['agent']}")
        print(f"    Timestamp : {entry['timestamp']}")
        print(f"    Tool      : {entry['tool_called']}")
        print(f"    Status    : {entry['status']}")
        print(f"    Input     : {entry['input_summary'][:80]}...")
        print(f"    Output    : {entry['output_preview'][:80]}...")
    print("\n" + "=" * 60)
    print(f"Full trace saved to: {LOG_FILE}")
    print("=" * 60 + "\n")