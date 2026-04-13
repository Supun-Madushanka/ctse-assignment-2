"""
Defines the global state schema passed between all agents in the
News Summarizer Multi-Agent System.

Every agent reads from and writes to this single state dictionary.
No agent communicates directly with another — all data flows through here.
"""

from typing import TypedDict

class Article(TypedDict):
    """Represents a single raw news article fetched by Agent 1."""
    title: str
    content: str
    url: str
    published_at: str
    source: str

class Summary(TypedDict):
    """Represents a cleaned, summarized article produced by Agent 2."""
    title: str
    summary: str
    url: str
    published_at: str
    source: str

class LogEntry(TypedDict):
    """Represents a single structured log entry from the observability logger."""
    timestamp: str
    agent: str
    input_summary: str
    tool_called: str
    tool_output_summary: str
    output_preview: str
    status: str

class NewsState(TypedDict):
    """
    The complete global state of the News Summarizer MAS.

    Fields
    ------
    topic : str
        The news topic entered by the user (e.g. 'AI news today').
    raw_articles : list[Article]
        Raw articles fetched by Agent 1 (News Fetcher).
    summaries : list[Summary]
        Cleaned summaries produced by Agent 2 (Summarizer).
    report_path : str
        File path of the saved digest produced by Agent 3 (Report Writer).
    logs : list[LogEntry]
        Structured log entries appended by the logger after each agent runs.
    """
    topic: str
    raw_articles: list[Article]
    summaries: list[Summary]
    report_path: str
    logs: list[LogEntry]


def create_initial_state(topic: str) -> NewsState:
    """
    Creates and returns a fresh initial state for a new pipeline run.

    Args:
        topic (str): The news topic provided by the user.

    Returns:
        NewsState: A fully initialized state dictionary with empty collections.
    """
    return NewsState(
        topic=topic,
        raw_articles=[],
        summaries=[],
        report_path="",
        logs=[]
    )