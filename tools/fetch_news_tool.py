"""
Custom tool for Agent 1 (News Fetcher) in the News Summarizer MAS.

Fetches real news articles from the NewsAPI.org free tier based on
a user-provided topic. Returns a structured list of article dicts
for downstream agents to process.

Individual contribution: Member A
"""

import os
import requests
from datetime import datetime, timedelta, timezone
from state.shared_state import Article

# NewsAPI current REST base (v2). The old v1 endpoints return 404.
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
DEFAULT_MAX_ARTICLES = 10
MIN_CONTENT_LENGTH = 100

def fetch_news_tool(
    topic: str,
    max_articles: int = DEFAULT_MAX_ARTICLES,
    api_key: str | None = None
) -> list[Article]:
    """
    Fetches news articles related to a given topic from the NewsAPI.org API.

    Retrieves articles published in the last 7 days, sorted by relevance.
    Filters out articles with missing or very short content. Returns a
    list of structured Article dicts ready for the summarizer agent.

    Args:
        topic (str): The news topic or keyword to search for
                     (e.g. 'artificial intelligence', 'climate change').
        max_articles (int): Maximum number of articles to return.
                            Must be between 1 and 10. Defaults to 10.
        api_key (str | None): NewsAPI key. If None, reads from the
                              NEWSAPI_KEY environment variable.

    Returns:
        list[Article]: A list of Article dicts, each containing:
                       - title (str): Headline of the article.
                       - content (str): Body text of the article.
                       - url (str): Direct URL to the article.
                       - published_at (str): ISO 8601 publication date.
                       - source (str): Name of the news source.

    Raises:
        ValueError: If topic is empty or max_articles is out of range.
        EnvironmentError: If no API key is found.
        RuntimeError: If the NewsAPI request fails or returns an error.

    Example:
        >>> articles = fetch_news_tool("artificial intelligence", max_articles=5)
        >>> print(articles[0]["title"])
        'OpenAI releases new model...'
    """
    if not topic or not topic.strip():
        raise ValueError("Topic must be a non-empty string.")

    if not (1 <= max_articles <= 10):
        raise ValueError("max_articles must be between 1 and 10.")

    resolved_key: str = api_key or os.getenv("NEWSAPI_KEY", "")
    if not resolved_key:
        raise EnvironmentError(
            "No NewsAPI key found. Set the NEWSAPI_KEY environment variable "
            "or pass api_key directly to fetch_news_tool()."
        )

    from_date: str = (
        datetime.now(timezone.utc) - timedelta(days=7)
    ).strftime("%Y-%m-%d")

    params: dict = {
        "q": topic.strip(),
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": resolved_key
    }

    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("NewsAPI request timed out after 10 seconds.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Failed to connect to NewsAPI. Check your internet connection.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"NewsAPI returned HTTP error: {e}")

    data: dict = response.json()

    if data.get("status") != "ok":
        error_msg: str = data.get("message", "Unknown error from NewsAPI.")
        raise RuntimeError(f"NewsAPI error: {error_msg}")

    raw_articles: list[dict] = data.get("articles", [])

    articles: list[Article] = []
    for raw in raw_articles:
        content: str = raw.get("content") or raw.get("description") or ""
        title: str = raw.get("title") or ""
        url: str = raw.get("url") or ""

        if not title or not url:
            continue

        if len(content.strip()) < MIN_CONTENT_LENGTH:
            continue

        articles.append(Article(
            title=title.strip(),
            content=content.strip(),
            url=url.strip(),
            published_at=raw.get("publishedAt", ""),
            source=raw.get("source", {}).get("name", "Unknown")
        ))

        if len(articles) >= max_articles:
            break

    return articles