"""
Custom tool for Agent 2 (Summarizer) in the News Summarizer MAS.

Filters a list of raw articles before they are passed to the LLM for
summarization. Removes duplicates, articles that are too short, and
articles missing required fields. This ensures the LLM only processes
high-quality input and avoids wasting tokens.

Individual contribution: Member B
"""

from state.shared_state import Article

DEFAULT_MIN_CONTENT_LENGTH = 150
DEFAULT_MIN_TITLE_LENGTH = 10

def filter_articles_tool(
    articles: list[Article],
    min_content_length: int = DEFAULT_MIN_CONTENT_LENGTH,
    min_title_length: int = DEFAULT_MIN_TITLE_LENGTH
) -> list[Article]:
    """
    Filters a list of raw Article dicts to remove low-quality entries
    before LLM summarization.

    Applies the following filters in order:
    1. Removes articles with missing title, content, or URL.
    2. Removes articles whose title is shorter than min_title_length.
    3. Removes articles whose content is shorter than min_content_length.
    4. Removes duplicate articles based on URL.
    5. Removes articles with placeholder/removed content from NewsAPI
       (e.g. content ending with '[+N chars]' only).

    Args:
        articles (list[Article]): The raw list of Article dicts returned
                                  by the fetch_news_tool.
        min_content_length (int): Minimum number of characters required
                                  in the article content. Defaults to 150.
        min_title_length (int): Minimum number of characters required
                                in the article title. Defaults to 10.

    Returns:
        list[Article]: A filtered list of Article dicts ready for
                       LLM summarization. May be empty if all articles
                       fail the filters.

    Raises:
        ValueError: If min_content_length or min_title_length is negative.
        TypeError: If articles is not a list.

    Example:
        >>> raw = fetch_news_tool("climate change", max_articles=10)
        >>> filtered = filter_articles_tool(raw, min_content_length=200)
        >>> print(f"Kept {len(filtered)} of {len(raw)} articles")
        Kept 7 of 10 articles
    """
    if not isinstance(articles, list):
        raise TypeError(f"articles must be a list, got {type(articles).__name__}.")

    if min_content_length < 0:
        raise ValueError("min_content_length must be a non-negative integer.")

    if min_title_length < 0:
        raise ValueError("min_title_length must be a non-negative integer.")

    if not articles:
        return []

    seen_urls: set[str] = set()
    filtered: list[Article] = []

    for article in articles:
        title: str = (article.get("title") or "").strip()
        content: str = (article.get("content") or "").strip()
        url: str = (article.get("url") or "").strip()

        if not title or not content or not url:
            continue

        if len(title) < min_title_length:
            continue

        if len(content) < min_content_length:
            continue

        if url in seen_urls:
            continue

        if _is_placeholder_content(content):
            continue

        seen_urls.add(url)
        filtered.append(article)

    return filtered

def _is_placeholder_content(content: str) -> bool:
    """
    Detects whether a NewsAPI article content is a truncated placeholder.

    NewsAPI free tier truncates article content and appends a string like
    '[+1234 chars]'. If the entire meaningful content is just this
    truncation notice, the article is not useful for summarization.

    Args:
        content (str): The article content string to check.

    Returns:
        bool: True if the content appears to be a placeholder, False otherwise.

    Example:
        >>> _is_placeholder_content("Some intro text [+500 chars]")
        False
        >>> _is_placeholder_content("[+500 chars]")
        True
    """
    import re
    stripped = content.strip()
    pattern = r"^\[\+\d+ chars\]$"
    return bool(re.match(pattern, stripped))