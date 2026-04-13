"""
test_fetcher.py
---------------
Test suite for Agent 1 (News Fetcher) and fetch_news_tool.

Tests cover: happy path, edge cases, error handling, and output
schema validation. Each test is independent and uses mocking to
avoid real API calls during automated testing.

Individual contribution: Member A
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from tools.fetch_news_tool import fetch_news_tool
from state.shared_state import create_initial_state


MOCK_API_RESPONSE = {
    "status": "ok",
    "totalResults": 3,
    "articles": [
        {
            "title": "AI makes breakthrough in medical diagnosis",
            "content": "Researchers at MIT have developed an AI system that can diagnose rare diseases with 95% accuracy, surpassing human specialists in blind trials conducted this week.",
            "url": "https://example.com/ai-medical",
            "publishedAt": "2026-04-13T08:00:00Z",
            "source": {"name": "TechCrunch"}
        },
        {
            "title": "New AI regulation bill passes in EU parliament",
            "content": "The European Union has passed a landmark AI regulation bill requiring all AI systems above a certain risk threshold to undergo mandatory safety audits before deployment.",
            "url": "https://example.com/eu-ai-law",
            "publishedAt": "2026-04-12T14:00:00Z",
            "source": {"name": "Reuters"}
        },
        {
            "title": "OpenAI announces new research direction",
            "content": "OpenAI has announced a significant shift in its research priorities, focusing on interpretability and alignment rather than raw capability scaling for the remainder of 2026.",
            "url": "https://example.com/openai-research",
            "publishedAt": "2026-04-11T10:00:00Z",
            "source": {"name": "The Verge"}
        }
    ]
}


class TestFetchNewsTool:

    def test_returns_list_of_articles(self):
        """Happy path: tool returns a non-empty list of Article dicts."""
        with patch("tools.fetch_news_tool.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = MOCK_API_RESPONSE
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = fetch_news_tool(topic="artificial intelligence", api_key="testkey")

            assert isinstance(result, list), "Result must be a list"
            assert len(result) > 0, "Result must not be empty"

    def test_each_article_has_required_fields(self):
        """Each returned article must have all required schema fields."""
        with patch("tools.fetch_news_tool.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = MOCK_API_RESPONSE
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = fetch_news_tool(topic="AI", api_key="testkey")

            required_fields = {"title", "content", "url", "published_at", "source"}
            for article in result:
                missing = required_fields - set(article.keys())
                assert not missing, f"Article missing fields: {missing}"

    def test_respects_max_articles_limit(self):
        """Tool must not return more articles than max_articles."""
        with patch("tools.fetch_news_tool.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = MOCK_API_RESPONSE
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = fetch_news_tool(topic="AI", max_articles=2, api_key="testkey")

            assert len(result) <= 2, "Must not exceed max_articles limit"

    def test_raises_on_empty_topic(self):
        """Tool must raise ValueError when topic is empty."""
        with pytest.raises(ValueError, match="non-empty"):
            fetch_news_tool(topic="", api_key="testkey")

    def test_raises_on_whitespace_topic(self):
        """Tool must raise ValueError when topic is only whitespace."""
        with pytest.raises(ValueError, match="non-empty"):
            fetch_news_tool(topic="   ", api_key="testkey")

    def test_raises_on_invalid_max_articles(self):
        """Tool must raise ValueError for max_articles out of range."""
        with pytest.raises(ValueError, match="between 1 and 10"):
            fetch_news_tool(topic="AI", max_articles=0, api_key="testkey")

        with pytest.raises(ValueError, match="between 1 and 10"):
            fetch_news_tool(topic="AI", max_articles=11, api_key="testkey")

    def test_raises_on_missing_api_key(self):
        """Tool must raise EnvironmentError when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="NEWSAPI_KEY"):
                fetch_news_tool(topic="AI")

    def test_raises_on_api_error_status(self):
        """Tool must raise RuntimeError when NewsAPI returns status != ok."""
        with patch("tools.fetch_news_tool.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "error",
                "message": "Invalid API key."
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError, match="NewsAPI error"):
                fetch_news_tool(topic="AI", api_key="badkey")

    def test_filters_articles_with_short_content(self):
        """Articles with very short content must be excluded from results."""
        response = {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "title": "Short article",
                    "content": "Too short.",
                    "url": "https://example.com/short",
                    "publishedAt": "2026-04-13T08:00:00Z",
                    "source": {"name": "Test"}
                }
            ]
        }
        with patch("tools.fetch_news_tool.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = fetch_news_tool(topic="AI", api_key="testkey")
            assert result == [], "Short-content articles must be filtered out"

    def test_handles_network_timeout_gracefully(self):
        """Tool must raise RuntimeError on network timeout."""
        import requests as req
        with patch("tools.fetch_news_tool.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.Timeout()

            with pytest.raises(RuntimeError, match="timed out"):
                fetch_news_tool(topic="AI", api_key="testkey")

class TestFetcherAgentState:

    def test_state_raw_articles_is_empty_initially(self):
        """Initial state must have an empty raw_articles list."""
        state = create_initial_state("AI news")
        assert state["raw_articles"] == []

    def test_state_topic_is_preserved(self):
        """Initial state must preserve the topic string exactly."""
        state = create_initial_state("climate change")
        assert state["topic"] == "climate change"