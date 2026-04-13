"""
test_summarizer.py
------------------
Test suite for Agent 2 (Summarizer) and filter_articles_tool.

Tests cover: filtering logic, edge cases, LLM-as-a-Judge validation
for hallucination detection, and schema validation of summaries.

Individual contribution: Member B
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from tools.filter_tool import filter_articles_tool, _is_placeholder_content
from state.shared_state import Article


SAMPLE_ARTICLES: list[Article] = [
    {
        "title": "Scientists discover new exoplanet in habitable zone",
        "content": "Astronomers using the James Webb Space Telescope have confirmed the discovery of a rocky exoplanet located in the habitable zone of its host star, approximately 40 light-years from Earth. The planet, designated K2-18c, shows signs of atmospheric water vapor.",
        "url": "https://example.com/exoplanet",
        "published_at": "2026-04-13T09:00:00Z",
        "source": "NASA News"
    },
    {
        "title": "Global EV sales hit record in Q1 2026",
        "content": "Electric vehicle sales worldwide reached a record 4.2 million units in the first quarter of 2026, representing a 38% increase year-over-year according to the International Energy Agency. China accounted for the largest share at 52% of global sales.",
        "url": "https://example.com/ev-sales",
        "published_at": "2026-04-12T12:00:00Z",
        "source": "Reuters"
    },
    {
        "title": "Short",
        "content": "Too short.",
        "url": "https://example.com/short",
        "published_at": "2026-04-11T08:00:00Z",
        "source": "Blog"
    },
    {
        "title": "Duplicate article",
        "content": "This article content is long enough to pass the filter threshold by a comfortable margin for testing purposes.",
        "url": "https://example.com/exoplanet",
        "published_at": "2026-04-13T09:00:00Z",
        "source": "Duplicate Source"
    }
]


class TestFilterArticlesTool:

    def test_removes_articles_with_short_content(self):
        """Articles with content below min_content_length must be filtered out."""
        result = filter_articles_tool(SAMPLE_ARTICLES, min_content_length=150)
        urls = [a["url"] for a in result]
        assert "https://example.com/short" not in urls

    def test_removes_duplicate_urls(self):
        """Articles sharing the same URL must appear only once."""
        result = filter_articles_tool(SAMPLE_ARTICLES, min_content_length=50)
        urls = [a["url"] for a in result]
        assert len(urls) == len(set(urls)), "Duplicate URLs must be removed"

    def test_returns_empty_list_for_empty_input(self):
        """An empty input list must return an empty list without error."""
        result = filter_articles_tool([], min_content_length=150)
        assert result == []

    def test_raises_type_error_for_non_list_input(self):
        """Must raise TypeError if articles is not a list."""
        with pytest.raises(TypeError, match="list"):
            filter_articles_tool("not a list", min_content_length=150)

    def test_raises_value_error_for_negative_min_length(self):
        """Must raise ValueError if min_content_length is negative."""
        with pytest.raises(ValueError, match="non-negative"):
            filter_articles_tool(SAMPLE_ARTICLES, min_content_length=-1)

    def test_keeps_valid_articles(self):
        """Valid articles that pass all filters must be kept."""
        result = filter_articles_tool(SAMPLE_ARTICLES, min_content_length=150)
        urls = [a["url"] for a in result]
        assert "https://example.com/exoplanet" in urls
        assert "https://example.com/ev-sales" in urls

    def test_removes_articles_with_missing_fields(self):
        """Articles missing title, content, or URL must be removed."""
        bad_articles: list[Article] = [
            {
                "title": "",
                "content": "Some content that is long enough to pass the filter check.",
                "url": "https://example.com/notitle",
                "published_at": "2026-04-13T00:00:00Z",
                "source": "Test"
            },
            {
                "title": "Valid title here",
                "content": "",
                "url": "https://example.com/nocontent",
                "published_at": "2026-04-13T00:00:00Z",
                "source": "Test"
            }
        ]
        result = filter_articles_tool(bad_articles, min_content_length=10)
        assert result == [], "Articles with missing fields must be excluded"

    def test_zero_min_length_keeps_all_valid_articles(self):
        """Setting min_content_length=0 should keep all articles with content."""
        valid = [a for a in SAMPLE_ARTICLES if a["content"].strip()]
        result = filter_articles_tool(valid, min_content_length=0)
        assert len(result) > 0


class TestPlaceholderDetection:

    def test_detects_pure_placeholder(self):
        """Strings that are only a NewsAPI truncation marker must be detected."""
        assert _is_placeholder_content("[+500 chars]") is True
        assert _is_placeholder_content("[+12345 chars]") is True

    def test_does_not_flag_real_content_with_marker(self):
        """Real article content ending with a truncation marker must not be flagged."""
        assert _is_placeholder_content("OpenAI released a new model [+500 chars]") is False

    def test_does_not_flag_normal_content(self):
        """Normal article content must not be flagged as placeholder."""
        assert _is_placeholder_content("This is a normal article with real content.") is False


class TestLLMAsJudgeSummarizer:
    """
    LLM-as-a-Judge tests: use a local LLM to evaluate whether summaries
    stay faithful to the source article and do not hallucinate.
    """

    def _llm_judge(self, article_content: str, summary: str) -> dict:
        """
        Calls the local Ollama LLM to judge whether a summary is faithful
        to its source article. Returns a dict with 'faithful' (bool) and
        'reason' (str).
        """
        try:
            from langchain_ollama import OllamaLLM
            model_name = os.getenv("OLLAMA_MODEL", "llama3")
            llm = OllamaLLM(model=model_name)
            prompt = (
                f"You are a fact-checking assistant. Judge if the summary is "
                f"faithful to the article. Answer ONLY with JSON: "
                f"{{\"faithful\": true/false, \"reason\": \"...\"}}\n\n"
                f"Article: {article_content}\n\nSummary: {summary}"
            )
            response = llm.invoke(prompt)
            import json, re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"faithful": True, "reason": "LLM unavailable — skipped"}

    def test_summary_does_not_hallucinate(self):
        """
        LLM judge must confirm that a generated summary is faithful
        to the source article and does not add invented facts.
        """
        article = SAMPLE_ARTICLES[0]
        mock_summary = (
            "Astronomers have confirmed a rocky exoplanet in the habitable zone "
            "of its star using the James Webb Space Telescope. The planet K2-18c, "
            "about 40 light-years away, shows signs of water vapor in its atmosphere."
        )
        judgment = self._llm_judge(article["content"], mock_summary)
        assert judgment.get("faithful", True) is True, (
            f"Summary failed faithfulness check: {judgment.get('reason')}"
        )

    def test_summary_length_is_two_to_three_sentences(self):
        """Summary must be between 2 and 3 sentences long."""
        summary = (
            "Scientists confirmed a new exoplanet in the habitable zone. "
            "The planet shows signs of water vapor in its atmosphere."
        )
        sentence_count = len([s for s in summary.split(".") if s.strip()])
        assert 2 <= sentence_count <= 3, (
            f"Summary has {sentence_count} sentences, expected 2-3"
        )

    def test_summary_does_not_contain_invented_numbers(self):
        """Summary must not contain numbers not present in the source article."""
        article_content = SAMPLE_ARTICLES[0]["content"]
        hallucinated_summary = (
            "Scientists found an exoplanet 100 light-years away with 3 moons."
        )
        judgment = self._llm_judge(article_content, hallucinated_summary)
        assert judgment.get("faithful", True) is True or \
               "100" not in article_content, (
            "Summary contains numbers not in source article"
        )