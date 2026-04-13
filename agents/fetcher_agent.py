"""
Agent 1 — News Fetcher for the News Summarizer MAS.

Responsible for understanding the user's topic, deciding on the best
search query, invoking the fetch_news_tool, and writing raw articles
into the shared state.

Individual contribution: Member A
"""

from crewai import Agent, Task, Crew
from crewai.tools import tool
from state.shared_state import NewsState, Article
from tools.fetch_news_tool import fetch_news_tool
from logger.agent_logger import log_agent_run
from llm_config import get_crewai_llm, crewai_tool_calls_enabled


FETCHER_SYSTEM_PROMPT = """
You are a professional news research assistant specializing in finding
relevant, high-quality news articles on any given topic.

Your responsibilities:
- Receive a news topic from the user
- Use the fetch_news_tool to retrieve up to 10 recent articles
- Return the articles exactly as fetched — do NOT summarize, modify,
  or add any information not present in the raw API response
- If the tool returns fewer than 3 articles, report this clearly
- Never fabricate article titles, URLs, or content

Constraints:
- Only use the fetch_news_tool — do not rely on your internal knowledge
  for news content
- Do not filter or discard articles — that is the Summarizer's job
- Always confirm how many articles were successfully fetched
"""


@tool("fetch_news_tool")
def crewai_fetch_news_tool(topic: str) -> str:
    """
    Fetches up to 10 recent news articles on the given topic from NewsAPI.
    Returns a JSON string of article objects containing title, content,
    url, published_at, and source.

    Args:
        topic: The news topic or keyword to search for.

    Returns:
        A JSON string representing a list of article dicts.
    """
    import json
    articles = fetch_news_tool(topic=topic, max_articles=10)
    return json.dumps(articles, ensure_ascii=False)

def build_fetcher_agent() -> Agent:
    """
    Builds and returns the configured CrewAI News Fetcher agent.

    Returns:
        Agent: A CrewAI Agent instance with the Fetcher persona,
               system prompt, and fetch_news_tool attached.
    """
    return Agent(
        role="News Fetcher",
        goal=(
            "Fetch up to 10 recent, relevant news articles for the given "
            "topic using the NewsAPI tool and return them as structured data."
        ),
        backstory=FETCHER_SYSTEM_PROMPT.strip(),
        llm=get_crewai_llm(),
        tools=[crewai_fetch_news_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

def build_fetcher_task(agent: Agent, topic: str) -> Task:
    """
    Builds the CrewAI Task for the News Fetcher agent.

    Args:
        agent (Agent): The Fetcher agent instance.
        topic (str): The user's news topic string.

    Returns:
        Task: A configured CrewAI Task for the Fetcher agent.
    """
    return Task(
        description=(
            f"Search for and fetch up to 10 recent news articles about: '{topic}'. "
            f"Use the fetch_news_tool with this exact topic. "
            f"Return the complete list of article objects as JSON."
        ),
        expected_output=(
            "A JSON array of article objects. Each object must contain: "
            "title, content, url, published_at, source. "
            "Report the total number of articles fetched."
        ),
        agent=agent
    )

def run_fetcher_agent(state: NewsState) -> NewsState:
    """
    Runs the News Fetcher agent, populates state['raw_articles'],
    and logs the run.

    Args:
        state (NewsState): The current shared state. Must have 'topic' set.

    Returns:
        NewsState: Updated state with 'raw_articles' populated.

    Raises:
        RuntimeError: If the agent fails to fetch any articles.
    """
    import json

    topic: str = state["topic"]

    agent = build_fetcher_agent()
    task = build_fetcher_task(agent, topic)

    result = None
    if crewai_tool_calls_enabled():
        try:
            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
        except Exception:
            result = None

    try:
        raw_text: str = str(result)
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        json_str = raw_text[start:end]
        articles: list[Article] = json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        articles = fetch_news_tool(topic=topic, max_articles=10)

    state["raw_articles"] = articles

    log_agent_run(
        state=state,
        agent_name="NewsFetcher",
        input_data={"topic": topic},
        tool_called="fetch_news_tool",
        tool_output=f"{len(articles)} articles fetched",
        output_data=[a["title"] for a in articles],
        status="success" if articles else "error"
    )

    if not articles:
        raise RuntimeError(
            f"News Fetcher returned 0 articles for topic: '{topic}'. "
            "Try a different topic or check your NewsAPI key."
        )

    print(f"[Fetcher] Fetched {len(articles)} articles for topic: '{topic}'")
    return state