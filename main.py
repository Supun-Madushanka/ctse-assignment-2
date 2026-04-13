"""
Entry point for the News Summarizer Multi-Agent System.

Orchestrates the full pipeline:
  User topic → Agent 1 (Fetch) → Agent 2 (Summarize) → Agent 3 (Write)

Manages the shared state across all agents and prints a final trace
summary at the end of every successful run.

Usage:
    python main.py
    python main.py --topic "climate change"
"""

import argparse
import sys

from dotenv import load_dotenv

from state.shared_state import create_initial_state, NewsState
from agents.fetcher_agent import run_fetcher_agent
from agents.summarizer_agent import run_summarizer_agent
from agents.writer_agent import run_writer_agent
from logger.agent_logger import print_trace_summary


def _load_environment() -> None:
    """Loads environment variables from a local .env file if present."""
    load_dotenv(override=False)

def run_pipeline(topic: str) -> NewsState:
    """
    Runs the full 3-agent news summarization pipeline for a given topic.

    Initializes shared state, runs each agent in sequence, and returns
    the final state containing all articles, summaries, report path,
    and logs.

    Args:
        topic (str): The news topic to search for and summarize.

    Returns:
        NewsState: The final shared state after all agents have run.

    Raises:
        RuntimeError: If any agent in the pipeline fails critically.
    """
    print("\n" + "=" * 60)
    print("NEWS SUMMARIZER MULTI-AGENT SYSTEM")
    print("=" * 60)
    print(f"Topic: {topic}")
    print("=" * 60 + "\n")

    state: NewsState = create_initial_state(topic=topic)

    print("[Pipeline] Starting Agent 1: News Fetcher...")
    state = run_fetcher_agent(state)
    print(f"[Pipeline] Agent 1 complete. {len(state['raw_articles'])} articles fetched.\n")

    print("[Pipeline] Starting Agent 2: Summarizer...")
    state = run_summarizer_agent(state)
    print(f"[Pipeline] Agent 2 complete. {len(state['summaries'])} summaries produced.\n")

    print("[Pipeline] Starting Agent 3: Report Writer...")
    state = run_writer_agent(state)
    print(f"[Pipeline] Agent 3 complete. Report saved to: {state['report_path']}\n")

    print_trace_summary(state)

    return state

def main() -> None:
    """
    Parses command-line arguments and runs the news summarization pipeline.
    Falls back to an interactive prompt if no topic is provided via CLI.
    """
    parser = argparse.ArgumentParser(
        description="News Summarizer Multi-Agent System"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="The news topic to search for (e.g. 'artificial intelligence')"
    )
    args = parser.parse_args()

    _load_environment()

    if args.topic:
        topic: str = args.topic.strip()
    else:
        print("News Summarizer MAS")
        print("-" * 30)
        topic = input("Enter a news topic: ").strip()

    if not topic:
        print("Error: Topic cannot be empty.")
        sys.exit(1)

    try:
        final_state = run_pipeline(topic=topic)
        print(f"\nDone! Your digest is at: {final_state['report_path']}")
        sys.exit(0)
    except RuntimeError as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()