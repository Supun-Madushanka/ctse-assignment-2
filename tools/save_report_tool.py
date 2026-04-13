"""
Custom tool for Agent 3 (Report Writer) in the News Summarizer MAS.

Saves the final formatted news digest produced by the LLM to a local
markdown file in the outputs/ directory. Returns the full file path
on success so the shared state can record it.

Individual contribution: Member C
"""

import os
from datetime import datetime, timezone

OUTPUT_DIR = "outputs"

def save_report_tool(
    content: str,
    filename: str | None = None
) -> str:
    """
    Saves a formatted news digest string to a local markdown (.md) file.

    Creates the outputs/ directory if it does not already exist. If no
    filename is provided, generates one automatically using the current
    UTC date (e.g. 'digest_2026-04-13.md'). Validates content before
    writing to avoid creating empty or malformed files.

    Args:
        content (str): The complete formatted digest text to save.
                       Must be a non-empty string.
        filename (str | None): The filename to use for the saved file.
                               Must end with '.md' if provided.
                               If None, a timestamped filename is generated.

    Returns:
        str: The full relative file path of the saved file
             (e.g. 'outputs/digest_2026-04-13.md').

    Raises:
        ValueError: If content is empty or filename has an invalid extension.
        IOError: If the file cannot be written due to a filesystem error.

    Example:
        >>> path = save_report_tool("# AI News Digest\\n\\n1. Article one...")
        >>> print(path)
        outputs/digest_2026-04-13.md
    """
    if not content or not content.strip():
        raise ValueError(
            "Content must be a non-empty string. Cannot save an empty report."
        )

    if filename is not None:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string when provided.")
        if not filename.endswith(".md"):
            raise ValueError(
                f"filename must end with '.md', got: '{filename}'"
            )
        clean_filename: str = filename.strip()
    else:
        date_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        clean_filename = f"digest_{date_str}.md"

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        raise IOError(
            f"Failed to create output directory '{OUTPUT_DIR}': {e}"
        )

    filepath: str = os.path.join(OUTPUT_DIR, clean_filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
            f.write("\n")
    except OSError as e:
        raise IOError(f"Failed to write report to '{filepath}': {e}")

    return filepath

def load_report(filepath: str) -> str:
    """
    Reads and returns the content of a previously saved digest file.

    Useful for testing and verification that the saved content matches
    what was passed to save_report_tool.

    Args:
        filepath (str): The full path to the saved .md file.

    Returns:
        str: The full text content of the file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        IOError: If the file cannot be read.

    Example:
        >>> content = load_report("outputs/digest_2026-04-13.md")
        >>> print(content[:50])
        # AI News Digest — April 13, 2026
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No report file found at: '{filepath}'"
        )

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise IOError(f"Failed to read report from '{filepath}': {e}")