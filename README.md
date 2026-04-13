# ctse-assignment-2

News Summarizer Multi‑Agent System (MAS):
1) Fetches recent articles for a topic (NewsAPI)
2) Summarizes them (LLM)
3) Writes a markdown digest to `outputs/`

## Prerequisites

- Python 3.10+ (3.11 recommended)
- A NewsAPI key (https://newsapi.org) exported as an environment variable (free tier)
- Ollama installed and running locally (this assignment prohibits paid/cloud LLM APIs)

## Step-by-step: Run on Windows (PowerShell)

### 1) Open a terminal in the project folder

From VS Code: Terminal → New Terminal.

### 2) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Set required environment variables

#### 4.1) NewsAPI key (required)

```powershell
$env:NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"
```

To persist it for future terminals (optional):

```powershell
setx NEWSAPI_KEY "YOUR_NEWSAPI_KEY_HERE"
```

Close/reopen the terminal after using `setx`.

#### 4.2) LLM for CrewAI (required)

This project uses CrewAI agents, configured to run **locally with Ollama only**.

1) Install Ollama and start it.

2) Choose a model and pull it.

Recommended (better for tool/function calling and structured output):
- `llama3.1:8b-instruct`
- `qwen2.5:7b-instruct`

Example:

```powershell
$env:OLLAMA_MODEL = "llama3.1:8b-instruct"
ollama pull $env:OLLAMA_MODEL
```

3) Point OpenAI-compatible clients at Ollama:

```powershell
$env:OPENAI_API_BASE = "http://localhost:11434/v1"
$env:OPENAI_API_KEY = "ollama"
```

4) (Optional) Enable CrewAI tool-calling in local mode:

Some Ollama models reject OpenAI-style `tools` payloads. If you see an error like
"does not support tools", switch models.

```powershell
$env:CREWAI_LOCAL_TOOL_CALLS = "true"
```

Notes:
- Depending on your CrewAI/LiteLLM version, you may need `OPENAI_BASE_URL` instead of `OPENAI_API_BASE`.

### 5) Run the pipeline

Run with a topic argument:

```powershell
python main.py --topic "artificial intelligence"
```

Or run interactively:

```powershell
python main.py
```

### 6) Check outputs

- The markdown digest is written under `outputs/` (e.g. `outputs/digest_YYYY-MM-DD.md`).
- Agent trace logs are appended to `outputs/agent_trace.json`.

## Step-by-step: Run tests

```powershell
pytest -q
```

Notes:
- Some tests include “LLM-as-a-judge” checks via Ollama. If Ollama isn’t running / `llama3` isn’t available, those tests are designed to skip gracefully.

## Troubleshooting

- **`No NewsAPI key found`**: ensure `NEWSAPI_KEY` is set in the same terminal you run `python main.py`.
- **CrewAI fails immediately**: you likely haven’t configured an LLM provider. Set `OPENAI_API_KEY` (OpenAI) or configure Ollama as above.
- **"does not support tools" (Ollama)**: your selected Ollama model rejects tool-calling payloads. Either switch models or set `CREWAI_LOCAL_TOOL_CALLS=true` only if your model supports tools.
- **Ollama connection errors**: verify Ollama is running and `http://localhost:11434` is reachable.