# Azure AI Agents - API Samples for Deep Research and Bing Grounding

End-to-end samples showing how to:

1. Create and configure Azure AI Agents (Deep Research + Web Search/Bing Grounding)
2. Expose them through a lightweight FastAPI service (REST + streaming)
3. Experiment and iterate using Jupyter notebooks for agent creation, testing, and evaluation

> These samples are intended as a learning and starter kit and are **not production hardened**.

---

## Repository Structure

```
├── api/                         # FastAPI service exposing agent endpoints
│   ├── main.py                  # Deep Research + streaming agent endpoints (thread, run, status, download)
│   ├── requirements.txt         # Python dependencies for the API layer
│   └── Dockerfile               # (Optional) container build for the API
├── notebooks/                   # Interactive setup & experimentation notebooks
│   ├── 01A_DeepResearch_Agent_Creation.ipynb   # Create/configure a Deep Research agent
│   ├── 01B_WebSearch_Agent_Creation.ipynb      # Create/configure a Web Search (Bing grounded) agent
│   ├── 02A_DeepResearch_Testing.ipynb          # Test Deep Research runs & output formatting
│   └── 02B_WebSearch_Agent_testing.ipynb       # Test streaming + citation handling
├── LICENSE
└── README.md                  # (This file)
```

### Components
| Component | Purpose |
|-----------|---------|
| `api/main.py` | REST + streaming interface for Azure AI Agent Threads & Runs (Deep Research + Web Search agent usage). |
| Notebooks 01A / 01B | Guided creation of agents inside an Azure AI Foundry Project (model selection, Bing connection, capabilities). |
| Notebooks 02A / 02B | Run, validate, and parse responses (citations, references, report generation). |
| `api/requirements.txt` | Runtime dependencies (FastAPI, Azure AI SDK, PDF generation libs, etc.). |
| `Dockerfile` | Containerizing the API for deployment. |

---

## Features
* Create a thread, submit Deep Research queries, poll run status
* Convert final agent response into Markdown or PDF (inline numeric citations → references section)
* Streaming endpoint for a Web Search (Bing grounded) agent with incremental deltas & embedded citation payload
* Automatic citation formatting & optional base64 image streaming for code interpreter outputs (if enabled in agent)

---

## Prerequisites
You will need:

1. An Azure subscription with access to Azure AI Foundry
2. An Azure AI Foundry Project
3. One or more deployed GPT model(s) (e.g., `gpt-4o`, `gpt-4o-mini`, or other supported latest model) in the project and the `o3-deep-research` model
4. A Bing Search (Grounding) connection added to the project (for web search agent)
5. Two Azure AI Agents (recommended naming):
	 * Deep Research Agent (uses reasoning / deep research capability)
	 * Web Search Agent (uses Bing grounding for near-real-time citations)
6. Local Python 3.11+ (matches compiled wheel expectations for some libs)
7. Azure CLI (optional but convenient) OR a Service Principal for non-interactive auth

> NOTE: The existing `api/README.md` references video generation (Sora) and extra variables—those endpoints are not present in the current `main.py`. Treat that section as legacy / optional future work.

---

## Azure Resource Setup (High-Level)

1. Create an Azure AI Foundry Project in the Azure portal (Azure AI Foundry)
2. Deploy a model (e.g., `gpt-4o` or `gpt-4o-mini`) into the project
3. Add a Bing Search connection (Grounding) inside the project (Connections > Add > Bing)
4. Create Agents (through the Azure AI Foundry portal UI or SDK). For each agent record its ID:
	 * Deep Research Agent: enable deep research / reasoning mode if available
	 * Web Search Agent: attach Bing connection for grounded citations
5. Collect the Project Endpoint (found in Project Settings)
6. Decide on authentication method:
	 * Developer laptop: `az login` + DefaultAzureCredential
	 * CI / server: create a Service Principal and set `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`

---

## Environment Variables
Create a `.env` in the `api/` directory (or export in your shell). Minimum required:

```
PROJECT_ENDPOINT=<<your-foundry-project-endpoint>>
MODEL_DEPLOYMENT_NAME=<<primary-model-deployment>>
DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME=<<research-model-deployment>>
BING_RESOURCE_NAME=<<bing-connection-name>>
AGENT_ID=<<deep-research-agent-id>>
WEB_SEARCH_AGENT_ID=<<web-search-agent-id>>
```

---

## Local Setup & Run

From the `api/` folder:

```pwsh
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Azure CLI auth (interactive) OR ensure SPN vars are set
az login

# Run the API (reload for dev)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit:
* Swagger / OpenAPI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

---

## API Workflow Overview

1. Create a thread
2. Submit a Deep Research query referencing that thread
3. Poll status until `completed`
4. Download or stream the final formatted report
5. (Alternative) Use `/agent/stream` with the Web Search agent for real-time output

### Core Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/thread/create` | Returns new `thread_id` |
| POST | `/deep_research/submit?thread_id=...` | Starts a research run for query |
| GET | `/deep_research/{thread_id}/{run_id}/status` | Poll run status + messages + citations |
| GET | `/deep_research/{thread_id}/{run_id}/download?format=md|pdf` | Retrieve final report |
| POST | `/agent/stream` | Streaming responses from Web Search agent |

### Sample Calls
Create thread:
```pwsh
curl -X POST http://localhost:8000/thread/create
```

Submit research:
```pwsh
curl -X POST "http://localhost:8000/deep_research/submit?thread_id=THREAD_ID" `
	-H "Content-Type: application/json" `
	-d '{"query":"Impact of edge AI on energy efficiency"}'
```

Poll status:
```pwsh
curl http://localhost:8000/deep_research/THREAD_ID/RUN_ID/status
```

Download PDF:
```pwsh
curl "http://localhost:8000/deep_research/THREAD_ID/RUN_ID/download?format=pdf" | Out-File report.json
```

Streaming (Web Search agent):
```pwsh
curl -N -X POST http://localhost:8000/agent/stream `
	-H "Content-Type: application/json" `
	-d '{"thread_id":"THREAD_ID","message":"Latest Azure AI agent features","agent_id":"WEB_SEARCH_AGENT_ID"}'
```

> The streaming response yields incremental text chunks; when complete, a special `##CITATIONS:` JSON block is appended containing raw citation metadata.

---

## Notebooks Usage
Open notebooks in order:

1. `01A_DeepResearch_Agent_Creation.ipynb` – Programmatically create or validate a Deep Research agent and capture its ID
2. `01B_WebSearch_Agent_Creation.ipynb` – Create a Bing-grounded agent and capture its ID
3. `02A_DeepResearch_Testing.ipynb` – Execute research runs, inspect message structure, test markdown/PDF generation
4. `02B_WebSearch_Agent_testing.ipynb` – Exercise streaming output and citation formatting

You can then copy the produced agent IDs into your `.env` file.

---

## Output & Report Generation
When a Deep Research run completes:
* Messages are enumerated in chronological order
* Agent messages have inline citation references converted to `[n](url)` in JSON responses
* The download endpoint transforms the latest agent message into:
	* Markdown with `## References`
	* PDF (prefers ReportLab → falls back to FPDF2 → minimal handcrafted PDF)

---


## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| 401 / auth errors | Missing login or SPN vars | Run `az login` or set SPN env vars |
| Agent not found | Wrong ID or region mismatch | Re-check agent ID in Azure AI Foundry portal |
| No citations in output | Agent lacks Bing connection or query type | Attach Bing connection; re-run |
| PDF generation fails | Missing ReportLab & FPDF2 | Install one (`pip install reportlab`) |
| Empty streaming | Agent not configured for capability | Verify agent settings + model deployment |

---

## License
See `LICENSE` for details.

---
