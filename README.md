# DeepDeliberate: Multi‑Agent Simulation for Synthetic Data Generation

DeepDeliberate is a lightweight, configurable framework that simulates realistic conversations to generate high‑quality synthetic datasets for fine‑tuning and agent evaluation. It orchestrates three cooperating agents:

- Generator: Creates user queries conditioned by a persona (DeepSeek R1 on Azure AI Inference).
- Assistant: Answers using your target agent (PydanticAI + Azure OpenAI `o4-mini`).
- Judge: Scores the assistant response across weighted dimensions (DeepSeek R1).

Run in fully automated batches (auto) or step‑by‑step with manual approval (approve). Outputs are written to CSV and JSON for easy analysis.

Note: This is a research project intended to bootstrap datasets and iterate quickly toward MVPs or pilots; it is not production‑hardened.

## Why DeepDeliberate

- Reduce the startup cost of building a fine‑tuning dataset when real data is diffuse across SMEs and documents.
- Simulate personas and edge cases at scale, overnight if needed, to quickly test agent behavior and collect examples.
- Use open models (DeepSeek R1) where reasoning traces are helpful, and keep your assistant model choice flexible (Azure OpenAI in examples).

## How It Works

- Generator: Prompts DeepSeek R1 to produce a realistic query based on a chosen persona, templates, and tone. Reasoning is captured where available.
- Assistant: Executes your PydanticAI agent (examples provided) backed by Azure OpenAI `o4-mini`.
- Judge: Evaluates the response with DeepSeek R1 using persona‑weighted criteria (accuracy, relevance, completion, safety). Optional enhanced mode adds a fifth dimension.
- Modes: 
  - auto: runs end‑to‑end without intervention.
  - approve: pauses between steps to Continue/Retry/Exit.
- Outputs: CSV logs per interaction, decisions CSV (approve mode), checkpoints for recovery, and a session summary JSON.

## Repo Structure

- `deepdeliberate.py`: Main CLI entrypoint (or install and use the `deepdeliberate` command).
- `deepdeliberate/`: Framework core (query gen, evaluator, orchestration, logging).
- `setup/agents/`: Example PydanticAI agents (customer service, education, healthcare).
- `setup/configs/`: Example persona configurations (JSON).
- `.env.example`: Environment variables template for DeepSeek and Azure OpenAI.
- `test_results/`: Default output directory (see config for domain‑specific subfolders).

## Requirements

- Python 3.13.3+
- Azure OpenAI access (for the assistant agent) and credentials.
- DeepSeek R1 access on Azure AI Inference and credentials.

## Setup

1) Clone and enter the repo

```bash
git clone https://github.com/Deep-Deliberate/Deep-Deliberate.git
cd Deep-Deliberate
```

2) Create an environment and install deps

```bash
# Option A: uv (recommended)
uv sync

# Option B: venv + pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

3) Configure environment variables

```bash
cp .env.example .env
# Edit .env with your values
```

Key variables (see `.env.example`):

- `DEEPSEEK_API_KEY`, `DEEPSEEK_ENDPOINT`, `DEEPSEEK_MODEL_NAME` (e.g., `DeepSeek-R1-0528`)
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME` (e.g., `o4-mini`)
- Rate limiting knobs: `API_RATE_LIMIT_PER_MINUTE`, `REQUEST_TIMEOUT_SECONDS` (env); `MAX_CONCURRENT_REQUESTS` (config)

Verify CLI:

```bash
python deepdeliberate.py --help
# or after install
deepdeliberate --help
```

## Quick Start

The examples below match the demo flow and your local paths. Adjust as needed.

```bash
cd /Users/NarotamDhaliwal/Documents/GitHub/Deep-Deliberate
source .venv/bin/activate

# Approve/manual mode (step-by-step)
python deepdeliberate.py \
  -file setup/agents/customer_service_agent.py \
  -mode approve \
  -config setup/configs/customer_service_config.json \
  -persona frustrated_customer \
  -count 1

# Auto mode (fully automated)
python deepdeliberate.py \
  -file setup/agents/customer_service_agent.py \
  -mode auto \
  -config setup/configs/customer_service_config.json \
  -persona frustrated_customer \
  -count 3

# Using uv to run
uv run python deepdeliberate.py \
  -file setup/agents/customer_service_agent.py \
  -mode auto \
  -config setup/configs/customer_service_config.json \
  -persona frustrated_customer \
  -count 2
```

Flags:

- `-file`: Path to your PydanticAI agent file.
- `-mode`: `auto` or `approve`.
- `-config`: Which persona config JSON to use.
- `-persona`: Which persona from the config to run (e.g., `frustrated_customer`).
- `-count`: Number of conversations to generate.
- `-enhanced`: Optional; use enhanced 5‑dimension evaluator.

Tip: In approve mode you’ll be prompted to Continue/Retry/Exit at key steps. In auto mode, runs unattended (great for overnight batches).

## Outputs

For a customer service run (see `setup/configs/customer_service_config.json`):

- Interactions CSV: `test_results/customer_service/<session_id>_interactions.csv`
- Decisions CSV (approve mode): `test_results/customer_service/<session_id>_decisions.csv`
- Checkpoints JSON: `test_results/customer_service/<session_id>_checkpoints.json`
- Summary JSON: `test_results/customer_service/<session_id>_summary.json`

Note: `<session_id>` already includes a timestamp and unique suffix (e.g., `session_20250914_223125_0e08d81f`).

Each interaction includes timestamp, persona, query, agent response, evaluation score, reasoning summary, execution time, and metadata.

## Using the Sample Agents

- Customer service: `setup/agents/customer_service_agent.py`
- Education tutor: `setup/agents/education_agent.py`
- Healthcare info: `setup/agents/healthcare_agent.py`

Each uses PydanticAI with an Azure OpenAI model (`o4-mini`). Ensure your `.env` has valid Azure settings.

## Configuring Personas

Persona JSON structure (examples in `setup/configs/*.json`):

- `name`, `description`, `behavioral_patterns`, `tone_specifications`
- `scenario_templates` with `{placeholders}` and `template_parameters` values
- `evaluation_criteria` weights (must sum to 1.0)
- `logging_settings.output_directory` controls where CSV/JSON outputs go

DeepDeliberate uses these to generate queries with a variety of strategies (standard, boundary conditions, adversarial inputs, ambiguous requests, multi‑part, context‑aware, error scenarios). The judge applies persona‑weighted scoring.

## Create Your Own Agent

1) Implement a PydanticAI agent (see `setup/agents/customer_service_agent.py` for a template). It should construct an `Agent` and optionally a deps model for context.
2) Point the CLI `-file` at your agent file.
3) Pick or create a persona config under `setup/configs/` and pass it with `-config`.

If your agent needs tools/RAG, wire them in as usual—DeepDeliberate treats your agent as the black‑box assistant.

## Execution Modes

- auto: Generates query → gets agent response → evaluates → logs, repeated for `-count`.
- approve: Pauses between steps to review, Retry, Continue, or Exit; logs your decisions.

## Troubleshooting

- Missing keys: The CLI prints which env vars are missing (e.g., `DEEPSEEK_API_KEY`, Azure vars). Copy `.env.example` and fill in values.
- Azure errors: Ensure `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, and `AZURE_OPENAI_DEPLOYMENT_NAME` match your deployment.
- Rate limits: Lower `API_RATE_LIMIT_PER_MINUTE` (e.g., 10–15) and reduce `MAX_CONCURRENT_REQUESTS`.
- Timeouts: Increase `REQUEST_TIMEOUT_SECONDS`.
- Output paths: Confirm `logging_settings.output_directory` in your config; directories are created automatically.

## Notes on Reasoning Content

- DeepSeek R1 (generator and judge) can emit reasoning traces, which DeepDeliberate captures for transparency and debugging.
- Azure OpenAI models like `o4-mini` do not return chain‑of‑thought; only final responses are logged.

## License

MIT. See `LICENSE` if present in your distribution.

## Acknowledgements

- Built around PydanticAI, Azure AI Inference (for DeepSeek R1), and Azure OpenAI.
- Inspired by the need to quickly assemble realistic, persona‑rich datasets for agent development.
