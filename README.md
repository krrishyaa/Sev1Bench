---
title: Sev1Bench
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Sev1Bench

Sev1Bench is an OpenEnv incident-response environment for evaluating whether an agent can investigate a live production-style outage, identify the true failing service, apply the correct remediation, communicate truthfully, and restore service health before the episode ends.

The project is packaged for both local execution and deployment as a Docker-based Hugging Face Space.

## Overview

Each episode begins in a degraded state with active alerts, user impact, and deterministic evidence distributed across services. An agent must complete the full incident-response loop:

1. inspect logs with `read_logs`
2. identify the real root-cause service
3. apply the correct remediation to that exact service
4. post a truthful status update
5. restore the system to healthy state

The environment tracks:

- current tick count
- system health
- active alerts
- users affected
- root-cause discovery
- correct remediation
- truthful communication
- episode resolution or failure

## Task set

Sev1Bench defines three task IDs:

### `easy`
- Root cause service: `api-service`
- Initial health: `0.55`
- Correct remediation: `rollback`

### `medium`
- Root cause service: `auth-service`
- Initial health: `0.40`
- Correct remediation: `restart_service`

### `hard`
- Root cause service: `db-cluster`
- Initial health: `0.28`
- Correct remediation: `scale_up`

If an unknown task ID is provided, the environment falls back to `easy`.

## Supported actions

The environment exposes the following actions:

- `read_logs`
- `restart_service`
- `scale_up`
- `rollback`
- `post_status_update`

## Truthful status update behavior

Status updates are evaluated against the current recovery state.

Before recovery, a valid update must avoid falsely claiming the system is healthy or resolved, and should communicate that the team is still investigating, mitigating, or restoring service.

After recovery, a valid update should clearly communicate that service has been restored, the incident is resolved, or the system is healthy again.

## Reward and completion

The environment uses shaped per-step rewards for productive investigation, correct remediation, and truthful communication, along with penalties for misleading updates, unsupported actions, or incorrect remediations.

Episodes terminate when either:

- the incident is fully resolved, or
- the maximum step limit is reached before restoration

Final reward depends on:

- whether the root cause was found
- whether the correct fix was applied
- whether a truthful update was posted
- whether the incident was actually resolved
- time-decay based on the number of steps taken

## Repository layout

Submission-relevant files:

- `README.md`
- `inference.py`
- `models.py`
- `server/app.py`
- `server/environment.py`
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile`

Key implementation points:

- `server/environment.py` defines `IncidentResponseEnvironment`
- `server/app.py` exposes the FastAPI/OpenEnv app
- `models.py` defines the typed action, observation, and state models
- `inference.py` is the root evaluation entrypoint

## OpenEnv app contract

The environment class is `IncidentResponseEnvironment` in `server/environment.py`.

The application in `server/app.py` uses the official OpenEnv server interface via:

```python
from openenv.core.env_server import create_fastapi_app
```

and serves the environment through a FastAPI app. A minimal fallback app is also present for compatibility, while the canonical evaluation surface remains the OpenEnv API.

## Inference entrypoint

The root `inference.py` file is the evaluation-facing entrypoint.

It:

- uses the `openai` Python client
- reads `API_BASE_URL`
- reads `MODEL_NAME`
- requires `HF_TOKEN`
- supports task selection with `TASK_ID`
- prints structured execution markers:
  - `[START]`
  - `[STEP]`
  - `[END]`

### Environment variables

`inference.py` reads these variables:

- `HF_TOKEN`  
  Required. The script exits with an error summary if it is missing.

- `API_BASE_URL`  
  Optional. Default:
  `https://router.huggingface.co/v1`

- `MODEL_NAME`  
  Optional. Default:
  `openai/gpt-4o-mini`

- `TASK_ID`  
  Optional. Default:
  `easy`

- `SEED`  
  Optional. Default:
  `42`

## Local run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run inference locally

Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && python inference.py
```

### 4. Run a specific task

Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && set TASK_ID=medium && python inference.py
```

## Docker deployment

The included `Dockerfile`:

- uses `python:3.11-slim`
- installs dependencies from `requirements.txt`
- copies the root inference and model files plus the `server/` package
- serves the FastAPI app on port `7860`

Container startup command:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Hugging Face Space deployment

Recommended Space configuration:

- Space type: `Docker`
- Hardware: `CPU basic`
- Port: `7860`
- Secret to add: `HF_TOKEN`

Recommended deployment checklist:

1. Create or open the Space `Krrishya/Sev1Bench`
2. Set the Space SDK to `Docker`
3. Push the final repository contents
4. Add the secret `HF_TOKEN`
5. Wait for the Space build to complete successfully
6. Verify the landing page and `/docs`
7. Run final inference verification

## Verification

After deployment, verify the submission with both browser-visible and programmatic checks:

### Browser checks

- the Space loads successfully
- the landing page renders
- `/docs` is available
- the Space is serving on port `7860`

### Evaluation checks

Run:

```bat
set HF_TOKEN=your_token && python inference.py
```

Confirm the script emits:

- `[START]`
- one or more `[STEP]`
- `[END]`

Optional task-specific verification:

```bat
set HF_TOKEN=your_token && set TASK_ID=hard && python inference.py
```

## Dependencies

Runtime dependencies:

- `openenv-core>=0.2.3,<1.0.0`
- `openai>=1.0.0,<3.0.0`
- `fastapi>=0.110.0,<1.0.0`
- `uvicorn>=0.30.0,<1.0.0`
- `pydantic>=2.0.0,<3.0.0`

Python requirement:

- `>=3.11`

## Project links

- GitHub: `https://github.com/krrishyaa/Sev1Bench`
- Hugging Face Space: `https://huggingface.co/spaces/Krrishya/Sev1Bench`

## Final submission checklist

Before submitting:

- verify GitHub contains only the final repository surface
- verify the Hugging Face Space is built from the same final files
- verify `HF_TOKEN` is configured in the Space secrets
- verify the Space root page and `/docs` both load
- verify `python inference.py` completes and emits `[START]`, `[STEP]`, and `[END]`
- verify task IDs `easy`, `medium`, and `hard` all behave as expected
