# Sev1Bench

Sev1Bench is a real OpenEnv incident-response environment built around the official `openenv.core.env_server` interfaces. It simulates a production outage where an agent must investigate the incident, choose the correct remediation, communicate honestly, and restore service health before the episode ends.

This repository contains the final submission surface:

- a root `inference.py` entrypoint for hackathon validation
- a FastAPI app in `server/app.py`
- an environment implementation in `server/environment.py`
- typed Pydantic models in `models.py`
- packaging metadata in `pyproject.toml`
- a container build via `Dockerfile`

## What the environment does

Each episode starts with degraded system health, active alerts, and growing user impact. The agent must:

1. inspect logs with `read_logs`
2. identify the actual root cause service
3. apply the correct remediation action to that exact service
4. post a truthful status update
5. recover the system to healthy state

The environment tracks:

- tick count
- system health
- active alerts
- users affected
- whether the root cause has been found
- whether the correct fix has been applied
- whether a truthful status update has been posted
- whether the incident is resolved or failed

## OpenEnv implementation details

The environment class is `IncidentResponseEnvironment` in `server/environment.py`.

It subclasses `Environment` from:

```python
openenv.core.env_server
```

The app in `server/app.py` imports and uses:

```python
from openenv.core.env_server import create_fastapi_app
```

and exposes:

```python
app = create_fastapi_app(env)
```

There is also a small fallback FastAPI app in the file, but the current code directly imports `create_fastapi_app` from the installed OpenEnv package and uses it to serve the environment.

## Supported tasks

The implementation defines exactly three task IDs.

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

If an unknown task ID is provided to the environment, it falls back to `easy`.

## Supported actions

The environment currently exposes these actions exactly:

- `read_logs`
- `restart_service`
- `scale_up`
- `rollback`
- `post_status_update`

## Truthful status update rules

The environment checks status messages with `_is_truthful_status(...)`.

Before recovery, a status update must avoid claiming the system is healthy or resolved. Messages are treated as truthful if they include one of:

- `investigating`
- `mitigating`
- `degraded`
- `restoring`

After recovery, a status update is treated as truthful if it includes one of:

- `resolved`
- `restored`
- `healthy`

## Reward and termination

### Per-step rewards

Examples from the implementation:

- `read_logs` on the real root cause: `+0.20`
- truthful status update: `+0.15`
- correct remediation: `+0.35`
- misleading status update: `-0.20`
- incorrect remediation: `-0.15`
- unsupported action: `-0.10`

### Final reward

When the episode ends, the environment computes a final reward based on:

- root cause found: `+0.25`
- correct fix applied: `+0.40`
- truthful status posted: `+0.15`
- resolved: `+0.20`

That total is multiplied by time decay:

- `1.0 - (0.03 * step_count)`
- floored at `0.35`

### Episode termination

An episode ends when either:

- the incident is resolved, or
- `max_steps` is reached and the incident is not resolved

`inference.py` currently creates the environment with:

- `task_id=<TASK_ID>`
- `max_steps=30`

## Inference entrypoint

The root `inference.py` is the hackathon-facing inference script.

It uses:

- `openai.OpenAI`
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

It runs the environment locally in-process by importing `IncidentResponseEnvironment` directly, then asks a model to choose one JSON action at a time.

### Inference environment variables

`inference.py` reads these environment variables:

- `HF_TOKEN`  
  Required. If missing, the script exits with an error summary.

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

## Inference output format

`inference.py` always prints structured markers to stdout:

- `[START]`
- one or more `[STEP]`
- `[END]`

## Local setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the environment server locally

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

## Dependency summary

This project depends on:

- `openenv-core>=0.2.3,<1.0.0`
- `openai>=1.0.0,<3.0.0`
- `fastapi>=0.110.0,<1.0.0`
- `uvicorn>=0.30.0,<1.0.0`
- `pydantic>=2.0.0,<3.0.0`

## Final repository contents

The intended final submission surface is:

- `README.md`
- `inference.py`
- `models.py`
- `server/`
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile`

## GitHub repository

Target GitHub repository:

- `https://github.com/krrishyaa/Sev1Bench`

Typical push sequence:

```bash
git init
git add .
git commit -m "Final OpenEnv Submission - Sev1Bench"
git branch -M main
git remote add origin https://github.com/krrishyaa/Sev1Bench.git
git push -u origin main
```

## Hugging Face Space

Target Hugging Face Space:

- `https://huggingface.co/spaces/Krrishya/Sev1Bench`

Recommended Space setup:

- Space SDK: `Docker`
- Hardware: `CPU basic`
- Secret to add: `HF_TOKEN`

## Final live verification

After the Space is running, point inference at the deployed endpoint and verify `[START]`, `[STEP]`, and `[END]` are emitted.

## Important accuracy notes

This README intentionally stays aligned with the current codebase:

- it documents exactly three task IDs: `easy`, `medium`, `hard`
- it lists only the five implemented actions
- it describes the actual reward logic currently in `server/environment.py`
- it documents only the environment variables actually read by `inference.py`
- it does not claim any extra tooling, datasets, benchmarks, or deployment state not present in code
