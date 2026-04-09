---
title: Sev1Bench
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Sev1Bench

Sev1Bench is an OpenEnv-based incident response benchmark.  
An agent must investigate a degraded production-style environment, identify the true failing service, apply the correct remediation, communicate status truthfully, and restore service health before the episode ends.

## Overview

The benchmark is designed to evaluate whether an agent can complete a realistic incident-response workflow:

1. inspect evidence
2. identify the root cause
3. apply the correct remediation
4. post a truthful status update
5. restore the system to a healthy state

## Task IDs

Supported task IDs:

- `easy`
- `medium`
- `hard`

If an unknown task ID is provided, the environment falls back to `easy`.

## Supported actions

The environment supports the following actions:

- `read_logs`
- `restart_service`
- `scale_up`
- `rollback`
- `post_status_update`

## Repository structure

Main files:

- `README.md`
- `inference.py`
- `models.py`
- `server/app.py`
- `server/environment.py`
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile`

## Environment variables

`inference.py` reads the following environment variables:

- `HF_TOKEN`  
  Required.

- `API_BASE_URL`  
  Optional. Default: `https://router.huggingface.co/v1`

- `MODEL_NAME`  
  Optional. Default: `openai/gpt-4o-mini`

- `TASK_ID`  
  Optional. Default: `easy`

- `SEED`  
  Optional. Default: `42`

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run inference locally in Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && python inference.py
```

Run a specific task:

```bat
set HF_TOKEN=your_token && set TASK_ID=medium && python inference.py
```

## Hugging Face Space

Recommended configuration:

- Space SDK: `Docker`
- Port: `7860`

Recommended secret:

- `HF_TOKEN`

Optional variables:

- `API_BASE_URL`
- `MODEL_NAME`

If you are using the defaults in `inference.py`, you only need to set `HF_TOKEN`.

## Verification

The inference script should print:

- `[START]`
- one or more `[STEP]`
- `[END]`

Example:

```bat
set HF_TOKEN=your_token && python inference.py
```

## Links

- GitHub: `https://github.com/krrishyaa/Sev1Bench`
- Hugging Face Space: `https://huggingface.co/spaces/Krrishya/Sev1Bench`
