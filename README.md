---
title: Sev1Bench
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Sev1Bench

Sev1Bench is an OpenEnv incident-response benchmark where an agent investigates a live outage, finds the true failing service, applies the correct remediation, posts a truthful status update, and restores service health before the episode ends.

## Tasks

Available task IDs:

- `easy`
- `medium`
- `hard`

If an unknown task ID is provided, the environment falls back to `easy`.

## Supported actions

The environment exposes these actions:

- `read_logs`
- `restart_service`
- `scale_up`
- `rollback`
- `post_status_update`

## Project structure

Important files:

- `README.md`
- `inference.py`
- `models.py`
- `server/app.py`
- `server/environment.py`
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile`

## Environment variables

`inference.py` uses:

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

## Local run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run inference in Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && python inference.py
```

Run a specific task:

```bat
set HF_TOKEN=your_token && set TASK_ID=medium && python inference.py
```

## Hugging Face Space setup

Recommended Space configuration:

- Space type: `Docker`
- Hardware: `CPU basic`
- Port: `7860`

Secrets / variables:

- Required secret: `HF_TOKEN`
- Optional variable: `API_BASE_URL`
- Optional variable: `MODEL_NAME`

You do not need to add `API_BASE_URL` or `MODEL_NAME` in Hugging Face if you want to use the defaults already defined in `inference.py`.

## Verification

Expected inference output includes:

- `[START]`
- one or more `[STEP]`
- `[END]`

Example:

```bat
set HF_TOKEN=your_token && python inference.py
```

Project links:

- GitHub: `https://github.com/krrishyaa/Sev1Bench`
- Hugging Face Space: `https://huggingface.co/spaces/Krrishya/Sev1Bench`
