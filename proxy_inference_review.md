# inference.py LiteLLM proxy review

## Files reviewed
- inference.py

## Key findings

### 1. The script requires `HF_TOKEN`, which can prevent any proxy request from being made
Current code:

```python
HF_TOKEN = os.getenv("HF_TOKEN")
...
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
if not HF_TOKEN.strip():
    raise ValueError("HF_TOKEN must not be empty")
...
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)
```

Why this matters:
- LiteLLM/OpenAI-compatible proxy setups commonly inject `API_KEY` or `OPENAI_API_KEY`.
- If the validator provides `API_KEY` but not `HF_TOKEN`, `main()` raises before `client.chat.completions.create(...)` is ever called.
- That directly explains a validator message like "no API calls were made through provided LiteLLM proxy."

### 2. `API_BASE_URL` defaults to the real OpenAI endpoint, which can bypass the proxy
Current code:

```python
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
```

Why this matters:
- If the runtime provides a proxy URL under another env var name, the script silently falls back to the OpenAI public endpoint.
- That means requests may not go through the provided LiteLLM proxy even if a real request happens.
- Submission-safe behavior should fail clearly when no runtime base URL is found, not silently substitute a hardcoded endpoint.

### 3. Env var support is too narrow
Current code only reads:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Likely runtime-compatible aliases that are not checked:
- base URL: `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `LITELLM_PROXY_URL`, `BASE_URL`
- API key: `API_KEY`, `OPENAI_API_KEY`, `LITELLM_API_KEY`
- model: `OPENAI_MODEL`, `INFERENCE_MODEL`, `HF_MODEL_ID`

This can cause either no request or a request to the wrong place with the wrong credentials.

### 4. There is only one guaranteed early LLM attempt
Current flow in `run_task()`:
- First step tries `_query_llm_once(...)`
- On failure it falls back to `_fallback_action(...)`
- Subsequent steps often use `_deterministic_policy_action(...)`

Why this matters:
- If the first request is blocked by env validation or auth mismatch, there may be no successful model traffic afterward.
- This is not the primary bug, but it reduces visibility of failures.

### 5. Runtime logging is misleading for debugging in a Space
Current config log shows only `hf_token_present`, not whether a generic API key was resolved. That makes UI/debug inspection harder.

## Recommended exact code changes

### A. Add a helper to resolve env aliases
Add near the top of `inference.py`:

```python
def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value
    return None
```

### B. Replace current global env reads
Replace:

```python
IMAGE_NAME = os.getenv("IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
```

with:

```python
IMAGE_NAME = os.getenv("IMAGE_NAME")

API_BASE_URL = _first_env(
    "API_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "LITELLM_PROXY_URL",
    "BASE_URL",
)

API_KEY = _first_env(
    "API_KEY",
    "OPENAI_API_KEY",
    "LITELLM_API_KEY",
    "HF_TOKEN",
)

MODEL_NAME = _first_env(
    "MODEL_NAME",
    "OPENAI_MODEL",
    "INFERENCE_MODEL",
    "HF_MODEL_ID",
) or "gpt-4.1-mini"
```

### C. Remove hardcoded endpoint fallback and fail clearly
Replace current validation in `main()`:

```python
if not API_BASE_URL.strip():
    raise ValueError("API_BASE_URL must not be empty")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
if not HF_TOKEN.strip():
    raise ValueError("HF_TOKEN must not be empty")
```

with:

```python
if API_BASE_URL is None:
    raise ValueError(
        "API base URL is required via API_BASE_URL, OPENAI_BASE_URL, OPENAI_API_BASE, LITELLM_PROXY_URL, or BASE_URL"
    )
if API_KEY is None:
    raise ValueError(
        "API key is required via API_KEY, OPENAI_API_KEY, LITELLM_API_KEY, or HF_TOKEN"
    )
```

This keeps the project submission-safe:
- no hardcoded endpoint
- no hardcoded credentials
- explicit runtime requirement

### D. Use the resolved API key in the OpenAI client
Replace:

```python
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)
```

with:

```python
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)
```

### E. Update runtime logging to reflect resolved API key presence
Replace:

```python
def log_runtime_config(task: str, model: str, api_base_url: str, hf_token_present: bool) -> None:
    print(
        "[CONFIG] "
        f"task={task} "
        f"model={model} "
        f"api_base_url={api_base_url} "
        f"hf_token_present={str(hf_token_present).lower()}",
        flush=True,
    )
```

with:

```python
def log_runtime_config(task: str, model: str, api_base_url: str, api_key_present: bool) -> None:
    print(
        "[CONFIG] "
        f"task={task} "
        f"model={model} "
        f"api_base_url={api_base_url} "
        f"api_key_present={str(api_key_present).lower()}",
        flush=True,
    )
```

and update the call site to:

```python
log_runtime_config(
    task=TASK_NAME,
    model=MODEL_NAME,
    api_base_url=API_BASE_URL,
    api_key_present=bool(API_KEY),
)
```

### F. Leave the actual request path as-is, but note behavior
This existing code is correct for sending a real request once configuration is fixed:

```python
response = client.chat.completions.create(
    model=model_name,
    messages=[...],
    temperature=0.0,
    max_tokens=300,
)
```

No change required there.

## Bottom line
Most likely root cause:
1. Validator injects proxy credentials as `API_KEY` / `OPENAI_API_KEY`
2. Current code requires `HF_TOKEN`
3. Script exits before making any request

Secondary likely issue:
1. Runtime base URL is not found under exactly `API_BASE_URL`
2. Current code falls back to `https://api.openai.com/v1`
3. Any request bypasses the provided LiteLLM proxy

## Suggested parent-agent implementation priority
1. Add env alias resolution helper
2. Stop requiring `HF_TOKEN` specifically
3. Stop defaulting `API_BASE_URL` to OpenAI public endpoint
4. Use `API_KEY` resolved from aliases for `OpenAI(...)`
5. Improve runtime config logging for Space visibility