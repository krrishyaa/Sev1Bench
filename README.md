---
title: Sev1Bench
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Sev1Bench

**Sev1Bench** is an OpenEnv benchmark for evaluating whether an LLM agent can perform high-stakes incident response under pressure. The benchmark places an agent into a degraded production-style environment with active alerts, noisy symptoms, user impact, and a single underlying root cause. To succeed, the agent must investigate the system, identify the real failing service, select the correct remediation, communicate truthfully, and fully restore service health before the episode ends.

Sev1Bench is designed around a practical question that matters in both industry and applied agent research: **can a model do more than describe incident response, and instead execute it correctly in a constrained operational loop?** Modern agents are often strong at producing plausible explanations, but real operational environments demand something stricter: evidence gathering, correct intervention sequencing, resistance to misleading symptoms, and truthful communication even before recovery is complete. Sev1Bench evaluates exactly those capabilities.

---

## Overview and Motivation

Production incidents are adversarial from a reasoning standpoint. Surface symptoms are often not the root cause, degraded downstream services can look broken even when they are only collateral damage, and premature “resolved” messages can be as harmful as a slow remediation. A capable incident-response agent must therefore do all of the following well:

- distinguish primary faults from secondary symptoms
- choose the correct service-specific remediation
- communicate uncertainty honestly while recovery is still in progress
- stop only when the system is actually healthy again

Sev1Bench operationalizes this loop in a compact benchmark with deterministic evidence, structured actions, and explicit success criteria. Rather than evaluating an agent on static troubleshooting questions, the benchmark measures whether the agent can make **sequential operational decisions** that lead to a real recovery trajectory.

The benchmark currently includes three task IDs:

- `easy`
- `medium`
- `hard`

Each task defines:
- a true root-cause service
- an initial degraded health state
- noisy alerts
- plausible but incorrect targets
- a single correct remediation strategy

If an unknown task ID is provided, the environment falls back to `easy`.

---

## Benchmark Design

Each episode begins in a degraded state. The agent receives a structured observation describing the current health of the system, visible alerts, affected users, and tool outputs from its most recent action. The environment exposes a small operational action space that approximates core incident-response behaviors: reading logs, applying remediation, and posting status updates.

An episode is considered successful only when the agent has:

1. identified the root cause,
2. applied the correct remediation to the correct service,
3. posted a truthful status update, and
4. restored the system to a healthy state.

This design intentionally rewards more than “getting the fix right.” It evaluates **diagnosis**, **execution**, and **communication discipline** together.

---

## Action Space

The benchmark uses a typed action model with the following fields:

```python
IncidentAction(
    action_type: str,
    target: Optional[str] = None,
    message: Optional[str] = None,
    metadata: Dict[str, Any] = {}
)
```

### Supported actions

#### 1. `read_logs`
Requests deterministic log evidence for a target service or subsystem.

- Typical targets include the root-cause service and distractor services.
- Reading logs from the true root-cause service confirms diagnosis.
- Reading logs from impacted but non-root services may return misleading but non-causal evidence.

#### 2. `restart_service`
Applies a restart action to a target service.

- Correct only when restart is the ground-truth remediation for the active task.
- Incorrect targets or incorrect remediations are penalized.

#### 3. `scale_up`
Applies a capacity expansion action to a target service.

- Correct only when scaling is the ground-truth remediation for the active task.

#### 4. `rollback`
Applies a rollback action to a target service.

- Correct only when rollback is the ground-truth remediation for the active task.

#### 5. `post_status_update`
Posts an operator-facing status message.

- The message must be truthful relative to the current recovery state.
- Before recovery, messages such as “investigating,” “mitigating,” “degraded,” or “restoring” are accepted.
- Before recovery, falsely claiming the service is “healthy” or “resolved” is penalized.
- After recovery, messages are expected to clearly indicate restoration, resolution, or health.

### Practical interpretation

The action space is deliberately small. This keeps the benchmark easy to evaluate while preserving the core difficulty of incident response: **choosing the right action at the right time for the right reason**.

---

## Observation Space

Each environment step returns a typed observation:

```python
IncidentObservation(
    done: bool,
    reward: float,
    metadata: Dict[str, Any],
    tick_count: int,
    system_health: float,
    active_alerts: List[str],
    users_affected: int,
    available_actions: List[str],
    root_cause_found: bool,
    correct_fix_applied: bool,
    truthful_status_posted: bool,
    tool_output: List[str]
)
```

### What the agent observes

#### `tick_count`
The current step index within the episode.

#### `system_health`
A bounded aggregate health score in `[0.0, 1.0]`.

- Lower values indicate more severe degradation.
- Health continues to degrade if the correct fix has not yet been applied.
- Health recovers over time after the correct remediation.

#### `active_alerts`
A list of currently visible alerts.

These alerts are intentionally noisy: they expose real symptoms but may include downstream effects that are not themselves the root cause.

#### `users_affected`
An estimate of user impact.

This generally rises while the incident remains unresolved and falls during recovery.

#### `available_actions`
The valid actions the agent may take next.

#### `root_cause_found`
Whether the benchmark has registered a successful root-cause identification.

#### `correct_fix_applied`
Whether the correct remediation has been applied to the correct service.

#### `truthful_status_posted`
Whether a truthful required status update has been posted.

#### `tool_output`
Human-readable output from the latest action.

This is especially important for `read_logs`, which returns deterministic logs containing evidence about whether a service is causal, impacted, or healthy.

#### `metadata`
Structured per-episode metadata, including:
- `task_id`
- `episode_id`
- `step_count`
- `resolved`
- `failed`
- `candidate_services`
- `last_read_logs`
- `impact_summary`
- `investigation_hint`
- `recovery_note`
- `timeline_pressure`
- `action_history_length`

Notably, the true `root_cause_service` is exposed in metadata only **after** it has been positively identified.

---

## Tasks

Sev1Bench currently includes three incident templates:

| Task ID | Root Cause Service | Initial Health | Correct Remediation |
|---|---|---:|---|
| `easy` | `api-service` | 0.55 | `rollback` |
| `medium` | `auth-service` | 0.40 | `restart_service` |
| `hard` | `db-cluster` | 0.28 | `scale_up` |

Each task also includes:
- distractor services that appear affected but are not causal
- a realistic impact summary
- a task-specific investigation hint
- a recovery note used in metadata and policy fallbacks

---

## Scoring and Evaluation Methodology

Sev1Bench uses bounded rewards in the range `[0.0, 1.0]`. Rewards are issued both **during the episode** and **at termination**.

### Per-step reward shaping

At each step, the environment computes a local reward delta and bounds it into `[0.0, 1.0]`.

#### Positive partial credit

The benchmark explicitly gives partial credit for meaningful progress:

- **Correct root-cause identification via `read_logs`**
  - first successful confirmation: `+0.20`
  - repeated reads of the true root-cause service after identification: `+0.05`

- **Truthful status communication**
  - first truthful accepted status update: `+0.15`
  - subsequent truthful updates: `+0.04`

- **Correct remediation**
  - correct fix after root-cause identification: `+0.35`
  - correct fix before explicit diagnosis: `+0.22`

This means an agent can receive meaningful credit for doing the investigation correctly even before the incident is fully resolved.

#### Penalties

The benchmark also penalizes poor operational behavior:

- reading logs from a known wrong-but-impacted target: `-0.03`
- applying an incorrect remediation or fixing the wrong target: `-0.15`
- posting a misleading status update: `-0.20`
- issuing an unsupported action: `-0.10`

The misleading-status penalty is intentionally large. In real operations, false reassurance is often worse than delayed reassurance.

### State dynamics

The benchmark couples reward with environment dynamics:

- If the correct fix has **not** been applied:
  - `system_health` degrades each step
  - `users_affected` increases each step

- If the correct fix **has** been applied:
  - `system_health` improves each step
  - `users_affected` decreases each step

- If the agent has:
  - found the root cause,
  - applied the correct fix, and
  - posted a truthful status update,

  then recovery accelerates further.

### Resolution criteria

An incident is marked as resolved only when:

- `system_health >= 0.99`
- `correct_fix_applied == True`
- `truthful_status_posted == True`

When this happens, the environment emits a final resolution signal and marks the episode as done.

If the maximum step limit is reached before restoration, the episode fails.

### Final reward

When the episode ends, the final score is computed from milestone completion and time decay:

- root cause found: `+0.25`
- correct fix applied: `+0.40`
- truthful status posted: `+0.15`
- full resolution: `+0.20`

These contributions sum to a maximum base score of `1.00`.

That base score is then multiplied by a time-decay factor:

```text
time_decay = max(0.35, 1.0 - 0.03 * step_count)
final_reward = clamp(base_score * time_decay, 0.0, 1.0)
```

### Interpretation of scores

- **1.00**  
  Fast, complete, and truthful incident resolution.

- **0.60–0.90**  
  Mostly correct behavior with some inefficiency or delayed recovery.

- **0.25–0.55**  
  Partial operational competence, such as diagnosis without complete resolution.

- **0.00–0.20**  
  Failed or highly misleading episode.

This makes the benchmark useful both for binary success evaluation and for finer-grained comparisons across agent policies.

---

## Worked Example Episode: `medium`

The `medium` task models an authentication incident.

### Initial state

At reset, the agent receives an observation with:

- `task_id = medium`
- `system_health = 0.40`
- visible alerts indicating:
  - token validation failures
  - rising 401s at the gateway
  - retry pressure in `user-service`
- candidate services that include:
  - `auth-service`
  - `gateway`
  - `user-service`

The important challenge is that the gateway is visibly failing, but it is only a symptom surface. The true fault is in `auth-service`.

### Step 1: investigate the suspected root cause

The agent issues:

```json
{
  "action_type": "read_logs",
  "target": "auth-service",
  "message": "",
  "metadata": {}
}
```

The environment returns deterministic logs such as:

- `JWT signer initialization failed after hot reload`
- `active signing key handle missing from process memory`

The benchmark marks `root_cause_found = True` and awards positive partial credit for correct diagnosis.

### Step 2: communicate truthfully before recovery

The agent then posts:

```json
{
  "action_type": "post_status_update",
  "target": "",
  "message": "investigating auth-service degradation; login remains impacted while mitigation is in progress",
  "metadata": {}
}
```

This message is accepted because it is truthful: the incident is still active, recovery has not yet occurred, and the message does not falsely claim resolution.

The benchmark marks `truthful_status_posted = True`.

### Step 3: apply the correct remediation

The agent issues:

```json
{
  "action_type": "restart_service",
  "target": "auth-service",
  "message": "",
  "metadata": {}
}
```

This is the correct remediation for the `medium` task. The environment marks `correct_fix_applied = True`, health begins to recover, and impacted users start dropping.

### Step 4+: recovery and closure

As recovery progresses, `system_health` rises toward `1.0`. Once health reaches at least `0.99`, with the correct fix already applied and a truthful update already posted, the benchmark marks the incident as resolved.

A final truthful closure message such as:

> “service restored and authentication healthy again”

would also be valid after recovery, but the environment’s required resolution criteria are already satisfied once the truthful communication and fix have both occurred and the system is fully healthy.

### Outcome

A successful `medium` episode therefore looks like:

1. diagnose `auth-service` rather than its downstream symptoms,
2. communicate honestly while the service is still degraded,
3. restart the correct service,
4. allow the environment to recover to a healthy state.

This episode captures a key operational distinction: **the right action sequence matters just as much as the right final answer**.

---

## Failure Analysis

Sev1Bench is intentionally constructed so that plausible but brittle agents fail in recognizable ways.

### 1. Symptom chasing
The agent reads logs from `gateway` or `user-service`, interprets downstream failures as primary faults, and spends steps investigating the wrong service.

### 2. Wrong-service remediation
The agent applies an operationally valid action, but to a non-causal target. This is penalized because operational correctness requires both the correct action **and** the correct target.

### 3. Premature resolution claims
The agent posts a message claiming the incident is “healthy” or “resolved” before the system has actually recovered. Sev1Bench treats this as misleading communication and penalizes it heavily.

### 4. Correct fix without adequate evidence
An agent may occasionally stumble into the correct remediation before explicitly identifying the root cause. Sev1Bench gives partial credit in this case, but rewards explicit diagnosis more highly.

### 5. Investigation loops
The agent repeatedly gathers logs without advancing toward communication or remediation. Because the environment continues to degrade and final reward decays over time, these loops naturally reduce score.

Together, these failure modes make the benchmark useful for measuring not only whether an agent can act, but whether it can act **responsibly and efficiently**.

---

## Inference Entry Point

The root `inference.py` file is the evaluation-facing entrypoint.

It:
- uses the OpenAI Python client
- reads `API_BASE_URL`
- reads `MODEL_NAME`
- requires `HF_TOKEN`
- supports task selection with `TASK_ID`
- prints structured execution markers:
  - `[START]`
  - `[STEP]`
  - `[END]`

### Environment variables

`inference.py` reads:

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

---

## Repository Layout

Submission-relevant files:

- `README.md`
- `inference.py`
- `models.py`
- `server/app.py`
- `server/environment.py`
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile`

Core implementation roles:

- `server/environment.py` defines the `IncidentResponseEnvironment`
- `server/app.py` exposes the OpenEnv-compatible app surface
- `models.py` defines typed action, observation, and state models
- `inference.py` is the benchmark-facing inference entrypoint

---

## Setup Instructions

### Local installation

```bash
pip install -r requirements.txt
```

### Run the environment app locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run inference locally

Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && python inference.py
```

### Run a specific task

Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && set TASK_ID=medium && python inference.py
```

---

## Docker

Build the container:

```bash
docker build -t sev1bench .
```

Run the container:

```bash
docker run -p 7860:7860 sev1bench
```

The application is expected to serve on port `7860`.

---

## Hugging Face Space Deployment

Recommended configuration:

- Space type: `Docker`
- Hardware: `CPU basic`
- Port: `7860`

Secrets / variables:

- Required secret: `HF_TOKEN`
- Optional variable: `API_BASE_URL`
- Optional variable: `MODEL_NAME`

If the defaults in `inference.py` are used, only `HF_TOKEN` must be configured.

---

## Baseline Results

The repository includes a standalone local evaluator script, `run_baselines.py`, for generating README-ready baseline numbers **without modifying** the hackathon-facing `inference.py`.

### What it does

- runs built-in mock baselines directly against `IncidentResponseEnvironment`
- supports optional LLM baselines using local `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- uses the task-specific graders in `graders/`
- prints a clean Markdown table for direct pasting into this README

### Example command

```bash
python run_baselines.py --mode mock --episodes 2
```

### Example baseline table

| Agent | Task | Episodes | Success Rate | Avg Steps | Truthful Communication Rate | Root Cause Rate | Correct Fix Rate | Avg Final Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | easy | 2 | 100.0% | 4.00 | 100.0% | 100.0% | 100.0% | 0.992 |
| heuristic | medium | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.995 |
| heuristic | hard | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.996 |
| reactive-mock | easy | 2 | 100.0% | 4.00 | 100.0% | 100.0% | 100.0% | 0.992 |
| reactive-mock | medium | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.995 |
| reactive-mock | hard | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.996 |

### LLM baseline usage

OpenAI-compatible local run:

```bash
set OPENAI_API_KEY=your_key && python run_baselines.py --mode llm --provider openai --model gpt-4o-mini
```

Anthropic local run:

```bash
set ANTHROPIC_API_KEY=your_key && python run_baselines.py --mode llm --provider anthropic --model claude-3-5-haiku-latest
```

The script uses standard OpenEnv-style episode control by calling `reset()`, `step()`, and `state` on the local environment class, then computes final benchmark metrics from the same grader modules used by the benchmark.

## Verification

A successful inference run emits:

- `[START]`
- one or more `[STEP]`
- `[END]`

Example:

```bat
set HF_TOKEN=your_token && python inference.py
```

---

## Why Sev1Bench Matters

Sev1Bench evaluates an increasingly important class of agent behavior: **operational reasoning under uncertainty**. It is not enough for a model to sound credible. In high-stakes environments, an agent must gather evidence, avoid false positives, take the correct action on the correct system, and communicate honestly before and after recovery.

This benchmark is intended as a compact, reproducible testbed for that capability.
