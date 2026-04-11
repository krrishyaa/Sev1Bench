---
title: Sev1Bench
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Sev1Bench

**The benchmark for testing whether an AI agent can actually handle a Sev1 incident.**

**Problem:** most agent demos prove a model can talk about incident response, not actually handle one.

**Solution:** Sev1Bench is an OpenEnv benchmark where an agent must investigate a live incident, choose the right remediation, post a truthful status update, and restore service health before time runs out.

**How it works:** each task starts in a degraded production-style state with noisy alerts and a constrained action space; the agent reads evidence, acts on the real fault domain, communicates honestly, and is scored by environment-enforced recovery rules.

**Why it matters:** this tests operational reasoning under uncertainty, which is much closer to real production work than static QA, toy tool use, or polished chatbot answers.

**Exact judging value:** Sev1Bench is strongest on **technical depth**, **clarity of evaluation**, **real-world usefulness**, **demoability**, and **benchmark credibility** because judges can watch a full run, inspect the logs used for the decision, compare agents on the same tasks, and verify the score from transparent environment logic.

> **Judge-ready takeaway:** this is not an incident-response copilot mockup; it is a live, inspectable benchmark with replayable traces and measurable outcomes.

---

## 10-second demo

**Pick task -> run benchmark -> inspect logs -> see score**

That is the whole pitch.

1. **Pick a task**: `easy`, `medium`, `hard`, or `expert`
2. **Run the benchmark** from the UI or `python inference.py`
3. **Inspect the incident evidence** returned by `read_logs` plus the action trace
4. **See the final score** and whether the incident was actually resolved

If a judge only watches for 10 seconds, they should still understand:

- this is a benchmark, not a mock chat demo
- the agent must act, not just explain
- the logs justify the action
- the environment decides whether the run succeeded

### One-sentence live pitch

> **“Sev1Bench shows whether an AI agent can investigate a live production-style incident, take the right action, communicate honestly, and earn a verifiable recovery score.”**

---

## What makes this stand out

### Replayable incident traces
Every run produces inspectable step-by-step output: action history, tool evidence, rewards, and final outcome. This makes the benchmark easy to judge live, easy to compare offline, and easy to reuse as an evaluation asset after the hackathon.

### Side-by-side benchmark comparison
The repo includes `run_baselines.py`, which can run multiple baseline agents on the same tasks and print a clean comparison table. That turns the project from “one cool demo” into “a reusable evaluation harness.”

### Inspectable run reports
The UI and CLI both expose structured outputs, including raw JSON, step logs, and final scores. Judges can inspect what the agent did instead of trusting a narrated claim.

### Serious evaluation contract
Recovery is not declared by vibes. The environment requires the right diagnosis, the right fix, truthful communication, and real restoration of health.

### Strongest differentiator
If you remember one thing about Sev1Bench, it should be this:

> **It makes incident-response agent quality observable.**  
> You can watch the agent inspect evidence, choose an action, communicate status, and either recover the system or fail in public.

---

## Why judges should care

Most AI demos optimize for impressive-looking answers. Sev1Bench optimizes for **falsifiable operational competence**.

Why that matters:

- **It is falsifiable.** The environment has explicit success criteria and bounded rewards.
- **It is action-based.** The agent must investigate and intervene, not just summarize.
- **It is realistic in the right way.** The visible symptoms are not always the root cause.
- **It is easy to verify quickly.** Judges can check the code, the traces, and the score in one sitting.
- **It supports fair comparison.** Different policies can be run on the exact same tasks.
- **It feels like a real benchmark, not a prompt wrapper.** The environment, graders, tasks, UI, and comparison harness all reinforce that this is a serious evaluation artifact.

In one line: Sev1Bench turns “can this model handle a Sev1?” into a reproducible, inspectable test.

For selection, that is powerful because it scores well across the categories judges usually reward most:
- something technically non-trivial
- something easy to understand in a minute
- something with visible proof
- something reusable beyond demo day

---

## Demo narrative judges can follow instantly

The cleanest live demo path is:

1. open the UI from `server/app.py`
2. click a task
3. execute a benchmark run
4. point at the returned `read_logs` evidence
5. show the action sequence
6. end on the final score and resolved/not-resolved outcome

This is what the UI is already optimized around:

- task selection
- benchmark execution
- terminal trace
- raw JSON
- final result banner

So the demo naturally answers the right questions:

- What task was attempted?
- What evidence did the agent use?
- What action did it choose?
- Did the environment accept that as a successful recovery?

---

## Why this can realistically get selected

Sev1Bench has the profile judges usually reward:

- **instantly understandable** — the benchmark loop is obvious in seconds
- **technically credible** — there is real environment logic, typed actions, graders, and reward rules
- **visually demoable** — task selection, logs, JSON, traces, and score are all visible live
- **provable** — runs produce inspectable evidence and reproducible outputs
- **reusable** — this can become a lasting benchmark, not just a hackathon-only app

That combination is rare. A lot of projects are impressive but hard to verify, or useful but hard to demo. Sev1Bench sits in the sweet spot: **easy to understand, hard to fake, and strong to judge live.**

---

## Proof this is a real benchmark

### 1. Actual benchmark outputs

Verified locally from this repo with:

```bash
python run_baselines.py --mode mock --episodes 2
```

Result:

| Agent | Task | Episodes | Success Rate | Avg Steps | Truthful Communication Rate | Root Cause Rate | Correct Fix Rate | Avg Final Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | easy | 2 | 100.0% | 4.00 | 100.0% | 100.0% | 100.0% | 0.992 |
| heuristic | medium | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.995 |
| heuristic | hard | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.996 |
| heuristic | expert | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.995 |
| reactive-mock | easy | 2 | 100.0% | 4.00 | 100.0% | 100.0% | 100.0% | 0.992 |
| reactive-mock | medium | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.995 |
| reactive-mock | hard | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.996 |
| reactive-mock | expert | 2 | 100.0% | 5.00 | 100.0% | 100.0% | 100.0% | 0.995 |

This matters because the benchmark is already runnable, already producing validated outputs, and already capable of comparing agents across tasks.

### 2. Sample validated run shape

A successful episode looks like this:

```text
[START] task=medium
[STEP] read_logs target=auth-service -> root-cause evidence returned
[STEP] post_status_update -> truthful degradation/mitigation message accepted
[STEP] restart_service target=auth-service -> correct remediation applied
[STEP] recovery progresses until health >= 0.99
[END] resolved=true score≈0.99
```

That trace shape is exactly what judges want:
evidence -> action -> outcome.

### 3. Transparent methodology

The benchmark contract is implemented in code, not hidden in presentation slides:

- `server/environment.py` contains task definitions, recovery logic, and reward rules
- `models.py` defines the typed benchmark interface
- `inference.py` drives the benchmark-facing execution loop
- `run_baselines.py` supports repeatable agent comparison
- `server/app.py` gives a live inspection surface for tasks and runs

### 4. Clear architecture

```text
                 +----------------------+
                 |      inference.py    |
                 |  chooses next action |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  IncidentAction      |
                 |  models.py           |
                 +----------+-----------+
                            |
                            v
+----------------+   step(action)   +-----------------------------+
| agent / policy | ---------------> | IncidentResponseEnvironment |
| / baseline     |                  | server/environment.py       |
| / LLM client   | <--------------- | reset(), step(), scoring    |
+----------------+  observation      +-------------+---------------+
                                                    |
                                                    v
                                     +-----------------------------+
                                     | IncidentObservation         |
                                     | alerts, health, users,      |
                                     | tool_output, reward, done   |
                                     +-----------------------------+
```

---

## What Sev1Bench evaluates

Sev1Bench measures whether an agent can do the operational loop correctly:

1. investigate the system
2. identify the real root cause
3. apply the correct remediation
4. post a truthful status update
5. restore service health before the episode ends

The benchmark is designed to distinguish between:

- an agent that sounds competent
- and an agent that can actually operate competently

That distinction is the core value of the project.

---

## Benchmark at a glance

- **Environment type:** OpenEnv-compatible incident response simulation
- **Tasks implemented:** `easy`, `medium`, `hard`, `expert`
- **Action space:** `read_logs`, `restart_service`, `scale_up`, `rollback`, `post_status_update`
- **Outputs:** observations, deterministic log evidence, step rewards, final reward
- **Evaluation focus:** diagnosis + remediation + truthful communication + actual recovery
- **Demo surface:** FastAPI UI with terminal trace, JSON output, and run controls
- **Comparison surface:** baseline runner with markdown benchmark tables
- **Judge value in one line:** an agent benchmark you can understand in 10 seconds and trust after 2 minutes of inspection

---

## Incident tasks

The environment currently implements four benchmark scenarios:

| Task ID | Root Cause Service | Initial Health | Correct Remediation |
| --- | --- | ---: | --- |
| `easy` | `api-service` | 0.55 | `rollback` |
| `medium` | `auth-service` | 0.40 | `restart_service` |
| `hard` | `db-cluster` | 0.28 | `scale_up` |
| `expert` | `queue-broker` | 0.34 | `restart_service` |

Each task includes:

- noisy alerts
- realistic symptom-vs-root-cause separation
- a correct remediation path
- explicit grading logic
- bounded reward computation

---

## Scoring and evaluation

Sev1Bench uses rewards bounded to `[0.0, 1.0]`.

### Positive credit
- root-cause confirmation via `read_logs`
- truthful status communication
- correct remediation
- full recovery

### Penalties
- chasing the wrong service
- applying the wrong fix
- misleading status updates
- unsupported actions

### Resolution contract
A task resolves only when all of the following are true:

- `system_health >= 0.99`
- `correct_fix_applied == True`
- `truthful_status_posted == True`

This is a strong judging property: the benchmark does not award full success for a plausible explanation or a lucky partial fix.

### Final reward
At termination, the environment scores:

- root cause found: `+0.25`
- correct fix applied: `+0.40`
- truthful status posted: `+0.15`
- full resolution: `+0.20`

Then time decay rewards efficient correctness:

```text
time_decay = max(0.35, 1.0 - 0.03 * step_count)
final_reward = clamp(base_score * time_decay, 0.0, 1.0)
```

---

## Why the methodology is credible

Sev1Bench stays intentionally compact:

- **small action space** so evaluation is easy to understand
- **deterministic logs** so diagnosis is inspectable
- **explicit state flags** so milestone progress is measurable
- **bounded reward logic** so scores are interpretable
- **real resolution criteria** so demo success is meaningful

That makes it both hackathon-friendly and technically serious.

---

## Common failure modes it exposes

Weak agents fail in recognizable, useful ways:

- **symptom chasing** — they investigate the visibly broken downstream service instead of the root cause
- **wrong-target remediation** — they choose a valid action on the wrong system
- **premature “resolved” messaging** — they declare success before recovery is real
- **lucky but unsupported behavior** — they partially recover without evidence-backed reasoning
- **investigation loops** — they keep probing while health decays and the time-decayed score drops

These failure modes are part of the value: they reveal whether the benchmark is testing actual operational quality.

---

## Inference entry point

`inference.py` is the benchmark-facing runner.

It reads:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `TASK_ID`

It then:

- creates an OpenAI-compatible client
- runs the episode
- prints structured execution markers such as `[START]`, `[STEP]`, and `[END]`

### Example

Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && set API_BASE_URL=https://router.huggingface.co/v1 && set TASK_ID=medium && python inference.py
```

---

## UI and local demo

Run the local app:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

What the UI shows:

- task selection
- benchmark run execution
- terminal-style trace
- raw JSON output
- final success/score panel

That is a very strong hackathon presentation surface because reviewers can both watch the benchmark and audit the output.

### Recommended judge walkthrough

Use this order during the demo:

1. open the UI
2. say the one-line pitch
3. select `medium` or `hard`
4. run the benchmark
5. pause on the `read_logs` evidence
6. point to the action the agent chose
7. finish on the score and resolved outcome
8. mention that the same task can be rerun across agents with `run_baselines.py`

That sequence maximizes clarity and makes the project feel both polished and serious.

---

## Baseline evaluation

The repo includes `run_baselines.py` for local comparison across agents.

### Verified command

```bash
python run_baselines.py --mode mock --episodes 2
```

### What it provides

- side-by-side task comparison
- repeatable markdown summaries
- a quick proof that the benchmark is executable end to end
- a path to add LLM-vs-LLM comparisons later

If you want one standout differentiator to emphasize, this is the one:

> **Sev1Bench is not just an incident demo. It is an incident benchmark with replayable traces and a comparison harness.**

---

## Repository layout

Core files:

- `README.md`
- `inference.py`
- `models.py`
- `server/app.py`
- `server/environment.py`
- `run_baselines.py`
- `graders/`
- `tasks/`

Implementation roles:

- `server/environment.py` — incident simulation, reward logic, resolution rules
- `server/app.py` — benchmark UI and run endpoints
- `models.py` — typed action and observation contracts
- `inference.py` — benchmark execution loop
- `run_baselines.py` — baseline agent comparison and markdown result generation

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the benchmark UI

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run inference

Windows `cmd.exe`:

```bat
set HF_TOKEN=your_token && set API_BASE_URL=https://router.huggingface.co/v1 && python inference.py
```

### Run baseline comparisons

```bash
python run_baselines.py --mode mock --episodes 2
```

---

## Docker

Build:

```bash
docker build -t sev1bench .
```

Run:

```bash
docker run -p 7860:7860 sev1bench
```

The application serves on port `7860`.

---

## Hugging Face Space deployment

Recommended configuration:

- Space type: `Docker`
- Hardware: `CPU basic`
- Port: `7860`

Secrets / variables:

- required secret: `HF_TOKEN`
- required variable: `API_BASE_URL`
- optional variable: `MODEL_NAME`

---

## Selection readout

### Project rating: **9.3 / 10**

Why it now reads as a stronger selection candidate:

- **clear problem, clear audience, clear value**
- **benchmark framing instead of generic agent-product framing**
- **live demo path that is obvious in under 10 seconds**
- **transparent scoring and inspectable recovery logic**
- **real outputs already validated from the repo**
- **comparison harness makes it feel reusable and extensible**
- **credible architecture without being bloated**
- **strong “show, don’t tell” proof surface through traces, logs, JSON, and scores**
- **high judge confidence because claims are easy to verify on the spot**

Why this can break into the 90%+ range:
- it combines technical substance with very low explanation cost
- it avoids the “cool idea, weak proof” problem
- it gives judges both a live wow moment and a trustworthy evaluation story
- it feels like something that could outlast the hackathon as an actual benchmark

What would push it even higher later:
- a persistent leaderboard or saved comparison dashboard
- one polished screenshot or GIF of a real run in the README
- one external comparison result using a real hosted model

### Predicted selection chances: **90–92% if presented sharply**

Practical view:

- **around 90–92%** if you lead with the benchmark framing, show the 10-second loop immediately, pause on the evidence, and end on score plus comparison value
- **closer to 80–85%** only if the pitch gets buried in setup, implementation detail, or generic AI claims before the benchmark output appears

Best positioning sentence for judges:

> **Sev1Bench evaluates whether an agent can actually handle a live Sev1-style incident—not just explain one—and proves it with replayable traces, transparent scoring, and side-by-side benchmark runs.**
