from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server import create_fastapi_app
from openai import OpenAI

from inference import SUPPORTED_TASK_IDS, resolve_runtime_config, run_task
from models import IncidentAction, IncidentObservation
from .environment import IncidentResponseEnvironment


TASK_DESCRIPTORS = {
    "easy": {
        "severity": "SEV-3",
        "root_cause": "api-service",
        "fix": "rollback",
        "summary": "Checkout latency regression after deploy with direct log evidence and tight rollback path.",
        "operator_focus": "Prioritize rapid evidence gathering, confirm deploy-induced misconfiguration, and communicate service degradation before closeout.",
        "grader_file": "graders/easy_grader.py",
        "yaml_file": "tasks/easy.yaml",
    },
    "medium": {
        "severity": "SEV-2",
        "root_cause": "auth-service",
        "fix": "restart_service",
        "summary": "Regional authentication instability with misleading downstream noise and identity retry storms.",
        "operator_focus": "Isolate signer-process failure from edge symptoms, restart the correct service, and keep status copy truthful during restoration.",
        "grader_file": "graders/medium_grader.py",
        "yaml_file": "tasks/medium.yaml",
    },
    "hard": {
        "severity": "SEV-1",
        "root_cause": "db-cluster",
        "fix": "scale_up",
        "summary": "Payment write path collapse driven by replication lag and overloaded persistence quorum.",
        "operator_focus": "Separate symptom-bearing services from the actual persistence bottleneck, then restore quorum before declaring service health.",
        "grader_file": "graders/hard_grader.py",
        "yaml_file": "tasks/hard.yaml",
    },
    "expert": {
        "severity": "SEV-1+",
        "root_cause": "queue-broker",
        "fix": "restart_service",
        "summary": "Fulfillment event transport outage where order orchestration and notification lag mask broker coordination loss.",
        "operator_focus": "Prove the transport-layer fault domain, recover broker coordination, and drive truthful status updates through backlog drain.",
        "grader_file": "graders/expert_grader.py",
        "yaml_file": "tasks/expert.yaml",
    },
}


def _fallback_app() -> FastAPI:
    app = FastAPI(title="Sev1Bench")

    env = IncidentResponseEnvironment()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/reset")
    def reset() -> dict:
        observation = env.reset()
        return observation.model_dump()

    @app.post("/step")
    def step(action: dict) -> dict:
        observation = env.step(IncidentAction(**action))
        return observation.model_dump()

    @app.get("/state")
    def state() -> dict:
        return env.state.model_dump()

    return app


print("API_BASE_URL=", os.getenv("API_BASE_URL"), file=sys.stderr, flush=True)
print("MODEL_NAME=", os.getenv("MODEL_NAME"), file=sys.stderr, flush=True)
print("HF_TOKEN set=", os.getenv("HF_TOKEN") is not None, file=sys.stderr, flush=True)

app = (
    create_fastapi_app(IncidentResponseEnvironment, IncidentAction, IncidentObservation)
    if create_fastapi_app is not None
    else _fallback_app()
)


@app.get("/ui/overview")
def ui_overview() -> JSONResponse:
    return JSONResponse(
        {
            "benchmark": "Sev1Bench",
            "entrypoint": "inference.py",
            "app_entrypoint": "server.app:app",
            "supported_tasks": list(SUPPORTED_TASK_IDS),
            "task_count": len(SUPPORTED_TASK_IDS),
            "active_graders": len(SUPPORTED_TASK_IDS),
            "reward_range": {"min": 0.0, "max": 1.0},
            "task_descriptors": TASK_DESCRIPTORS,
        }
    )


@app.get("/ui/test-run")
def ui_test_run(task_id: str = "easy") -> JSONResponse:
    selected_task = task_id if task_id in SUPPORTED_TASK_IDS else "easy"

    try:
        api_base_url, model_name = resolve_runtime_config()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            client = OpenAI(base_url=api_base_url, api_key=os.getenv("HF_TOKEN"))
            success, steps, score, rewards = run_task(
                task_id=selected_task,
                client=client,
                model_name=model_name,
            )

        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        llm_lines = [line for line in stdout_text.splitlines() if line.startswith("[LLM]")]
        step_lines = [line for line in stdout_text.splitlines() if line.startswith("[STEP]")]
        config_lines = [line for line in stdout_text.splitlines() if line.startswith("[CONFIG]")]
        summary = TASK_DESCRIPTORS[selected_task]

        return JSONResponse(
            {
                "ok": True,
                "task_id": selected_task,
                "model_name": model_name,
                "api_base_url": api_base_url,
                "success": success,
                "steps": steps,
                "score": score,
                "rewards": rewards,
                "llm_log_lines": llm_lines,
                "step_log_lines": step_lines,
                "config_log_lines": config_lines,
                "task_summary": summary,
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        )
    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "task_id": selected_task,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
            status_code=500,
        )


@app.get("/", response_class=HTMLResponse)
def landing_page() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sev1Bench Operations Console</title>
    <style>
      :root {
        --bg: #09111f;
        --bg-2: #0d1729;
        --panel: #101b2e;
        --panel-2: #13213a;
        --panel-3: #182844;
        --border: #243553;
        --border-strong: #35507c;
        --text: #ebf2ff;
        --muted: #98a9c7;
        --soft: #6f84a8;
        --accent: #83b0ff;
        --accent-2: #d4e3ff;
        --success: #32d296;
        --warn: #f0b429;
        --danger: #ff6b6b;
        --shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
        --radius: 16px;
        --mono: Consolas, "SFMono-Regular", Menlo, monospace;
        --sans: Inter, "Segoe UI", Arial, sans-serif;
      }

      * { box-sizing: border-box; }

      html, body {
        margin: 0;
        min-height: 100%;
        background:
          radial-gradient(circle at top right, rgba(131, 176, 255, 0.10), transparent 28%),
          linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
        color: var(--text);
        font-family: var(--sans);
      }

      body { padding: 20px; }

      .app {
        width: min(1600px, 100%);
        margin: 0 auto;
        display: grid;
        gap: 16px;
      }

      .card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
      }

      .topbar,
      .hero,
      .stats-grid,
      .main-grid,
      .task-grid,
      .ops-grid,
      .result-grid,
      .log-grid,
      .footer {
        display: grid;
        gap: 16px;
      }

      .topbar {
        grid-template-columns: 1.4fr 1fr auto;
        align-items: center;
        padding: 18px 22px;
      }

      .eyebrow,
      .label,
      .panel-label,
      .table-label {
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 11px;
        font-weight: 800;
        color: var(--soft);
      }

      .headline {
        margin: 8px 0 10px;
        font-size: 34px;
        line-height: 1.1;
        letter-spacing: -0.04em;
      }

      .subtext,
      .copy,
      .helper,
      .meta {
        color: var(--muted);
        line-height: 1.65;
      }

      .status-row,
      .pill-row,
      .control-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
      }

      .status-row { justify-content: flex-end; }

      .pill {
        min-height: 36px;
        padding: 0 12px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid var(--border);
        background: var(--panel-2);
        color: var(--muted);
        font-size: 13px;
        font-weight: 700;
      }

      .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--success);
      }

      .hero {
        grid-template-columns: 1.25fr 0.75fr;
        align-items: stretch;
      }

      .hero-main,
      .hero-side,
      .panel,
      .task-card,
      .result-card,
      .log-panel {
        padding: 20px;
      }

      .hero-main {
        background: linear-gradient(180deg, rgba(131, 176, 255, 0.10), transparent 42%), var(--panel);
      }

      .hero-side {
        background: var(--panel);
      }

      .hero-kpis {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 18px;
      }

      .mini-kpi {
        padding: 14px;
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 12px;
      }

      .mini-kpi strong {
        display: block;
        margin-top: 10px;
        color: var(--accent-2);
        font-size: 18px;
      }

      .stats-grid {
        grid-template-columns: repeat(5, minmax(0, 1fr));
      }

      .stat {
        padding: 18px;
      }

      .stat-value {
        margin-top: 10px;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.04em;
      }

      .main-grid {
        grid-template-columns: 1.25fr 0.75fr;
        align-items: start;
      }

      .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        gap: 16px;
        margin-bottom: 16px;
      }

      .panel-title {
        margin: 6px 0 0;
        font-size: 24px;
        letter-spacing: -0.03em;
      }

      .ops-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .op-box {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--panel-2);
      }

      .task-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .task-card {
        background: var(--panel-2);
        border: 1px solid var(--border);
      }

      .task-card.active {
        border-color: var(--border-strong);
        box-shadow: inset 0 0 0 1px rgba(131, 176, 255, 0.18);
      }

      .task-title {
        margin: 10px 0 12px;
        font-size: 20px;
      }

      .task-table {
        display: grid;
        gap: 8px;
        margin-top: 14px;
      }

      .task-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
        color: var(--muted);
      }

      .task-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
      }

      .task-row strong {
        color: var(--text);
        text-align: right;
      }

      button,
      select,
      .link-btn {
        font: inherit;
      }

      button,
      .link-btn {
        border: 1px solid var(--border-strong);
        background: var(--panel-2);
        color: var(--text);
        border-radius: 10px;
        height: 42px;
        padding: 0 14px;
        cursor: pointer;
        font-weight: 700;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }

      button.primary {
        background: var(--accent-2);
        color: #08111f;
        border-color: var(--accent-2);
      }

      select {
        min-width: 190px;
        height: 42px;
        padding: 0 12px;
        border-radius: 10px;
        border: 1px solid var(--border-strong);
        background: var(--panel-2);
        color: var(--text);
      }

      .banner {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--panel-2);
        font-weight: 800;
      }

      .banner.ok {
        color: #dcfff0;
        border-color: rgba(50, 210, 150, 0.5);
        background: rgba(50, 210, 150, 0.12);
      }

      .banner.fail {
        color: #ffe3e3;
        border-color: rgba(255, 107, 107, 0.45);
        background: rgba(255, 107, 107, 0.12);
      }

      .result-shell {
        display: grid;
        gap: 16px;
        margin-top: 18px;
      }

      .hidden { display: none; }

      .result-grid {
        grid-template-columns: repeat(5, minmax(0, 1fr));
      }

      .result-card {
        background: var(--panel-2);
        border: 1px solid var(--border);
      }

      .result-value {
        margin-top: 8px;
        font-size: 24px;
        font-weight: 800;
      }

      .log-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .log-panel {
        background: var(--panel-2);
        border: 1px solid var(--border);
      }

      pre {
        margin: 12px 0 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 12px;
        line-height: 1.6;
        font-family: var(--mono);
        color: #dce8ff;
      }

      .checklist,
      .insight-list {
        display: grid;
        gap: 12px;
      }

      .check-item,
      .insight {
        padding: 14px;
        border-radius: 12px;
        background: var(--panel-2);
        border: 1px solid var(--border);
      }

      .footer {
        grid-template-columns: 1fr auto auto;
        padding: 18px 22px;
        align-items: center;
      }

      @media (max-width: 1280px) {
        .topbar,
        .hero,
        .main-grid,
        .result-grid,
        .stats-grid,
        .footer {
          grid-template-columns: 1fr;
        }

        .status-row { justify-content: flex-start; }
      }

      @media (max-width: 960px) {
        .hero-kpis,
        .ops-grid,
        .task-grid,
        .log-grid,
        .stats-grid {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 760px) {
        body { padding: 12px; }
        .control-row { align-items: stretch; }
        button, select, .link-btn { width: 100%; }
      }
    </style>
  </head>
  <body>
    <main class="app">
      <section class="topbar card">
        <div>
          <div class="eyebrow">Meta hackathon submission console</div>
          <div class="headline">Sev1Bench validation and benchmark operations surface</div>
          <div class="subtext">
            Production-style incident-response benchmark with four explicitly declared tasks, four grader bindings,
            deterministic environment transitions, and validation-facing runtime telemetry exposed in one workspace.
          </div>
        </div>
        <div class="status-row">
          <div class="pill"><span class="dot"></span> Runtime healthy</div>
          <div class="pill">4 tasks / 4 graders registered</div>
          <div class="pill">FastAPI + OpenEnv + Docker</div>
        </div>
        <a class="link-btn" href="/docs">OpenAPI docs</a>
      </section>

      <section class="hero">
        <section class="card hero-main">
          <div class="eyebrow">Benchmark posture</div>
          <div class="headline" style="font-size: 28px; margin-top: 6px;">Validation-oriented execution model with evidence, remediation, communication, and recovery scoring</div>
          <div class="copy">
            The benchmark is intentionally structured to make grader visibility explicit: task YAML metadata, an importable
            central grader registry, runtime task mappings, validation-time file checks, and a live execution endpoint that
            exposes logs, scores, rewards, and raw JSON traces. This gives judges immediate visibility into how the environment,
            inference policy, and grader contract line up.
          </div>
          <div class="hero-kpis">
            <div class="mini-kpi">
              <div class="label">Registry contract</div>
              <strong>`graders/__init__.py` exports `GRADER_REGISTRY`</strong>
            </div>
            <div class="mini-kpi">
              <div class="label">Task declaration path</div>
              <strong>`openenv.yaml` + `tasks/*.yaml` define all active grader links</strong>
            </div>
            <div class="mini-kpi">
              <div class="label">Live review surface</div>
              <strong>Config logs, LLM logs, step logs, score, rewards, raw payload</strong>
            </div>
          </div>
        </section>

        <aside class="card hero-side">
          <div class="panel-label">Execution doctrine</div>
          <div class="checklist" style="margin-top: 14px;">
            <div class="check-item">
              <div class="label">1. Investigate precisely</div>
              <div class="helper">Use `read_logs` to separate fault domain from downstream symptom-bearing services.</div>
            </div>
            <div class="check-item">
              <div class="label">2. Apply the correct fix</div>
              <div class="helper">The environment rewards only the remediation action bound to the real root-cause service.</div>
            </div>
            <div class="check-item">
              <div class="label">3. Communicate truthfully</div>
              <div class="helper">Premature resolution claims are rejected; truthful degradation and recovery messaging is scored.</div>
            </div>
            <div class="check-item">
              <div class="label">4. Resolve completely</div>
              <div class="helper">Success requires restored health, validated remediation, and explicit incident closure signals.</div>
            </div>
          </div>
        </aside>
      </section>

      <section class="stats-grid">
        <div class="card stat">
          <div class="label">Active tasks</div>
          <div class="stat-value">4</div>
          <div class="helper">easy, medium, hard, expert</div>
        </div>
        <div class="card stat">
          <div class="label">Active graders</div>
          <div class="stat-value">4</div>
          <div class="helper">One deterministic grader per benchmark task</div>
        </div>
        <div class="card stat">
          <div class="label">Reward range</div>
          <div class="stat-value">0.0–1.0</div>
          <div class="helper">Reward values are bounded in validators and graders</div>
        </div>
        <div class="card stat">
          <div class="label">Execution contract</div>
          <div class="stat-value">Traceable</div>
          <div class="helper">Config logs, step logs, and raw payload preserved</div>
        </div>
        <div class="card stat">
          <div class="label">Submission objective</div>
          <div class="stat-value">Validation pass</div>
          <div class="helper">Designed to over-satisfy minimum grader-count requirements</div>
        </div>
      </section>

      <section class="main-grid">
        <section class="card panel">
          <div class="panel-header">
            <div>
              <div class="eyebrow">Task inventory</div>
              <div class="panel-title">Four-task benchmark matrix with explicit grader bindings</div>
            </div>
            <div class="pill-row">
              <div class="pill">Entrypoint: inference.py</div>
              <div class="pill">Validator: validate_submission.py</div>
            </div>
          </div>

          <div class="task-grid">
            <article class="task-card active">
              <div class="label">easy / SEV-3</div>
              <div class="task-title">API service out-of-memory incident</div>
              <div class="copy">Checkout degradation with direct signal clarity and rollback-driven recovery workflow.</div>
              <div class="task-table">
                <div class="task-row"><span>Root cause</span><strong>api-service</strong></div>
                <div class="task-row"><span>Expected remediation</span><strong>rollback</strong></div>
                <div class="task-row"><span>Task config</span><strong>tasks/easy.yaml</strong></div>
                <div class="task-row"><span>Grader file</span><strong>graders/easy_grader.py</strong></div>
              </div>
            </article>

            <article class="task-card active">
              <div class="label">medium / SEV-2</div>
              <div class="task-title">Auth service degradation with noisy downstream failures</div>
              <div class="copy">Identity-plane instability where edge 401s obscure the actual signer-process failure.</div>
              <div class="task-table">
                <div class="task-row"><span>Root cause</span><strong>auth-service</strong></div>
                <div class="task-row"><span>Expected remediation</span><strong>restart_service</strong></div>
                <div class="task-row"><span>Task config</span><strong>tasks/medium.yaml</strong></div>
                <div class="task-row"><span>Grader file</span><strong>graders/medium_grader.py</strong></div>
              </div>
            </article>

            <article class="task-card active">
              <div class="label">hard / SEV-1</div>
              <div class="task-title">Database incident under stakeholder pressure</div>
              <div class="copy">Persistence-tier bottleneck driving payment write failures and downstream settlement lag.</div>
              <div class="task-table">
                <div class="task-row"><span>Root cause</span><strong>db-cluster</strong></div>
                <div class="task-row"><span>Expected remediation</span><strong>scale_up</strong></div>
                <div class="task-row"><span>Task config</span><strong>tasks/hard.yaml</strong></div>
                <div class="task-row"><span>Grader file</span><strong>graders/hard_grader.py</strong></div>
              </div>
            </article>

            <article class="task-card active">
              <div class="label">expert / SEV-1+</div>
              <div class="task-title">Queue broker coordination failure during fulfillment event processing</div>
              <div class="copy">Transport-layer outage where orchestration and notification symptoms mask broker leadership loss.</div>
              <div class="task-table">
                <div class="task-row"><span>Root cause</span><strong>queue-broker</strong></div>
                <div class="task-row"><span>Expected remediation</span><strong>restart_service</strong></div>
                <div class="task-row"><span>Task config</span><strong>tasks/expert.yaml</strong></div>
                <div class="task-row"><span>Grader file</span><strong>graders/expert_grader.py</strong></div>
              </div>
            </article>
          </div>

          <div class="panel-header" style="margin-top: 22px;">
            <div>
              <div class="eyebrow">Live execution</div>
              <div class="panel-title">Run a task and inspect telemetry without leaving the page</div>
            </div>
          </div>

          <div class="control-row">
            <div class="label">Target task</div>
            <select id="task-select">
              <option value="easy">easy</option>
              <option value="medium">medium</option>
              <option value="hard">hard</option>
              <option value="expert">expert</option>
            </select>
            <button id="run-test-btn" class="primary" type="button">Execute benchmark run</button>
            <button id="clear-test-btn" type="button">Clear output</button>
          </div>

          <div id="result-shell" class="result-shell hidden">
            <div id="result-banner" class="banner"></div>

            <div class="result-grid">
              <div class="result-card">
                <div class="label">Task</div>
                <div id="result-task" class="result-value">—</div>
              </div>
              <div class="result-card">
                <div class="label">Success</div>
                <div id="result-success" class="result-value">—</div>
              </div>
              <div class="result-card">
                <div class="label">Steps</div>
                <div id="result-steps" class="result-value">—</div>
              </div>
              <div class="result-card">
                <div class="label">Score</div>
                <div id="result-score" class="result-value">—</div>
              </div>
              <div class="result-card">
                <div class="label">Rewards</div>
                <div id="result-rewards" class="result-value" style="font-size: 16px;">—</div>
              </div>
            </div>

            <div class="ops-grid">
              <div class="op-box">
                <div class="label">Selected task summary</div>
                <div id="result-summary" class="copy" style="margin-top: 10px;">No run yet.</div>
              </div>
              <div class="op-box">
                <div class="label">Execution context</div>
                <div class="task-table" style="margin-top: 10px;">
                  <div class="task-row"><span>Model</span><strong id="result-model">—</strong></div>
                  <div class="task-row"><span>API base URL</span><strong id="result-base-url">—</strong></div>
                  <div class="task-row"><span>Runtime config log</span><strong id="result-config-line">—</strong></div>
                </div>
              </div>
            </div>

            <div class="log-grid">
              <div class="log-panel">
                <div class="panel-label">LLM request log lines</div>
                <pre id="result-llm-logs">No run yet.</pre>
              </div>
              <div class="log-panel">
                <div class="panel-label">Environment step trace</div>
                <pre id="result-step-logs">No run yet.</pre>
              </div>
              <div class="log-panel">
                <div class="panel-label">Raw JSON response</div>
                <pre id="result-json">No run yet.</pre>
              </div>
              <div class="log-panel">
                <div class="panel-label">Captured stdout / stderr</div>
                <pre id="result-streams">No run yet.</pre>
              </div>
            </div>
          </div>
        </section>

        <aside class="card panel">
          <div class="panel-header">
            <div>
              <div class="eyebrow">Architecture and validation notes</div>
              <div class="panel-title">Why the grader count is visible to the pipeline</div>
            </div>
          </div>

          <div class="insight-list">
            <div class="insight">
              <div class="label">Registry import fix</div>
              <div class="helper">`graders/__init__.py` imports each grader module and exports `GRADER_REGISTRY`, giving the validator and runtime a central import surface instead of relying on incidental module discovery.</div>
            </div>
            <div class="insight">
              <div class="label">Task-name alignment</div>
              <div class="helper">`easy`, `medium`, `hard`, and `expert` now match across `server/environment.py`, `openenv.yaml`, `tasks/*.yaml`, `validate_submission.py`, `run_baselines.py`, and inference-time remediation routing.</div>
            </div>
            <div class="insight">
              <div class="label">Deterministic grading contract</div>
              <div class="helper">Each grader returns a bounded reward plus explicit flags for `root_cause_found`, `correct_fix_applied`, `truthful_status_posted`, and `resolved`, which simplifies automated pipeline verification.</div>
            </div>
            <div class="insight">
              <div class="label">Judge-facing runtime surface</div>
              <div class="helper">The dashboard exposes configuration logs, LLM request lines, step traces, task metadata, and raw response payloads so a reviewer can inspect both behavior and validation wiring from the same page.</div>
            </div>
            <div class="insight">
              <div class="label">Operational evaluation philosophy</div>
              <div class="helper">The benchmark is tuned for incident-management realism: identify the real fault domain, apply the right fix to the right target, communicate accurately under pressure, and restore system health completely before closure.</div>
            </div>
          </div>
        </aside>
      </section>

      <section class="card footer">
        <div class="meta">Dense review surface for benchmark judges, developer advocates, and incident-response platform engineers.</div>
        <a class="link-btn" href="/ui/overview">JSON overview</a>
        <a class="link-btn" href="/docs">API schema</a>
      </section>
    </main>

    <script>
      const runButton = document.getElementById("run-test-btn");
      const clearButton = document.getElementById("clear-test-btn");
      const taskSelect = document.getElementById("task-select");
      const resultShell = document.getElementById("result-shell");
      const resultBanner = document.getElementById("result-banner");

      const fields = {
        task: document.getElementById("result-task"),
        success: document.getElementById("result-success"),
        steps: document.getElementById("result-steps"),
        score: document.getElementById("result-score"),
        rewards: document.getElementById("result-rewards"),
        summary: document.getElementById("result-summary"),
        model: document.getElementById("result-model"),
        baseUrl: document.getElementById("result-base-url"),
        configLine: document.getElementById("result-config-line"),
        llmLogs: document.getElementById("result-llm-logs"),
        stepLogs: document.getElementById("result-step-logs"),
        json: document.getElementById("result-json"),
        streams: document.getElementById("result-streams")
      };

      function resetOutput() {
        resultShell.classList.add("hidden");
        resultBanner.className = "banner";
        resultBanner.textContent = "";
        fields.task.textContent = "—";
        fields.success.textContent = "—";
        fields.steps.textContent = "—";
        fields.score.textContent = "—";
        fields.rewards.textContent = "—";
        fields.summary.textContent = "No run yet.";
        fields.model.textContent = "—";
        fields.baseUrl.textContent = "—";
        fields.configLine.textContent = "—";
        fields.llmLogs.textContent = "No run yet.";
        fields.stepLogs.textContent = "No run yet.";
        fields.json.textContent = "No run yet.";
        fields.streams.textContent = "No run yet.";
      }

      async function runTest() {
        const task = taskSelect.value;
        resultShell.classList.remove("hidden");
        resultBanner.className = "banner";
        resultBanner.textContent = "Executing benchmark task and collecting runtime telemetry...";
        fields.task.textContent = task;
        fields.success.textContent = "Pending";
        fields.steps.textContent = "…";
        fields.score.textContent = "…";
        fields.rewards.textContent = "Loading…";
        fields.summary.textContent = "Loading task summary...";
        fields.model.textContent = "Loading…";
        fields.baseUrl.textContent = "Loading…";
        fields.configLine.textContent = "Loading…";
        fields.llmLogs.textContent = "Waiting for backend response...";
        fields.stepLogs.textContent = "Waiting for backend response...";
        fields.json.textContent = "Waiting for backend response...";
        fields.streams.textContent = "Waiting for backend response...";

        try {
          const response = await fetch(`/ui/test-run?task_id=${encodeURIComponent(task)}`);
          const data = await response.json();
          fields.json.textContent = JSON.stringify(data, null, 2);

          if (!response.ok || !data.ok) {
            resultBanner.className = "banner fail";
            resultBanner.textContent = `Run failed: ${data.error_type || "Error"} - ${data.error || "Unknown error"}`;
            fields.success.textContent = "false";
            fields.steps.textContent = "0";
            fields.score.textContent = "0.000";
            fields.rewards.textContent = "[]";
            fields.summary.textContent = "Task execution did not complete successfully.";
            fields.model.textContent = data.model_name || "Unavailable";
            fields.baseUrl.textContent = data.api_base_url || "Unavailable";
            fields.configLine.textContent = "Unavailable";
            fields.llmLogs.textContent = data.llm_log_lines?.join("\\n") || "No LLM logs captured.";
            fields.stepLogs.textContent = data.step_log_lines?.join("\\n") || "No step logs captured.";
            fields.streams.textContent = JSON.stringify({ stderr: data.error || "", stdout: "" }, null, 2);
            return;
          }

          const summary = data.task_summary || {};
          resultBanner.className = `banner ${data.success ? "ok" : "fail"}`;
          resultBanner.textContent = data.success
            ? "Benchmark run completed successfully with a fully resolved incident."
            : "Benchmark run completed, but the incident did not satisfy the full resolution contract.";
          fields.task.textContent = data.task_id;
          fields.success.textContent = String(data.success);
          fields.steps.textContent = String(data.steps);
          fields.score.textContent = Number(data.score).toFixed(3);
          fields.rewards.textContent = Array.isArray(data.rewards)
            ? data.rewards.map((value) => Number(value).toFixed(2)).join(", ")
            : "[]";
          fields.summary.textContent =
            `${summary.summary || "Summary unavailable."} Root cause: ${summary.root_cause || "n/a"}. ` +
            `Expected fix: ${summary.fix || "n/a"}. Operator focus: ${summary.operator_focus || "n/a"}`;
          fields.model.textContent = data.model_name || "Unavailable";
          fields.baseUrl.textContent = data.api_base_url || "Unavailable";
          fields.configLine.textContent = data.config_log_lines?.[0] || "No config log captured.";
          fields.llmLogs.textContent = data.llm_log_lines?.join("\\n") || "No LLM logs captured.";
          fields.stepLogs.textContent = data.step_log_lines?.join("\\n") || "No step logs captured.";
          fields.streams.textContent = `STDOUT:\\n${data.stdout || ""}\\n\\nSTDERR:\\n${data.stderr || ""}`;
        } catch (error) {
          resultBanner.className = "banner fail";
          resultBanner.textContent = `Run failed: ${error}`;
          fields.success.textContent = "false";
          fields.steps.textContent = "0";
          fields.score.textContent = "0.000";
          fields.rewards.textContent = "[]";
          fields.summary.textContent = "Task execution did not complete successfully.";
          fields.model.textContent = "Unavailable";
          fields.baseUrl.textContent = "Unavailable";
          fields.configLine.textContent = "Unavailable";
          fields.llmLogs.textContent = "No LLM logs captured.";
          fields.stepLogs.textContent = "No step logs captured.";
          fields.json.textContent = JSON.stringify({ error: String(error) }, null, 2);
          fields.streams.textContent = JSON.stringify({ error: String(error) }, null, 2);
        }
      }

      runButton.addEventListener("click", runTest);
      clearButton.addEventListener("click", resetOutput);
      resetOutput();
    </script>
  </body>
</html>
    """


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
