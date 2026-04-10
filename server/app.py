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

from inference import resolve_runtime_config, run_task
from models import IncidentAction, IncidentObservation
from .environment import IncidentResponseEnvironment


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


@app.get("/ui/test-run")
def ui_test_run(task_id: str = "easy") -> JSONResponse:
    selected_task = task_id if task_id in {"easy", "medium", "hard"} else "easy"

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
        <title>Sev1Bench</title>
        <style>
          :root {
            color-scheme: dark;
            --bg: #0b1020;
            --bg-alt: #11182d;
            --surface: #141c31;
            --surface-2: #1a243d;
            --surface-3: #202b46;
            --border: #2a3553;
            --border-strong: #3a4768;
            --text: #ebf1ff;
            --text-muted: #a7b4d1;
            --text-soft: #7f8cab;
            --success: #19c37d;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #60a5fa;
            --accent: #8fb4ff;
            --accent-strong: #c8d7ff;
            --shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
            --radius: 14px;
            --radius-sm: 10px;
            --mono: Consolas, "SFMono-Regular", Menlo, monospace;
            --sans: Inter, "Segoe UI", Arial, sans-serif;
          }

          * { box-sizing: border-box; }

          html, body {
            margin: 0;
            min-height: 100%;
            background: linear-gradient(180deg, var(--bg) 0%, #0f1728 100%);
            color: var(--text);
            font-family: var(--sans);
          }

          body {
            padding: 24px;
          }

          .app {
            width: min(1480px, 100%);
            margin: 0 auto;
            display: grid;
            gap: 16px;
          }

          .topbar,
          .summary-grid,
          .content-grid,
          .task-grid,
          .metrics-grid,
          .result-grid,
          .logs-grid,
          .footer-grid {
            display: grid;
            gap: 16px;
          }

          .topbar {
            grid-template-columns: 1.4fr 1fr auto;
            align-items: center;
            padding: 16px 20px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
          }

          .brand h1 {
            margin: 4px 0 0;
            font-size: 28px;
            letter-spacing: -0.03em;
          }

          .eyebrow,
          .label,
          .kpi-label,
          .panel-label,
          .table-head {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-soft);
          }

          .subtext,
          .body-copy,
          .meta,
          .task-copy,
          .helper {
            color: var(--text-muted);
            line-height: 1.6;
          }

          .status-cluster {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: flex-end;
          }

          .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 36px;
            padding: 0 12px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--surface-2);
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 600;
          }

          .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
          }

          .link-btn,
          button,
          select {
            font: inherit;
          }

          .link-btn,
          button {
            height: 40px;
            padding: 0 14px;
            border-radius: 10px;
            border: 1px solid var(--border-strong);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            cursor: pointer;
            transition: background 140ms ease, border-color 140ms ease, color 140ms ease;
          }

          .primary {
            background: #dbe7ff;
            color: #0b1020;
            border-color: #dbe7ff;
          }

          .primary:hover {
            background: #c8d7ff;
            border-color: #c8d7ff;
          }

          .secondary {
            background: var(--surface-2);
            color: var(--text);
          }

          .secondary:hover,
          select:hover {
            border-color: var(--accent);
          }

          .summary-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
          }

          .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
          }

          .summary-card,
          .panel,
          .task-card,
          .result-card,
          .log-panel {
            padding: 18px;
          }

          .kpi-value {
            margin-top: 10px;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.03em;
          }

          .content-grid {
            grid-template-columns: 1.25fr 0.75fr;
            align-items: start;
          }

          .panel-header {
            display: flex;
            align-items: start;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 16px;
          }

          .panel-title {
            margin: 4px 0 0;
            font-size: 22px;
            letter-spacing: -0.03em;
          }

          .metrics-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-bottom: 16px;
          }

          .metric {
            padding: 14px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 12px;
          }

          .metric strong {
            display: block;
            margin-top: 8px;
            font-size: 17px;
            color: var(--accent-strong);
          }

          .task-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }

          .task-card {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 12px;
          }

          .task-card.active-grader {
            border-color: #35508a;
            box-shadow: inset 0 0 0 1px rgba(143, 180, 255, 0.16);
          }

          .task-title {
            margin: 10px 0 8px;
            font-size: 18px;
          }

          .task-meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 14px;
          }

          .task-tag {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0 10px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--surface-3);
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 700;
          }

          .control-row {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
            margin: 18px 0 0;
            padding-top: 16px;
            border-top: 1px solid var(--border);
          }

          select {
            min-width: 180px;
            height: 40px;
            padding: 0 12px;
            color: var(--text);
            background: var(--surface-2);
            border: 1px solid var(--border-strong);
            border-radius: 10px;
            outline: none;
          }

          .result-shell {
            margin-top: 18px;
            display: grid;
            gap: 16px;
          }

          .hidden { display: none; }

          .banner {
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: var(--surface-2);
            font-weight: 700;
          }

          .banner.ok {
            color: #d9ffef;
            background: rgba(25, 195, 125, 0.12);
            border-color: rgba(25, 195, 125, 0.45);
          }

          .banner.fail {
            color: #ffe1e1;
            background: rgba(239, 68, 68, 0.12);
            border-color: rgba(239, 68, 68, 0.40);
          }

          .result-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
          }

          .result-value {
            margin-top: 8px;
            font-size: 24px;
            font-weight: 800;
          }

          .logs-grid {
            grid-template-columns: 1fr 1fr;
          }

          .key-value {
            display: grid;
            gap: 10px;
          }

          .kv-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
            color: var(--text-muted);
          }

          .kv-row:last-child {
            padding-bottom: 0;
            border-bottom: none;
          }

          .kv-row strong {
            color: var(--text);
            text-align: right;
          }

          pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            color: #d7e2ff;
            font-family: var(--mono);
            font-size: 12px;
            line-height: 1.6;
          }

          .side-list {
            display: grid;
            gap: 12px;
          }

          .side-item {
            padding: 14px;
            border-radius: 12px;
            background: var(--surface-2);
            border: 1px solid var(--border);
          }

          .footer-grid {
            grid-template-columns: 1fr auto auto;
            align-items: center;
            padding: 16px 20px;
          }

          @media (max-width: 1200px) {
            .summary-grid,
            .metrics-grid,
            .task-grid,
            .result-grid,
            .logs-grid,
            .footer-grid {
              grid-template-columns: 1fr 1fr;
            }

            .content-grid,
            .topbar {
              grid-template-columns: 1fr;
            }

            .status-cluster {
              justify-content: flex-start;
            }
          }

          @media (max-width: 760px) {
            body {
              padding: 12px;
            }

            .summary-grid,
            .metrics-grid,
            .task-grid,
            .result-grid,
            .logs-grid,
            .footer-grid {
              grid-template-columns: 1fr;
            }

            .control-row {
              align-items: stretch;
            }

            select,
            button,
            .link-btn {
              width: 100%;
            }
          }
        </style>
      </head>
      <body>
        <main class="app">
          <section class="topbar card">
            <div class="brand">
              <div class="eyebrow">Production benchmark console</div>
              <h1>Sev1Bench</h1>
              <div class="subtext">Deterministic incident-response evaluation for OpenEnv submission review.</div>
            </div>
            <div class="status-cluster">
              <div class="pill"><span class="dot"></span> Space runtime online</div>
              <div class="pill">3 tasks with active graders</div>
              <div class="pill">Entrypoint: inference.py</div>
            </div>
            <a class="link-btn secondary" href="/docs">API docs</a>
          </section>

          <section class="summary-grid">
            <div class="card summary-card">
              <div class="kpi-label">Validation target</div>
              <div class="kpi-value">≥ 3 graders</div>
              <div class="helper">All benchmark tasks declare active grader bindings.</div>
            </div>
            <div class="card summary-card">
              <div class="kpi-label">Execution model</div>
              <div class="kpi-value">Investigate → Remediate → Communicate</div>
              <div class="helper">Deterministic environment dynamics with bounded reward output.</div>
            </div>
            <div class="card summary-card">
              <div class="kpi-label">Judge visibility</div>
              <div class="kpi-value">Score + trace + raw payload</div>
              <div class="helper">Core review signals are visible from the main dashboard without navigation.</div>
            </div>
            <div class="card summary-card">
              <div class="kpi-label">Deployment contract</div>
              <div class="kpi-value">Docker / FastAPI / OpenEnv</div>
              <div class="helper">Aligned with the benchmark runtime and validation interface.</div>
            </div>
          </section>

          <section class="content-grid">
            <section class="card panel">
              <div class="panel-header">
                <div>
                  <div class="eyebrow">Primary dashboard</div>
                  <h2 class="panel-title">Task inventory and live execution</h2>
                </div>
                <div class="pill">Main judge workflow</div>
              </div>

              <div class="metrics-grid">
                <div class="metric">
                  <div class="label">Main tasks</div>
                  <strong>easy / medium / hard</strong>
                </div>
                <div class="metric">
                  <div class="label">Custom graders</div>
                  <strong>3 deterministic grade() functions</strong>
                </div>
                <div class="metric">
                  <div class="label">Success condition</div>
                  <strong>Root cause + correct fix + truthful status + resolution</strong>
                </div>
              </div>

              <div class="task-grid">
                <article class="task-card active-grader">
                  <div class="label">easy</div>
                  <h3 class="task-title">API service out-of-memory incident</h3>
                  <div class="task-copy">Root cause: <strong>api-service</strong>. Expected fix: <strong>rollback</strong>.</div>
                  <div class="task-meta">
                    <span class="task-tag">grader active</span>
                    <span class="task-tag">reward 0.0–1.0</span>
                    <span class="task-tag">deterministic</span>
                  </div>
                </article>
                <article class="task-card active-grader">
                  <div class="label">medium</div>
                  <h3 class="task-title">Auth service degradation with noisy downstream failures</h3>
                  <div class="task-copy">Root cause: <strong>auth-service</strong>. Expected fix: <strong>restart_service</strong>.</div>
                  <div class="task-meta">
                    <span class="task-tag">grader active</span>
                    <span class="task-tag">reward 0.0–1.0</span>
                    <span class="task-tag">deterministic</span>
                  </div>
                </article>
                <article class="task-card active-grader">
                  <div class="label">hard</div>
                  <h3 class="task-title">Database incident under stakeholder pressure</h3>
                  <div class="task-copy">Root cause: <strong>db-cluster</strong>. Expected fix: <strong>scale_up</strong>.</div>
                  <div class="task-meta">
                    <span class="task-tag">grader active</span>
                    <span class="task-tag">reward 0.0–1.0</span>
                    <span class="task-tag">deterministic</span>
                  </div>
                </article>
              </div>

              <div class="control-row">
                <div class="label">Run benchmark task</div>
                <select id="task-select">
                  <option value="easy">easy</option>
                  <option value="medium">medium</option>
                  <option value="hard">hard</option>
                </select>
                <button id="run-test-btn" class="primary" type="button">Execute live run</button>
                <button id="clear-test-btn" class="secondary" type="button">Reset output</button>
              </div>

              <div id="result-shell" class="result-shell hidden">
                <div id="result-banner" class="banner"></div>

                <div class="result-grid">
                  <div class="card result-card">
                    <div class="kpi-label">Task</div>
                    <div id="result-task" class="result-value">—</div>
                  </div>
                  <div class="card result-card">
                    <div class="kpi-label">Success</div>
                    <div id="result-success" class="result-value">—</div>
                  </div>
                  <div class="card result-card">
                    <div class="kpi-label">Steps</div>
                    <div id="result-steps" class="result-value">—</div>
                  </div>
                  <div class="card result-card">
                    <div class="kpi-label">Score</div>
                    <div id="result-score" class="result-value">—</div>
                  </div>
                </div>

                <div class="logs-grid">
                  <div class="card log-panel">
                    <div class="panel-label">Runtime values</div>
                    <div class="key-value" style="margin-top: 12px;">
                      <div class="kv-row"><span>Model</span><strong id="result-model">—</strong></div>
                      <div class="kv-row"><span>API base URL</span><strong id="result-base-url">—</strong></div>
                      <div class="kv-row"><span>Rewards</span><strong id="result-rewards">—</strong></div>
                    </div>
                  </div>
                  <div class="card log-panel">
                    <div class="panel-label">LLM request logs</div>
                    <pre id="result-llm-logs" style="margin-top: 12px;">No run yet.</pre>
                  </div>
                  <div class="card log-panel">
                    <div class="panel-label">Step logs</div>
                    <pre id="result-step-logs" style="margin-top: 12px;">No run yet.</pre>
                  </div>
                  <div class="card log-panel">
                    <div class="panel-label">Raw response</div>
                    <pre id="result-json" style="margin-top: 12px;">No run yet.</pre>
                  </div>
                </div>
              </div>
            </section>

            <aside class="card panel">
              <div class="panel-header">
                <div>
                  <div class="eyebrow">Submission control plane</div>
                  <h2 class="panel-title">Validation-facing details</h2>
                </div>
              </div>

              <div class="side-list">
                <div class="side-item">
                  <div class="label">Entrypoints</div>
                  <div class="body-copy">App: <strong>server.app:app</strong><br />Inference: <strong>inference.py</strong></div>
                </div>
                <div class="side-item">
                  <div class="label">Grader contract</div>
                  <div class="body-copy">Each task declares <strong>grader</strong>, <strong>grader_fn</strong>, and <strong>active: true</strong>.</div>
                </div>
                <div class="side-item">
                  <div class="label">Reward boundaries</div>
                  <div class="body-copy">All graders clamp reward output to the inclusive range <strong>[0.0, 1.0]</strong>.</div>
                </div>
                <div class="side-item">
                  <div class="label">Review surface</div>
                  <div class="body-copy">Main dashboard prioritizes validation metrics, active graded tasks, runtime values, trace logs, and raw execution payload.</div>
                </div>
              </div>
            </aside>
          </section>

          <section class="card footer-grid">
            <div class="meta">Enterprise review surface optimized for fast benchmark inspection and deterministic validation.</div>
            <a class="link-btn secondary" href="https://github.com/krrishyaa/Sev1Bench" target="_blank" rel="noreferrer">GitHub repository</a>
            <a class="link-btn secondary" href="/docs">Open API schema</a>
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
            model: document.getElementById("result-model"),
            baseUrl: document.getElementById("result-base-url"),
            rewards: document.getElementById("result-rewards"),
            llmLogs: document.getElementById("result-llm-logs"),
            stepLogs: document.getElementById("result-step-logs"),
            json: document.getElementById("result-json")
          };

          function resetOutput() {
            resultShell.classList.add("hidden");
            resultBanner.className = "banner";
            resultBanner.textContent = "";
            fields.task.textContent = "—";
            fields.success.textContent = "—";
            fields.steps.textContent = "—";
            fields.score.textContent = "—";
            fields.model.textContent = "—";
            fields.baseUrl.textContent = "—";
            fields.rewards.textContent = "—";
            fields.llmLogs.textContent = "No run yet.";
            fields.stepLogs.textContent = "No run yet.";
            fields.json.textContent = "No run yet.";
          }

          async function runTest() {
            const task = taskSelect.value;
            resultShell.classList.remove("hidden");
            resultBanner.className = "banner";
            resultBanner.textContent = "Executing live benchmark run...";
            fields.task.textContent = task;
            fields.success.textContent = "Pending";
            fields.steps.textContent = "…";
            fields.score.textContent = "…";
            fields.model.textContent = "Loading…";
            fields.baseUrl.textContent = "Loading…";
            fields.rewards.textContent = "Loading…";
            fields.llmLogs.textContent = "Waiting for backend response...";
            fields.stepLogs.textContent = "Waiting for backend response...";
            fields.json.textContent = "Waiting for backend response...";

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
                fields.model.textContent = data.model_name || "Unavailable";
                fields.baseUrl.textContent = data.api_base_url || "Unavailable";
                fields.rewards.textContent = "[]";
                fields.llmLogs.textContent = data.llm_log_lines?.join("\\n") || "No LLM logs captured.";
                fields.stepLogs.textContent = data.step_log_lines?.join("\\n") || "No step logs captured.";
                return;
              }

              resultBanner.className = `banner ${data.success ? "ok" : "fail"}`;
              resultBanner.textContent = data.success
                ? "Benchmark run completed successfully."
                : "Benchmark run completed but the incident did not reach full resolution.";
              fields.task.textContent = data.task_id;
              fields.success.textContent = String(data.success);
              fields.steps.textContent = String(data.steps);
              fields.score.textContent = Number(data.score).toFixed(3);
              fields.model.textContent = data.model_name || "Unavailable";
              fields.baseUrl.textContent = data.api_base_url || "Unavailable";
              fields.rewards.textContent = Array.isArray(data.rewards)
                ? data.rewards.map((value) => Number(value).toFixed(2)).join(", ")
                : "[]";
              fields.llmLogs.textContent = data.llm_log_lines?.join("\\n") || "No LLM logs captured.";
              fields.stepLogs.textContent = data.step_log_lines?.join("\\n") || "No step logs captured.";
            } catch (error) {
              resultBanner.className = "banner fail";
              resultBanner.textContent = `Run failed: ${error}`;
              fields.success.textContent = "false";
              fields.steps.textContent = "0";
              fields.score.textContent = "0.000";
              fields.model.textContent = "Unavailable";
              fields.baseUrl.textContent = "Unavailable";
              fields.rewards.textContent = "[]";
              fields.llmLogs.textContent = "No LLM logs captured.";
              fields.stepLogs.textContent = "No step logs captured.";
              fields.json.textContent = JSON.stringify({ error: String(error) }, null, 2);
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
