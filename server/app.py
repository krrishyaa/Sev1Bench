from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server import create_fastapi_app
from openai import OpenAI

from models import IncidentAction, IncidentObservation
from inference import resolve_runtime_config, run_task
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

app = create_fastapi_app(IncidentResponseEnvironment, IncidentAction, IncidentObservation) if create_fastapi_app is not None else _fallback_app()


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
            --bg: #040814;
            --bg-2: #081221;
            --panel: rgba(8, 18, 33, 0.82);
            --panel-strong: rgba(10, 23, 42, 0.94);
            --panel-soft: rgba(255, 255, 255, 0.045);
            --border: rgba(255, 255, 255, 0.08);
            --text: #eef5ff;
            --muted: #9ab0cf;
            --accent: #ff7a18;
            --accent-2: #ffb703;
            --cyan: #67e8f9;
            --green: #4ade80;
            --blue: #60a5fa;
            --danger: #fb7185;
            --warning: #fbbf24;
            --shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
          }

          * { box-sizing: border-box; }
          html { scroll-behavior: smooth; }

          body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background:
              radial-gradient(circle at 0% 0%, rgba(255, 122, 24, 0.18), transparent 25%),
              radial-gradient(circle at 100% 10%, rgba(103, 232, 249, 0.12), transparent 22%),
              radial-gradient(circle at 50% 100%, rgba(96, 165, 250, 0.12), transparent 28%),
              linear-gradient(180deg, var(--bg) 0%, #02050d 100%);
          }

          body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
              linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 36px 36px;
            mask-image: linear-gradient(180deg, rgba(0,0,0,0.25), rgba(0,0,0,0.95));
            opacity: 0.18;
          }

          .shell {
            width: min(1280px, calc(100% - 30px));
            margin: 0 auto;
            padding: 24px 0 60px;
            position: relative;
            z-index: 1;
          }

          .hero {
            position: relative;
            overflow: hidden;
            border-radius: 32px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.08);
            background:
              radial-gradient(circle at 15% 18%, rgba(255, 183, 3, 0.16), transparent 18%),
              radial-gradient(circle at 85% 16%, rgba(103, 232, 249, 0.10), transparent 18%),
              radial-gradient(circle at 50% 0%, rgba(96, 165, 250, 0.12), transparent 26%),
              linear-gradient(135deg, rgba(255, 122, 24, 0.10), rgba(10, 22, 42, 0.96) 35%, rgba(4, 9, 20, 0.98));
            box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.05);
          }

          .hero::after {
            content: "";
            position: absolute;
            inset: auto -80px -80px auto;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 122, 24, 0.18), transparent 65%);
            filter: blur(8px);
          }

          .topbar, .hero-grid, .section, .callout-grid, .metrics-grid, .two-col {
            display: grid;
            gap: 18px;
          }

          .topbar {
            grid-template-columns: 1fr auto;
            align-items: center;
          }

          .badge, .status-chip, .task-chip, .pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            border-radius: 999px;
            padding: 10px 14px;
            font-size: 13px;
          }

          .badge {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--muted);
          }

          .status-chip {
            background: rgba(10, 26, 47, 0.82);
            border: 1px solid rgba(103, 232, 249, 0.28);
          }

          .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 16px rgba(74, 222, 128, 0.75);
          }

          h1 {
            margin: 20px 0 12px;
            max-width: 820px;
            font-size: clamp(42px, 8vw, 82px);
            line-height: 0.93;
            letter-spacing: -0.06em;
          }

          .lead {
            margin: 0;
            max-width: 820px;
            font-size: 18px;
            line-height: 1.8;
            color: var(--muted);
          }

          .lead strong, .metric strong, .highlight strong, .mini-card strong, .task-chip strong { color: var(--text); }

          .hero-highlight, .metrics-grid, .actions {
            display: grid;
            gap: 12px;
          }

          .hero-highlight { margin-top: 22px; grid-template-columns: repeat(4, minmax(0, 1fr)); }
          .hero-grid { margin-top: 24px; grid-template-columns: 1.2fr 0.8fr; }
          .section, .callout-grid, .two-col { margin-top: 20px; grid-template-columns: 1fr 1fr; }
          .metrics-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
          .actions { margin-top: 18px; grid-template-columns: repeat(5, minmax(0, 1fr)); }

          .highlight, .mini-card, .task-chip {
            padding: 16px 18px;
            border-radius: 18px;
            background: var(--panel-soft);
            border: 1px solid rgba(255,255,255,0.08);
          }

          .highlight span, .mini-card span, .eyebrow {
            display: block;
            margin-bottom: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
          }

          .judge-banner, .card {
            border: 1px solid var(--border);
            border-radius: 24px;
          }

          .judge-banner {
            margin-top: 18px;
            padding: 16px 18px;
            background: linear-gradient(90deg, rgba(255, 122, 24, 0.12), rgba(96, 165, 250, 0.08));
            line-height: 1.7;
          }

          .card {
            background: linear-gradient(180deg, rgba(10, 22, 42, 0.88), rgba(8, 18, 33, 0.78));
            padding: 22px;
            backdrop-filter: blur(14px);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 40px rgba(0,0,0,0.18);
          }

          .card h2 {
            margin: 0 0 14px;
            font-size: 20px;
            letter-spacing: -0.02em;
          }

          .card p, .small-note, ul { color: var(--muted); line-height: 1.7; }
          .metric {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            padding: 11px 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            color: var(--muted);
          }

          .metric:last-child { border-bottom: none; padding-bottom: 0; }
          .pill {
            justify-content: center;
            padding: 14px 12px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(13, 32, 60, 0.96), rgba(9, 20, 38, 0.88));
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--text);
            font-weight: 600;
          }

          .task-chip {
            border-radius: 20px;
            display: block;
          }

          .task-chip.easy { border-color: rgba(74, 222, 128, 0.25); }
          .task-chip.medium { border-color: rgba(251, 191, 36, 0.25); }
          .task-chip.hard { border-color: rgba(251, 113, 133, 0.25); }

          code, pre {
            font-family: Consolas, monospace;
          }

          code { color: var(--cyan); }
          pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            color: #d7e8ff;
            font-size: 12px;
            line-height: 1.55;
          }

          ul {
            margin: 0;
            padding-left: 18px;
          }

          .control-row {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
          }

          select, button {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            background: rgba(255,255,255,0.06);
            color: var(--text);
            padding: 12px 14px;
            font: inherit;
          }

          button {
            cursor: pointer;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #06101c;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(255, 122, 24, 0.24);
          }

          button.secondary {
            background: rgba(255,255,255,0.06);
            color: var(--text);
            box-shadow: none;
          }

          .result-shell {
            margin-top: 16px;
            display: grid;
            gap: 14px;
          }

          .result-banner {
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
          }

          .result-banner.ok { border-color: rgba(74, 222, 128, 0.25); }
          .result-banner.fail { border-color: rgba(251, 113, 133, 0.25); }

          .result-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
          }

          .result-stat {
            padding: 14px;
            border-radius: 16px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
          }

          .result-stat span {
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }

          .result-stat strong {
            font-size: 18px;
            color: var(--text);
          }

          .panel-stack {
            display: grid;
            gap: 12px;
          }

          .log-panel {
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(1, 6, 16, 0.55);
            padding: 14px;
            min-height: 160px;
          }

          .footer {
            margin-top: 22px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
          }

          .link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 13px 17px;
            border-radius: 14px;
            text-decoration: none;
            color: #06101c;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            font-weight: 800;
          }

          .link.secondary {
            color: var(--text);
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--border);
          }

          .meta-note {
            margin-left: auto;
            font-size: 13px;
            color: var(--muted);
          }

          .hidden { display: none; }

          @media (max-width: 1040px) {
            .hero-highlight, .hero-grid, .section, .callout-grid, .metrics-grid, .actions, .result-grid, .two-col, .topbar {
              grid-template-columns: 1fr;
            }
            .metric { flex-direction: column; align-items: flex-start; }
            .meta-note { margin-left: 0; }
          }

          @media (max-width: 640px) {
            .shell { width: min(100% - 18px, 1280px); padding-top: 18px; }
            .hero { padding: 22px; border-radius: 24px; }
            h1 { font-size: clamp(38px, 18vw, 64px); }
            .lead { font-size: 16px; }
          }
        </style>
      </head>
      <body>
        <main class="shell">
          <section class="hero">
            <div class="topbar">
              <div class="badge">🚨 Sev1Bench · OpenEnv Incident Response Benchmark</div>
              <div class="status-chip"><span class="status-dot"></span> Hugging Face Space + LiteLLM-ready runtime</div>
            </div>

            <h1>Sev1Bench</h1>

            <p class="lead">
              <strong>Sev1Bench</strong> evaluates whether an agent can investigate deterministic evidence, identify the
              true failing service, apply the correct remediation, communicate truthfully, and restore production health.
              This interface now includes a built-in Hugging Face test runner so reviewers can trigger a live benchmark
              episode and inspect the values returned from the configured runtime.
            </p>

            <div class="hero-highlight">
              <div class="highlight"><span>Runtime</span><strong>Environment-driven LiteLLM / HF config</strong></div>
              <div class="highlight"><span>Tasks</span><strong>easy · medium · hard</strong></div>
              <div class="highlight"><span>Entrypoint</span><strong><code>inference.py</code></strong></div>
              <div class="highlight"><span>Goal</span><strong>Investigate → Fix → Communicate</strong></div>
            </div>

            <div class="judge-banner">
              Built as a polished incident-response benchmark surface for Hugging Face Spaces: premium presentation,
              live runtime validation, and a cleaner reviewer experience around tasks, metrics, and execution traces.
            </div>

            <div class="hero-grid">
              <div class="card">
                <h2>Live benchmark cockpit</h2>
                <div class="metric"><span>Experience</span><strong>Premium, reviewer-facing, HF-native UI</strong></div>
                <div class="metric"><span>Runtime telemetry</span><strong>Model, score, rewards, steps, request logs</strong></div>
                <div class="metric"><span>Execution mode</span><strong>Interactive task runs from the landing page</strong></div>
                <div class="metric"><span>Validation flow</span><strong>One-click benchmark run with raw output inspection</strong></div>
                <div class="metric"><span>Design goal</span><strong>Fast trust-building for judges and builders</strong></div>
              </div>

              <div class="card">
                <h2>Submission contract</h2>
                <div class="metric"><span>Environment app</span><strong><code>server/app.py</code></strong></div>
                <div class="metric"><span>Evaluation entrypoint</span><strong><code>inference.py</code></strong></div>
                <div class="metric"><span>Space slug</span><strong><code>Krrishya/Sev1Bench</code></strong></div>
                <div class="metric"><span>API docs</span><strong><code>/docs</code></strong></div>
                <div class="metric"><span>Live tester route</span><strong><code>/ui/test-run</code></strong></div>
              </div>
            </div>

            <div class="actions">
              <div class="pill">read_logs</div>
              <div class="pill">restart_service</div>
              <div class="pill">scale_up</div>
              <div class="pill">rollback</div>
              <div class="pill">post_status_update</div>
            </div>

            <div class="section">
              <div class="card">
                <h2>Scenario matrix</h2>
                <div class="metrics-grid">
                  <div class="task-chip easy"><span>easy</span><strong>api-service · rollback</strong></div>
                  <div class="task-chip medium"><span>medium</span><strong>auth-service · restart</strong></div>
                  <div class="task-chip hard"><span>hard</span><strong>db-cluster · scale up</strong></div>
                  <div class="task-chip"><span>Success condition</span><strong>health ≥ 0.99 + truthful update</strong></div>
                </div>
              </div>

              <div class="card">
                <h2>Reviewer workflow</h2>
                <ul>
                  <li>Inspect the live OpenEnv endpoints in <code>/docs</code>.</li>
                  <li>Run a browser-side test below against the same runtime configuration.</li>
                  <li>Confirm the response shows the model, proxy base URL, step logs, and final score.</li>
                  <li>Verify the repo still uses the canonical programmatic evaluation flow.</li>
                </ul>
              </div>
            </div>

            <div class="callout-grid">
              <div class="card">
                <h2>Run a live Hugging Face test</h2>
                <p>
                  Launch a benchmark episode directly from the landing page and inspect a richer result surface with
                  run status, execution metrics, rewards, request traces, and raw backend output in one place.
                </p>

                <div class="control-row" style="margin-top:14px;">
                  <label for="task-select" class="small-note">Task</label>
                  <select id="task-select">
                    <option value="easy">easy</option>
                    <option value="medium">medium</option>
                    <option value="hard">hard</option>
                  </select>
                  <button id="run-test-btn" type="button">Run test on Hugging Face</button>
                  <button id="clear-test-btn" class="secondary" type="button">Clear</button>
                </div>

                <div id="result-shell" class="result-shell hidden">
                  <div id="result-banner" class="result-banner"></div>

                  <div class="result-grid">
                    <div class="result-stat"><span>Task</span><strong id="result-task">—</strong></div>
                    <div class="result-stat"><span>Success</span><strong id="result-success">—</strong></div>
                    <div class="result-stat"><span>Steps</span><strong id="result-steps">—</strong></div>
                    <div class="result-stat"><span>Score</span><strong id="result-score">—</strong></div>
                  </div>

                  <div class="two-col">
                    <div class="panel-stack">
                      <div class="card">
                        <span class="eyebrow">Runtime values</span>
                        <div class="metric"><span>Model</span><strong id="result-model">—</strong></div>
                        <div class="metric"><span>API base URL</span><strong id="result-base-url">—</strong></div>
                        <div class="metric"><span>Rewards</span><strong id="result-rewards">—</strong></div>
                      </div>
                      <div class="log-panel">
                        <span class="eyebrow">LLM request logs</span>
                        <pre id="result-llm-logs">No run yet.</pre>
                      </div>
                    </div>

                    <div class="panel-stack">
                      <div class="log-panel">
                        <span class="eyebrow">Step logs</span>
                        <pre id="result-step-logs">No run yet.</pre>
                      </div>
                      <div class="log-panel">
                        <span class="eyebrow">Raw response</span>
                        <pre id="result-json">No run yet.</pre>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="card">
                <h2>What this interface does better</h2>
                <ul>
                  <li>Presents Sev1Bench like a polished product instead of a basic landing page.</li>
                  <li>Lets reviewers trigger benchmark runs without leaving the Space homepage.</li>
                  <li>Surfaces the most important outputs immediately: success, score, steps, rewards, and logs.</li>
                  <li>Keeps the raw API contract available while making the visual review experience feel top tier.</li>
                </ul>
              </div>
            </div>

            <div class="footer">
              <a class="link" href="https://github.com/krrishyaa/Sev1Bench" target="_blank" rel="noreferrer">Open GitHub Repo</a>
              <a class="link secondary" href="/docs">Open API Docs</a>
              <div class="meta-note">Built for OpenEnv incident-response evaluation on Hugging Face Spaces.</div>
            </div>
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
            resultBanner.className = "result-banner";
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
            resultBanner.className = "result-banner";
            resultBanner.textContent = "Running live benchmark episode...";
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
                resultBanner.className = "result-banner fail";
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

              resultBanner.className = `result-banner ${data.success ? "ok" : "fail"}`;
              resultBanner.textContent = data.success
                ? "Run completed successfully through the configured runtime."
                : "Run completed but did not fully resolve the incident.";
              fields.task.textContent = data.task_id;
              fields.success.textContent = String(data.success);
              fields.steps.textContent = String(data.steps);
              fields.score.textContent = Number(data.score).toFixed(3);
              fields.model.textContent = data.model_name || "Unavailable";
              fields.baseUrl.textContent = data.api_base_url || "Unavailable";
              fields.rewards.textContent = Array.isArray(data.rewards) ? data.rewards.map((value) => Number(value).toFixed(2)).join(", ") : "[]";
              fields.llmLogs.textContent = data.llm_log_lines?.join("\\n") || "No LLM logs captured.";
              fields.stepLogs.textContent = data.step_log_lines?.join("\\n") || "No step logs captured.";
            } catch (error) {
              resultBanner.className = "result-banner fail";
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
