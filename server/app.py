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
            color-scheme: light;
            --bg: #f6f2ea;
            --bg-soft: #fbf8f3;
            --surface: rgba(255, 255, 255, 0.72);
            --surface-strong: rgba(255, 255, 255, 0.88);
            --surface-solid: #ffffff;
            --border: rgba(122, 100, 75, 0.12);
            --border-strong: rgba(122, 100, 75, 0.18);
            --text: #1f2937;
            --text-soft: #5f6b7a;
            --title: #182230;
            --accent: #d97757;
            --accent-strong: #bc5f3f;
            --mint: #d7efe6;
            --blue-soft: #e6eefb;
            --amber-soft: #f6ead6;
            --rose-soft: #f8e6e6;
            --shadow: 0 30px 80px rgba(84, 62, 42, 0.10);
            --glass-shadow: 0 18px 45px rgba(78, 56, 36, 0.08);
            --radius-xl: 30px;
            --radius-lg: 22px;
            --radius-md: 18px;
          }

          * { box-sizing: border-box; }
          html { scroll-behavior: smooth; }

          body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            color: var(--text);
            background:
              radial-gradient(circle at top left, rgba(240, 212, 191, 0.75), transparent 24%),
              radial-gradient(circle at top right, rgba(220, 235, 246, 0.85), transparent 22%),
              linear-gradient(180deg, #fcfaf6 0%, var(--bg) 46%, #f3ede2 100%);
          }

          body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
              linear-gradient(rgba(255,255,255,0.28), rgba(255,255,255,0.28)),
              linear-gradient(120deg, rgba(255,255,255,0.35), transparent 36%);
            opacity: 0.6;
          }

          .page {
            position: relative;
            z-index: 1;
            width: min(1240px, calc(100% - 36px));
            margin: 0 auto;
            padding: 28px 0 64px;
          }

          .hero {
            border-radius: var(--radius-xl);
            background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(255,255,255,0.68));
            border: 1px solid rgba(255,255,255,0.7);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            overflow: hidden;
          }

          .hero-inner {
            padding: 28px;
          }

          .topbar,
          .hero-main,
          .signal-row,
          .priority-grid,
          .detail-grid,
          .runner-grid,
          .result-grid,
          .insight-grid,
          .footer-row {
            display: grid;
            gap: 18px;
          }

          .topbar {
            grid-template-columns: 1fr auto;
            align-items: center;
          }

          .hero-main {
            grid-template-columns: 1.1fr 0.9fr;
            margin-top: 18px;
            align-items: start;
          }

          .signal-row,
          .priority-grid,
          .detail-grid,
          .insight-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
          }

          .runner-grid {
            grid-template-columns: 1.2fr 0.8fr;
            margin-top: 24px;
          }

          .result-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
          }

          .footer-row {
            grid-template-columns: auto auto 1fr;
            align-items: center;
            margin-top: 26px;
          }

          .eyebrow,
          .mini-label,
          .stat-label,
          .panel-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 11px;
            font-weight: 700;
            color: #7a6c5d;
          }

          .brand-pill,
          .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(122, 100, 75, 0.12);
            box-shadow: 0 8px 24px rgba(78, 56, 36, 0.06);
            font-size: 13px;
            color: var(--text-soft);
          }

          .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #67b58d;
            box-shadow: 0 0 0 6px rgba(103, 181, 141, 0.14);
          }

          h1 {
            margin: 14px 0 14px;
            font-size: clamp(44px, 7vw, 78px);
            line-height: 0.95;
            letter-spacing: -0.06em;
            color: var(--title);
          }

          .lead {
            margin: 0;
            max-width: 720px;
            font-size: 18px;
            line-height: 1.8;
            color: var(--text-soft);
          }

          .lead strong,
          .stat-value,
          .feature-card strong,
          .glass-card strong,
          .scenario-card strong {
            color: var(--title);
          }

          .hero-copy {
            padding-right: 8px;
          }

          .signal-card,
          .glass-card,
          .feature-card,
          .scenario-card,
          .result-card,
          .log-card {
            border-radius: var(--radius-lg);
            background: var(--surface);
            border: 1px solid rgba(255,255,255,0.78);
            box-shadow: var(--glass-shadow);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
          }

          .signal-card,
          .feature-card,
          .scenario-card,
          .result-card {
            padding: 18px;
          }

          .glass-card {
            padding: 22px;
          }

          .signal-card {
            min-height: 106px;
          }

          .signal-value {
            display: block;
            margin-top: 10px;
            font-size: 15px;
            line-height: 1.5;
            font-weight: 700;
          }

          .feature-card p,
          .glass-card p,
          .small-copy,
          ul {
            margin: 0;
            color: var(--text-soft);
            line-height: 1.7;
          }

          .hero-panel {
            display: grid;
            gap: 16px;
          }

          .hero-panel .glass-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.66));
          }

          .section-title {
            margin: 0 0 10px;
            font-size: 22px;
            letter-spacing: -0.03em;
            color: var(--title);
          }

          .metric-list {
            display: grid;
            gap: 12px;
          }

          .metric-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 14px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text-soft);
          }

          .metric-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
          }

          .metric-row strong {
            text-align: right;
          }

          .priority-grid {
            margin-top: 22px;
          }

          .feature-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,255,255,0.68));
          }

          .feature-card h3 {
            margin: 10px 0 8px;
            font-size: 18px;
            letter-spacing: -0.02em;
            color: var(--title);
          }

          .runner-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,255,255,0.72));
          }

          .runner-hero {
            display: grid;
            gap: 12px;
            margin-bottom: 18px;
          }

          .runner-title {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
          }

          .runner-title h2 {
            margin: 0;
            font-size: 28px;
            letter-spacing: -0.04em;
            color: var(--title);
          }

          .runner-highlight {
            color: var(--accent-strong);
            font-weight: 700;
          }

          .workflow-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(255,255,255,0.70));
          }

          .workflow-steps {
            display: grid;
            gap: 12px;
            margin-top: 14px;
          }

          .workflow-step {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 12px;
            align-items: start;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
          }

          .workflow-step:last-child {
            border-bottom: none;
            padding-bottom: 0;
          }

          .step-index {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #f1e7da;
            color: var(--accent-strong);
            font-weight: 800;
            font-size: 13px;
          }

          .step-copy strong {
            display: block;
            margin-bottom: 4px;
            font-size: 15px;
          }

          .control-bar {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
            margin-top: 18px;
            padding: 14px;
            border-radius: 18px;
            background: rgba(255,255,255,0.6);
            border: 1px solid rgba(122, 100, 75, 0.10);
          }

          .control-label {
            font-weight: 700;
            color: var(--text-soft);
          }

          select,
          button {
            border-radius: 14px;
            padding: 12px 14px;
            font: inherit;
            border: 1px solid rgba(122, 100, 75, 0.14);
            transition: 180ms ease;
          }

          select {
            min-width: 150px;
            background: rgba(255,255,255,0.85);
            color: var(--title);
          }

          button {
            cursor: pointer;
            font-weight: 800;
          }

          button.primary {
            background: linear-gradient(135deg, #e88c68, #d97757);
            color: white;
            box-shadow: 0 14px 30px rgba(217, 119, 87, 0.24);
          }

          button.secondary {
            background: rgba(255,255,255,0.75);
            color: var(--title);
          }

          button:hover,
          select:hover {
            transform: translateY(-1px);
          }

          .result-shell {
            margin-top: 18px;
            display: grid;
            gap: 14px;
          }

          .result-banner {
            padding: 16px 18px;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(122, 100, 75, 0.12);
            color: var(--title);
            font-weight: 700;
          }

          .result-banner.ok {
            background: rgba(215, 239, 230, 0.9);
            border-color: rgba(103, 181, 141, 0.24);
          }

          .result-banner.fail {
            background: rgba(248, 230, 230, 0.92);
            border-color: rgba(196, 105, 105, 0.20);
          }

          .result-card {
            background: rgba(255,255,255,0.74);
          }

          .stat-value {
            display: block;
            margin-top: 8px;
            font-size: 24px;
            letter-spacing: -0.03em;
          }

          .info-panel,
          .log-card {
            padding: 18px;
          }

          .info-panel {
            border-radius: var(--radius-lg);
            background: rgba(255,255,255,0.74);
            border: 1px solid rgba(255,255,255,0.82);
            box-shadow: var(--glass-shadow);
          }

          .info-grid {
            display: grid;
            gap: 12px;
          }

          .info-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
            color: var(--text-soft);
          }

          .info-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
          }

          .info-row strong {
            text-align: right;
            color: var(--title);
          }

          .log-stack {
            display: grid;
            gap: 14px;
          }

          .log-card {
            background: rgba(255,255,255,0.68);
          }

          pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: Consolas, monospace;
            font-size: 12px;
            line-height: 1.62;
            color: #3b4452;
          }

          .scenario-card.easy { background: linear-gradient(180deg, rgba(215, 239, 230, 0.9), rgba(255,255,255,0.8)); }
          .scenario-card.medium { background: linear-gradient(180deg, rgba(246, 234, 214, 0.92), rgba(255,255,255,0.82)); }
          .scenario-card.hard { background: linear-gradient(180deg, rgba(248, 230, 230, 0.92), rgba(255,255,255,0.82)); }
          .scenario-card.signal { background: linear-gradient(180deg, rgba(230, 238, 251, 0.92), rgba(255,255,255,0.82)); }

          .scenario-card strong {
            display: block;
            margin-top: 10px;
            font-size: 17px;
            line-height: 1.4;
          }

          .link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 13px 18px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 800;
            color: white;
            background: linear-gradient(135deg, #e88c68, #d97757);
            box-shadow: 0 12px 24px rgba(217, 119, 87, 0.20);
          }

          .link.secondary {
            color: var(--title);
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(122, 100, 75, 0.12);
            box-shadow: none;
          }

          .footer-note {
            justify-self: end;
            color: var(--text-soft);
            font-size: 13px;
          }

          .hidden { display: none; }

          @media (max-width: 1100px) {
            .hero-main,
            .signal-row,
            .priority-grid,
            .detail-grid,
            .runner-grid,
            .result-grid,
            .insight-grid,
            .footer-row,
            .topbar {
              grid-template-columns: 1fr;
            }

            .footer-note {
              justify-self: start;
            }
          }

          @media (max-width: 720px) {
            .page {
              width: min(100% - 18px, 1240px);
              padding-top: 18px;
            }

            .hero-inner {
              padding: 18px;
            }

            h1 {
              font-size: clamp(38px, 16vw, 62px);
            }

            .lead {
              font-size: 16px;
            }

            .control-bar {
              align-items: stretch;
            }

            select,
            button {
              width: 100%;
            }
          }
        </style>
      </head>
      <body>
        <main class="page">
          <section class="hero">
            <div class="hero-inner">
              <div class="topbar">
                <div class="brand-pill">Sev1Bench · Incident response benchmark for OpenEnv evaluation</div>
                <div class="status-pill"><span class="status-dot"></span> Live Hugging Face Space runtime</div>
              </div>

              <div class="hero-main">
                <div class="hero-copy">
                  <div class="eyebrow">Phase 2 · Round 1 review surface</div>
                  <h1>Sev1Bench</h1>
                  <p class="lead">
                    A clean, judge-friendly benchmark interface for validating whether an agent can investigate
                    deterministic evidence, identify the real failing service, apply the right remediation, and
                    communicate recovery truthfully under pressure.
                  </p>

                  <div class="priority-grid">
                    <div class="feature-card">
                      <div class="mini-label">Primary interaction</div>
                      <h3>Run the benchmark instantly</h3>
                      <p>The live task runner is the first-class experience so judges can validate behavior immediately.</p>
                    </div>
                    <div class="feature-card">
                      <div class="mini-label">Review flow</div>
                      <h3>Inspect, run, verify</h3>
                      <p>The interface is structured around the shortest path from curiosity to confidence.</p>
                    </div>
                    <div class="feature-card">
                      <div class="mini-label">Presentation quality</div>
                      <h3>Premium light-mode design</h3>
                      <p>Soft contrast, glass surfaces, and whitespace replace heavy cyber dashboard aesthetics.</p>
                    </div>
                    <div class="feature-card">
                      <div class="mini-label">Hackathon value</div>
                      <h3>High trust, low friction</h3>
                      <p>Core signals, execution traces, and benchmark context are visible without hunting through menus.</p>
                    </div>
                  </div>
                </div>

                <div class="hero-panel">
                  <div class="glass-card">
                    <div class="section-title">Reviewer workflow</div>
                    <div class="workflow-steps">
                      <div class="workflow-step">
                        <div class="step-index">01</div>
                        <div class="step-copy">
                          <strong>Choose a task</strong>
                          <div class="small-copy">Start with easy, medium, or hard directly from the live runner.</div>
                        </div>
                      </div>
                      <div class="workflow-step">
                        <div class="step-index">02</div>
                        <div class="step-copy">
                          <strong>Run the live episode</strong>
                          <div class="small-copy">Trigger the benchmark and observe score, step count, and raw traces.</div>
                        </div>
                      </div>
                      <div class="workflow-step">
                        <div class="step-index">03</div>
                        <div class="step-copy">
                          <strong>Inspect the evidence</strong>
                          <div class="small-copy">Review the runtime values, LLM lines, and backend response payload together.</div>
                        </div>
                      </div>
                      <div class="workflow-step">
                        <div class="step-index">04</div>
                        <div class="step-copy">
                          <strong>Validate the contract</strong>
                          <div class="small-copy">Open API docs and confirm the deployment matches the benchmark entrypoint.</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div class="glass-card">
                    <div class="section-title">Submission contract</div>
                    <div class="metric-list">
                      <div class="metric-row"><span>Environment app</span><strong>server/app.py</strong></div>
                      <div class="metric-row"><span>Evaluation entrypoint</span><strong>inference.py</strong></div>
                      <div class="metric-row"><span>Scenario coverage</span><strong>easy · medium · hard</strong></div>
                      <div class="metric-row"><span>Live tester route</span><strong>/ui/test-run</strong></div>
                      <div class="metric-row"><span>API schema</span><strong>/docs</strong></div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="signal-row" style="margin-top: 24px;">
                <div class="signal-card">
                  <div class="mini-label">Benchmark model</div>
                  <span class="signal-value">Investigate → Fix → Communicate</span>
                </div>
                <div class="signal-card">
                  <div class="mini-label">Space purpose</div>
                  <span class="signal-value">Judge-facing benchmark presentation with live execution visibility</span>
                </div>
                <div class="signal-card">
                  <div class="mini-label">Runtime posture</div>
                  <span class="signal-value">Environment-configured evaluation backed by the deployed Space runtime</span>
                </div>
                <div class="signal-card">
                  <div class="mini-label">Winning advantage</div>
                  <span class="signal-value">Cleaner verification flow, stronger aesthetics, faster reviewer comprehension</span>
                </div>
              </div>

              <div class="runner-grid">
                <div class="glass-card runner-shell">
                  <div class="runner-hero">
                    <div class="runner-title">
                      <div>
                        <div class="eyebrow">Judge focal point</div>
                        <h2>Run a live Hugging Face test</h2>
                      </div>
                      <div class="runner-highlight">Fastest path to validation</div>
                    </div>
                    <p>
                      Launch a benchmark episode from the homepage and review the exact execution surface judges care
                      about: outcome, score, step trace, runtime values, and raw response details.
                    </p>
                  </div>

                  <div class="detail-grid">
                    <div class="scenario-card easy">
                      <div class="mini-label">easy</div>
                      <strong>api-service rollback</strong>
                    </div>
                    <div class="scenario-card medium">
                      <div class="mini-label">medium</div>
                      <strong>auth-service restart</strong>
                    </div>
                    <div class="scenario-card hard">
                      <div class="mini-label">hard</div>
                      <strong>db-cluster scale up</strong>
                    </div>
                    <div class="scenario-card signal">
                      <div class="mini-label">success signal</div>
                      <strong>Health restored + truthful status update</strong>
                    </div>
                  </div>

                  <div class="control-bar">
                    <span class="control-label">Task</span>
                    <select id="task-select">
                      <option value="easy">easy</option>
                      <option value="medium">medium</option>
                      <option value="hard">hard</option>
                    </select>
                    <button id="run-test-btn" class="primary" type="button">Run live benchmark</button>
                    <button id="clear-test-btn" class="secondary" type="button">Clear results</button>
                  </div>

                  <div id="result-shell" class="result-shell hidden">
                    <div id="result-banner" class="result-banner"></div>

                    <div class="result-grid">
                      <div class="result-card">
                        <div class="stat-label">Task</div>
                        <span id="result-task" class="stat-value">—</span>
                      </div>
                      <div class="result-card">
                        <div class="stat-label">Success</div>
                        <span id="result-success" class="stat-value">—</span>
                      </div>
                      <div class="result-card">
                        <div class="stat-label">Steps</div>
                        <span id="result-steps" class="stat-value">—</span>
                      </div>
                      <div class="result-card">
                        <div class="stat-label">Score</div>
                        <span id="result-score" class="stat-value">—</span>
                      </div>
                    </div>

                    <div class="runner-grid" style="margin-top: 0;">
                      <div class="log-stack">
                        <div class="info-panel">
                          <div class="section-title">Runtime values</div>
                          <div class="info-grid">
                            <div class="info-row"><span>Model</span><strong id="result-model">—</strong></div>
                            <div class="info-row"><span>API base URL</span><strong id="result-base-url">—</strong></div>
                            <div class="info-row"><span>Rewards</span><strong id="result-rewards">—</strong></div>
                          </div>
                        </div>
                        <div class="log-card">
                          <div class="panel-label">LLM request logs</div>
                          <pre id="result-llm-logs">No run yet.</pre>
                        </div>
                      </div>

                      <div class="log-stack">
                        <div class="log-card">
                          <div class="panel-label">Step logs</div>
                          <pre id="result-step-logs">No run yet.</pre>
                        </div>
                        <div class="log-card">
                          <div class="panel-label">Raw response</div>
                          <pre id="result-json">No run yet.</pre>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div class="glass-card workflow-card">
                  <div class="eyebrow">Why this presentation is stronger</div>
                  <div class="insight-grid">
                    <div class="feature-card">
                      <div class="mini-label">Clarity</div>
                      <h3>Judges see the action first</h3>
                      <p>The live run trigger is elevated above supporting information instead of buried inside the page.</p>
                    </div>
                    <div class="feature-card">
                      <div class="mini-label">Flow</div>
                      <h3>Evidence follows the run</h3>
                      <p>Outcome, score, traces, and raw payload appear in a single visual flow after execution.</p>
                    </div>
                    <div class="feature-card">
                      <div class="mini-label">Aesthetic</div>
                      <h3>Light, modern, premium</h3>
                      <p>Soft glass cards and breathable spacing make the benchmark feel considered and competition-ready.</p>
                    </div>
                    <div class="feature-card">
                      <div class="mini-label">Strategy</div>
                      <h3>Lower reviewer effort</h3>
                      <p>Every key proof point is visible where the judge needs it, reducing cognitive overhead.</p>
                    </div>
                  </div>
                </div>
              </div>

              <div class="footer-row">
                <a class="link" href="https://github.com/krrishyaa/Sev1Bench" target="_blank" rel="noreferrer">Open GitHub Repo</a>
                <a class="link secondary" href="/docs">Open API Docs</a>
                <div class="footer-note">Elegant benchmark presentation for Meta × Hugging Face review.</div>
              </div>
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
                ? "Benchmark run completed successfully."
                : "Benchmark run completed but the incident was not fully resolved.";
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
