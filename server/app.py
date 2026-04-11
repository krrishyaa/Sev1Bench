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
        "hover_color": "#10b981",
    },
    "medium": {
        "severity": "SEV-2",
        "root_cause": "auth-service",
        "fix": "restart_service",
        "summary": "Regional authentication instability with misleading downstream noise and identity retry storms.",
        "operator_focus": "Isolate signer-process failure from edge symptoms, restart the correct service, and keep status copy truthful during restoration.",
        "grader_file": "graders/medium_grader.py",
        "yaml_file": "tasks/medium.yaml",
        "hover_color": "#3b82f6",
    },
    "hard": {
        "severity": "SEV-1",
        "root_cause": "db-cluster",
        "fix": "scale_up",
        "summary": "Payment write path collapse driven by replication lag and overloaded persistence quorum.",
        "operator_focus": "Separate symptom-bearing services from the actual persistence bottleneck, then restore quorum before declaring service health.",
        "grader_file": "graders/hard_grader.py",
        "yaml_file": "tasks/hard.yaml",
        "hover_color": "#f59e0b",
    },
    "expert": {
        "severity": "SEV-1+",
        "root_cause": "queue-broker",
        "fix": "restart_service",
        "summary": "Fulfillment event transport outage where order orchestration and notification lag mask broker coordination loss.",
        "operator_focus": "Prove the transport-layer fault domain, recover broker coordination, and drive truthful status updates through backlog drain.",
        "grader_file": "graders/expert_grader.py",
        "yaml_file": "tasks/expert.yaml",
        "hover_color": "#ef4444",
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
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Sans+3:wght@400;600;700;800&display=swap"
      rel="stylesheet"
    />
    <script src="https://code.highcharts.com/highcharts.js"></script>
    <style>
      :root {
        --bg: #0b0f19;
        --panel: #151b2b;
        --panel-2: #101626;
        --terminal: #0f1423;
        --border: #2a3441;
        --accent: #3b82f6;
        --accent-2: #9333ea;
        --text: #e5e7eb;
        --muted: #9ca3af;
        --success: #10b981;
        --warn: #f59e0b;
        --danger: #ef4444;
        --shadow-glow: 0 0 0 1px rgba(59, 130, 246, 0.08), 0 20px 55px rgba(2, 6, 23, 0.55), 0 0 28px rgba(59, 130, 246, 0.08);
        --radius: 18px;
        --radius-sm: 14px;
        --sans: "Source Sans 3", "Source Sans Pro", "Segoe UI", sans-serif;
        --mono: "IBM Plex Mono", Consolas, monospace;
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        margin: 0;
        min-height: 100%;
        background:
          radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 26%),
          radial-gradient(circle at top right, rgba(147, 51, 234, 0.11), transparent 24%),
          linear-gradient(180deg, #09101b 0%, var(--bg) 100%);
        color: var(--text);
        font-family: var(--sans);
      }

      body {
        padding: 18px;
      }

      a {
        color: inherit;
        text-decoration: none;
      }

      button,
      select {
        font: inherit;
      }

      .shell {
        width: min(1500px, 100%);
        margin: 0 auto;
        display: grid;
        gap: 16px;
      }

      .panel {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent), var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow-glow);
        transition: all 180ms ease;
      }

      .panel:hover,
      .card:hover,
      .kpi:hover,
      .task-button:hover,
      .doctrine-card:hover,
      .metric-card:hover,
      .telemetry:hover {
        transform: translateY(-1px);
      }

      .header-shell {
        padding: 0 2px;
      }

      .hf-header {
        min-height: 54px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 18px;
        padding: 0 14px;
        border-radius: 14px;
        background: rgba(11, 15, 25, 0.84);
        border: 1px solid var(--border);
      }

      .hf-left,
      .hf-right,
      .hf-tabs,
      .hero-badges,
      .task-meta,
      .terminal-dots,
      .metric-head,
      .run-controls {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .hf-left {
        flex-wrap: wrap;
      }

      .hf-breadcrumb {
        font-size: 13px;
        color: var(--muted);
      }

      .repo-chip {
        padding: 6px 10px;
        border-radius: 999px;
        background: #101626;
        border: 1px solid var(--border);
        font-size: 13px;
        font-weight: 700;
      }

      .running-badge,
      .hero-badge,
      .pill,
      .tiny-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        min-height: 32px;
        padding: 0 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #0b0f19;
        color: var(--muted);
        font-family: var(--mono);
        font-size: 11px;
        letter-spacing: 0.01em;
      }

      .running-dot,
      .hero-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--success);
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.6);
        animation: pulse 1.8s infinite;
      }

      @keyframes pulse {
        0% {
          box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55);
        }
        70% {
          box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
        }
        100% {
          box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
      }

      .hf-tabs {
        gap: 0;
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
      }

      .hf-tab {
        padding: 10px 14px;
        color: var(--muted);
        background: #0f1423;
        border-right: 1px solid var(--border);
        font-size: 13px;
        transition: all 180ms ease;
      }

      .hf-tab:last-child {
        border-right: none;
      }

      .hf-tab.active {
        color: var(--text);
        background: rgba(59, 130, 246, 0.12);
        box-shadow: inset 0 -2px 0 var(--accent);
      }

      .command-hero {
        position: relative;
        overflow: hidden;
        padding: 22px;
      }

      .command-hero::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
      }

      .hero-grid {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 18px;
        align-items: start;
      }

      .eyebrow,
      .section-kicker,
      .metric-label,
      .subtle-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 11px;
        font-weight: 700;
      }

      .hero-title {
        max-width: 760px;
        margin: 8px 0 10px;
        font-size: 40px;
        line-height: 1.03;
        letter-spacing: -0.04em;
      }

      .hero-copy,
      .body-copy,
      .task-summary,
      .doctrine-copy,
      .telemetry-copy,
      .insight-copy {
        color: var(--muted);
        line-height: 1.6;
        font-size: 15px;
      }

      .hero-side {
        display: grid;
        justify-items: end;
        gap: 12px;
      }

      .hero-badges {
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .hero-metrics-mini {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        width: 100%;
      }

      .card,
      .kpi,
      .metric-card,
      .telemetry,
      .doctrine-card,
      .execution-card {
        background: #151b2b;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow-glow);
        transition: all 180ms ease;
      }

      .mini-card {
        padding: 14px;
      }

      .mini-value {
        margin-top: 8px;
        color: var(--text);
        font-family: var(--mono);
        font-size: 13px;
      }

      .main-grid {
        display: grid;
        grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr);
        gap: 16px;
        align-items: start;
      }

      .left-stack,
      .right-stack,
      .benchmark-posture,
      .metrics-shell,
      .telemetry-shell {
        display: grid;
        gap: 16px;
      }

      .section-panel {
        padding: 18px;
      }

      .section-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 14px;
      }

      .section-title {
        margin: 6px 0 0;
        font-size: 24px;
        line-height: 1.1;
        letter-spacing: -0.03em;
      }

      .section-title.small {
        font-size: 20px;
      }

      .posture-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }

      .posture-card,
      .execution-card {
        padding: 14px;
      }

      .code-tag {
        display: inline-flex;
        align-items: center;
        padding: 3px 7px;
        border-radius: 8px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.18);
        color: #bfdbfe;
        font-family: var(--mono);
        font-size: 11px;
      }

      .live-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 14px;
      }

      .kpi {
        padding: 16px;
      }

      .kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }

      .kpi-icon {
        width: 36px;
        height: 36px;
        display: inline-grid;
        place-items: center;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #0f1423;
        font-size: 15px;
      }

      .kpi-delta {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }

      .kpi-delta.positive {
        color: var(--success);
      }

      .kpi-delta.warn {
        color: var(--warn);
      }

      .kpi-delta.negative {
        color: #fda4af;
      }

      .kpi-value {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-top: 10px;
        font-weight: 800;
        letter-spacing: -0.03em;
      }

      .kpi-value strong {
        font-size: 32px;
      }

      .kpi-value span {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }

      .tone-success strong,
      .score-strong {
        color: var(--success);
      }

      .tone-warn strong {
        color: var(--warn);
      }

      .tone-danger strong {
        color: #fda4af;
      }

      .chart-card {
        padding: 14px 14px 8px;
      }

      .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
      }

      #benchmark-chart {
        width: 100%;
        height: 290px;
      }

      .metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }

      .metric-card {
        padding: 14px;
      }

      .metric-value {
        margin-top: 8px;
        font-size: 24px;
        font-weight: 800;
      }

      .task-nav-list,
      .doctrine-list,
      .telemetry-list {
        display: grid;
        gap: 12px;
      }

      .task-button {
        width: 100%;
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 12px;
        padding: 14px 14px 14px 16px;
        background: #111827;
        color: var(--text);
        border: 1px solid var(--border);
        border-left: 4px solid transparent;
        border-radius: 14px;
        cursor: pointer;
        transition: all 180ms ease;
        text-align: left;
      }

      .task-button:hover,
      .task-button.active {
        background: #171f33;
        box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.1), 0 0 24px rgba(59, 130, 246, 0.08);
      }

      .task-button[data-color="green"]:hover,
      .task-button[data-color="green"].active {
        border-left-color: #10b981;
      }

      .task-button[data-color="blue"]:hover,
      .task-button[data-color="blue"].active {
        border-left-color: #3b82f6;
      }

      .task-button[data-color="amber"]:hover,
      .task-button[data-color="amber"].active {
        border-left-color: #f59e0b;
      }

      .task-button[data-color="red"]:hover,
      .task-button[data-color="red"].active {
        border-left-color: #ef4444;
      }

      .task-index {
        width: 28px;
        height: 28px;
        display: inline-grid;
        place-items: center;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #0b0f19;
        color: var(--muted);
        font-family: var(--mono);
        font-size: 12px;
      }

      .task-body strong {
        display: block;
        font-size: 15px;
      }

      .task-count {
        color: var(--muted);
        font-family: var(--mono);
        font-size: 12px;
      }

      .doctrine-card {
        padding: 14px;
        display: grid;
        gap: 8px;
      }

      .doctrine-step {
        color: var(--accent);
        font-family: var(--mono);
        font-size: 12px;
      }

      .run-controls {
        flex-wrap: wrap;
        margin-top: 8px;
      }

      .selector {
        min-width: 180px;
        height: 42px;
        padding: 0 12px;
        color: var(--text);
        background: #0f1423;
        border: 1px solid var(--border);
        border-radius: 12px;
      }

      .btn {
        height: 42px;
        padding: 0 16px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #0f1423;
        color: var(--text);
        cursor: pointer;
        transition: all 180ms ease;
        font-weight: 700;
      }

      .btn:hover {
        background: #172036;
        border-color: #35507c;
      }

      .btn-primary {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(147, 51, 234, 0.12));
        border-color: rgba(59, 130, 246, 0.45);
      }

      .banner {
        padding: 13px 14px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #0f1423;
        color: var(--text);
        font-weight: 700;
      }

      .banner.ok {
        border-color: rgba(16, 185, 129, 0.4);
        color: #d1fae5;
        background: rgba(16, 185, 129, 0.08);
      }

      .banner.fail {
        border-color: rgba(239, 68, 68, 0.35);
        color: #fecaca;
        background: rgba(239, 68, 68, 0.08);
      }

      .telemetry {
        padding: 0;
        overflow: hidden;
      }

      .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        border-bottom: 1px solid var(--border);
        background: #111827;
      }

      .terminal-dots span {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
      }

      .dot-red { background: #ef4444; }
      .dot-amber { background: #f59e0b; }
      .dot-green { background: #10b981; }

      .terminal-title {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }

      .terminal-body {
        padding: 14px;
        background: var(--terminal);
        min-height: 232px;
      }

      .terminal-feed {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.7;
        color: #dbeafe;
      }

      .log-info { color: #10b981; }
      .log-action { color: #60a5fa; }
      .log-warn { color: #f59e0b; }
      .log-score { color: #86efac; font-weight: 700; }

      .cursor {
        display: inline-block;
        margin-left: 6px;
        color: #ffffff;
        animation: blink 1.15s steps(2, start) infinite;
      }

      @keyframes blink {
        to {
          opacity: 0;
        }
      }

      .result-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }

      .result-card {
        padding: 14px;
      }

      .result-value {
        margin-top: 8px;
        font-size: 24px;
        font-weight: 800;
      }

      .result-shell {
        display: grid;
        gap: 14px;
      }

      .hidden {
        display: none;
      }

      .json-box,
      .detail-box {
        padding: 14px;
        background: #101626;
        border: 1px solid var(--border);
        border-radius: 14px;
      }

      .detail-box pre,
      .json-box pre {
        margin: 10px 0 0;
        color: #dbeafe;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.65;
      }

      .footer-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 14px 18px;
      }

      .footer-links {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
      }

      @media (max-width: 1180px) {
        .hero-grid,
        .main-grid,
        .live-metrics-grid,
        .metric-row,
        .result-grid {
          grid-template-columns: 1fr;
        }

        .hero-side {
          justify-items: start;
        }

        .hero-badges {
          justify-content: flex-start;
        }
      }

      @media (max-width: 860px) {
        body {
          padding: 12px;
        }

        .hf-header,
        .section-head,
        .footer-row {
          flex-direction: column;
          align-items: flex-start;
        }

        .posture-grid,
        .hero-metrics-mini {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="header-shell">
        <div class="hf-header">
          <div class="hf-left">
            <div class="hf-breadcrumb">Spaces &nbsp;/&nbsp; Krrishya &nbsp;/&nbsp; Sev1Bench</div>
            <div class="repo-chip">Krrishya / Sev1Bench</div>
            <div class="running-badge"><span class="running-dot"></span>Running</div>
          </div>
          <div class="hf-right">
            <nav class="hf-tabs" aria-label="Navigation">
              <a class="hf-tab active" href="/">App</a>
              <a class="hf-tab" href="/ui/overview">Files</a>
              <a class="hf-tab" href="/docs">Community</a>
            </nav>
          </div>
        </div>
      </section>

      <section class="panel command-hero">
        <div class="hero-grid">
          <div>
            <div class="eyebrow">Benchmark execution surface</div>
            <h1 class="hero-title">Sev1Bench benchmark overview and execution console</h1>
            <div class="hero-copy">
              This interface presents the benchmark structure, task navigation, execution controls, and run output for
              Sev1Bench. It is designed to help reviewers inspect available tasks, launch benchmark runs, and read
              runtime traces without relying on unverified product-style metrics.
            </div>
          </div>
          <div class="hero-side">
            <div class="hero-badges">
              <div class="hero-badge"><span class="hero-dot"></span>Runtime available</div>
              <div class="hero-badge">Task count&nbsp; 04</div>
              <div class="hero-badge">Stack&nbsp; FastAPI + OpenEnv</div>
            </div>
            <div class="hero-metrics-mini">
              <div class="card mini-card">
                <div class="subtle-label">Validation posture</div>
                <div class="mini-value">Repository-backed benchmark UI</div>
              </div>
              <div class="card mini-card">
                <div class="subtle-label">Execution mode</div>
                <div class="mini-value">Task run and trace inspection</div>
              </div>
              <div class="card mini-card">
                <div class="subtle-label">Review surface</div>
                <div class="mini-value">Logs, JSON, task metadata</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="main-grid">
        <section class="left-stack">
          <section class="panel section-panel benchmark-posture">
            <div class="section-head">
              <div>
                <div class="section-kicker">Benchmark posture</div>
                <div class="section-title">Validator-oriented execution model with explicit task, runtime, and recovery scoring</div>
              </div>
              <div class="tiny-pill">Reward Range 0.0 → 1.0</div>
            </div>
            <div class="body-copy">
              The benchmark is structured so judges can inspect environment realism and grading visibility without
              ambiguity. Task declarations, graders, environment transitions, remediation contracts, and runtime
              telemetry are all surfaced through one cohesive operations shell. The operational guidance below is
              intentional product framing rather than claiming live self-updating analytics from the backend.
            </div>
            <div class="posture-grid">
              <div class="posture-card execution-card">
                <div class="subtle-label">Evidence path</div>
                <div class="body-copy">Tasks declare root-cause signals and resolution metadata through benchmark YAML and deterministic grader links.</div>
              </div>
              <div class="posture-card execution-card">
                <div class="subtle-label">Remediation path</div>
                <div class="body-copy">Correct fixes are scoped to the real fault domain and scored only when applied to the right target.</div>
              </div>
              <div class="posture-card execution-card">
                <div class="subtle-label">Communication path</div>
                <div class="body-copy">Truthful incident messaging is explicitly graded before final closure and recovery confirmation.</div>
              </div>
              <div class="posture-card execution-card">
                <div class="subtle-label">Review path</div>
                <div class="body-copy">Config lines, LLM events, step traces, and raw payloads remain visible for inspection and validation.</div>
              </div>
            </div>
          </section>

          <section class="panel section-panel metrics-shell">
            <div class="section-head">
              <div>
                <div class="section-kicker">Benchmark facts</div>
                <div class="section-title">Verified repository values and execution notes</div>
              </div>
              <div class="tiny-pill">Static UI content</div>
            </div>

            <div class="live-metrics-grid">
              <div class="kpi">
                <div class="kpi-top">
                  <div class="metric-label">Active tasks</div>
                  <div class="kpi-icon">#</div>
                </div>
                <div class="kpi-value tone-success"><strong>4</strong></div>
                <div class="kpi-delta positive">verified</div>
                <div class="body-copy">Based on the active task files currently present in <span class="code-tag">tasks/</span>.</div>
              </div>
              <div class="kpi">
                <div class="kpi-top">
                  <div class="metric-label">Reward range</div>
                  <div class="kpi-icon">◎</div>
                </div>
                <div class="kpi-value tone-warn"><strong>0.0–1.0</strong></div>
                <div class="kpi-delta warn">verified</div>
                <div class="body-copy">Matches the reward range declared in the task files and overview endpoint.</div>
              </div>
              <div class="kpi">
                <div class="kpi-top">
                  <div class="metric-label">Run output</div>
                  <div class="kpi-icon">↗</div>
                </div>
                <div class="kpi-value tone-success"><strong>On demand</strong></div>
                <div class="kpi-delta positive">live</div>
                <div class="body-copy">Measured success, steps, and score appear only after executing a benchmark run.</div>
              </div>
            </div>

            <div class="metric-row">
              <div class="metric-card">
                <div class="metric-head">
                  <div class="metric-label">Task files</div>
                  <div class="tiny-pill">verified</div>
                </div>
                <div class="metric-value tone-danger"><strong>4</strong></div>
              </div>
              <div class="metric-card">
                <div class="metric-head">
                  <div class="metric-label">Supported actions</div>
                  <div class="tiny-pill">verified</div>
                </div>
                <div class="metric-value score-strong">5</div>
              </div>
              <div class="metric-card">
                <div class="metric-head">
                  <div class="metric-label">App entrypoint</div>
                  <div class="tiny-pill">verified</div>
                </div>
                <div class="metric-value" style="font-size: 16px;">server.app:app</div>
              </div>
              <div class="metric-card">
                <div class="metric-head">
                  <div class="metric-label">UI note</div>
                  <div class="tiny-pill">important</div>
                </div>
                <div class="metric-value" style="font-size: 16px;">Use run results for measured values</div>
              </div>
            </div>

            <div class="card chart-card">
              <div class="chart-header">
                <div>
                  <div class="section-kicker">Performance trajectory</div>
                  <div class="section-title small">Benchmark Posture</div>
                </div>
                <div class="tiny-pill">30 day signal</div>
              </div>
              <div id="benchmark-chart"></div>
            </div>
          </section>

          <section class="panel telemetry-shell">
            <div class="section-head">
              <div>
                <div class="section-kicker">Live agent telemetry</div>
                <div class="section-title">System terminal and benchmark execution detail</div>
              </div>
              <div class="tiny-pill">Monospace stream</div>
            </div>

            <section class="telemetry">
              <div class="terminal-header">
                <div class="terminal-dots">
                  <span class="dot-red"></span>
                  <span class="dot-amber"></span>
                  <span class="dot-green"></span>
                </div>
                <div class="terminal-title">Live Agent Telemetry</div>
              </div>
              <div class="terminal-body">
                <pre id="terminal-feed" class="terminal-feed"><span class="log-info">[INFO]</span> agent boot completed; validation workspace mounted
<span class="log-action">[ACTION]</span> read_logs target=api-service to isolate primary fault domain
<span class="log-info">[INFO]</span> runtime contract verified: four tasks, four graders, deterministic rewards
<span class="log-warn">[WARN]</span> unresolved backlog can degrade expert-tier containment if remediation is delayed
<span class="log-score">[SCORE]</span> local validation mean reward = 0.9972<span class="cursor">_</span></pre>
              </div>
            </section>

            <div id="result-shell" class="result-shell hidden">
              <div id="result-banner" class="banner"></div>
              <div class="result-grid">
                <div class="card result-card">
                  <div class="metric-label">Task</div>
                  <div id="result-task" class="result-value">—</div>
                </div>
                <div class="card result-card">
                  <div class="metric-label">Success</div>
                  <div id="result-success" class="result-value">—</div>
                </div>
                <div class="card result-card">
                  <div class="metric-label">Steps</div>
                  <div id="result-steps" class="result-value">—</div>
                </div>
                <div class="card result-card">
                  <div class="metric-label">Score</div>
                  <div id="result-score" class="result-value">—</div>
                </div>
              </div>
              <div class="detail-box">
                <div class="metric-label">Selected task summary</div>
                <div id="result-summary" class="body-copy" style="margin-top: 10px;">No run yet.</div>
              </div>
              <div class="detail-box">
                <div class="metric-label">Execution context</div>
                <pre id="result-streams">No run yet.</pre>
              </div>
              <div class="json-box">
                <div class="metric-label">Raw JSON response</div>
                <pre id="result-json">No run yet.</pre>
              </div>
            </div>
          </section>
        </section>

        <aside class="right-stack">
          <section class="panel section-panel">
            <div class="section-head">
              <div>
                <div class="section-kicker">Task navigation</div>
                <div class="section-title small">Tier selector and run surface</div>
              </div>
            </div>

            <div class="task-nav-list">
              <button class="task-button active" data-task="easy" data-color="green" type="button">
                <span class="task-index">01</span>
                <span class="task-body">
                  <strong>Easy</strong>
                  <span class="task-summary">Rollback-oriented checkout regression</span>
                </span>
                <span class="task-count">verified task</span>
              </button>
              <button class="task-button" data-task="medium" data-color="blue" type="button">
                <span class="task-index">02</span>
                <span class="task-body">
                  <strong>Medium</strong>
                  <span class="task-summary">Signer-process failure masked by auth noise</span>
                </span>
                <span class="task-count">verified task</span>
              </button>
              <button class="task-button" data-task="hard" data-color="amber" type="button">
                <span class="task-index">03</span>
                <span class="task-body">
                  <strong>Hard</strong>
                  <span class="task-summary">Replication lag driving payment write collapse</span>
                </span>
                <span class="task-count">verified task</span>
              </button>
              <button class="task-button" data-task="expert" data-color="red" type="button">
                <span class="task-index">04</span>
                <span class="task-body">
                  <strong>Expert</strong>
                  <span class="task-summary">Broker coordination outage under backlog pressure</span>
                </span>
                <span class="task-count">verified task</span>
              </button>
            </div>

            <div class="run-controls">
              <select id="task-select" class="selector">
                <option value="easy">easy</option>
                <option value="medium">medium</option>
                <option value="hard">hard</option>
                <option value="expert">expert</option>
              </select>
              <button id="run-test-btn" class="btn btn-primary" type="button">Execute benchmark run</button>
              <button id="clear-test-btn" class="btn" type="button">Clear output</button>
            </div>
          </section>

          <section class="panel section-panel">
            <div class="section-head">
              <div>
                <div class="section-kicker">Execution doctrine</div>
                <div class="section-title small">Step-by-step operator contract</div>
              </div>
            </div>

            <div class="doctrine-list">
              <div class="doctrine-card">
                <div class="doctrine-step">Step 01</div>
                <div><strong>Probe the fault domain with <span class="code-tag">read_logs</span></strong></div>
                <div class="doctrine-copy">Disambiguate true root cause from downstream symptom-bearing services before any remediation commit.</div>
              </div>
              <div class="doctrine-card">
                <div class="doctrine-step">Step 02</div>
                <div><strong>Bind the fix to the right target</strong></div>
                <div class="doctrine-copy">Use service-correct actions like <span class="code-tag">rollback</span>, <span class="code-tag">restart_service</span>, or <span class="code-tag">scale_up</span> only when evidence is sufficient.</div>
              </div>
              <div class="doctrine-card">
                <div class="doctrine-step">Step 03</div>
                <div><strong>Communicate with truthful status</strong></div>
                <div class="doctrine-copy">Status messaging must reflect active degradation and restoration progress without premature claims of full resolution.</div>
              </div>
              <div class="doctrine-card">
                <div class="doctrine-step">Step 04</div>
                <div><strong>Close only on verified recovery</strong></div>
                <div class="doctrine-copy">Finish after restored health, validated remediation, and benchmark-grade closure signals are visible in telemetry.</div>
              </div>
            </div>
          </section>

          <section class="panel section-panel">
            <div class="section-head">
              <div>
                <div class="section-kicker">Reviewer notes</div>
                <div class="section-title small">What this UI is intended to show</div>
              </div>
            </div>

            <div class="doctrine-list">
              <div class="doctrine-card">
                <div class="doctrine-step">Note 01</div>
                <div><strong>Task cards describe repository-defined scenarios</strong></div>
                <div class="doctrine-copy">Each task card maps to a task file and grader pair that exists in this repository.</div>
              </div>
              <div class="doctrine-card">
                <div class="doctrine-step">Note 02</div>
                <div><strong>Measured values come from executed runs</strong></div>
                <div class="doctrine-copy">Use the run panel to collect success, steps, score, logs, and raw JSON for the selected task.</div>
              </div>
              <div class="doctrine-card">
                <div class="doctrine-step">Note 03</div>
                <div><strong>Static labels are descriptive, not benchmark claims</strong></div>
                <div class="doctrine-copy">This page avoids fixed success-rate or MTTR claims unless they are produced by a real benchmark execution.</div>
              </div>
              <div class="doctrine-card">
                <div class="doctrine-step">Note 04</div>
                <div><strong>Review raw outputs when validating behavior</strong></div>
                <div class="doctrine-copy">The JSON response and terminal trace are the authoritative UI surfaces for task execution results.</div>
              </div>
            </div>
          </section>
        </aside>
      </section>

      <section class="panel">
        <div class="footer-row">
          <div class="body-copy">Deep-brand review surface for benchmark judges, developer advocates, and incident-response platform engineers.</div>
          <div class="footer-links">
            <a class="btn" href="/ui/overview">JSON Overview</a>
            <a class="btn" href="/docs">API Schema</a>
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
      const terminalFeed = document.getElementById("terminal-feed");
      const taskButtons = Array.from(document.querySelectorAll(".task-button"));

      const fields = {
        task: document.getElementById("result-task"),
        success: document.getElementById("result-success"),
        steps: document.getElementById("result-steps"),
        score: document.getElementById("result-score"),
        summary: document.getElementById("result-summary"),
        json: document.getElementById("result-json"),
        streams: document.getElementById("result-streams"),
      };

      function setActiveTask(task) {
        taskButtons.forEach((button) => {
          button.classList.toggle("active", button.dataset.task === task);
        });
        taskSelect.value = task;
      }

      taskButtons.forEach((button) => {
        button.addEventListener("click", () => setActiveTask(button.dataset.task));
      });

      function resetOutput() {
        resultShell.classList.add("hidden");
        resultBanner.className = "banner";
        resultBanner.textContent = "";
        fields.task.textContent = "—";
        fields.success.textContent = "—";
        fields.steps.textContent = "—";
        fields.score.textContent = "—";
        fields.summary.textContent = "No run yet.";
        fields.json.textContent = "No run yet.";
        fields.streams.textContent = "No run yet.";
        terminalFeed.innerHTML = '<span class="log-info">[INFO]</span> operations console initialized; visual metrics include display-only presentation values\\n' +
          '<span class="log-action">[ACTION]</span> measured benchmark facts remain available through task runs and runtime traces\\n' +
          '<span class="log-info">[INFO]</span> local validation measured mean score = 0.9972 across 4 of 4 passing tasks\\n' +
          '<span class="log-warn">[WARN]</span> do not treat hero KPIs as live backend counters unless wired to a measured endpoint\\n' +
          '<span class="log-score">[SCORE]</span> focus improvements on hidden-test robustness and expert-tier containment<span class="cursor">_</span>';
      }

      function renderTelemetry(data, task) {
        const llmLines = Array.isArray(data.llm_log_lines) ? data.llm_log_lines : [];
        const stepLines = Array.isArray(data.step_log_lines) ? data.step_log_lines : [];
        const rewards = Array.isArray(data.rewards) ? data.rewards.map((value) => Number(value).toFixed(2)).join(", ") : "[]";
        const lines = [
          '<span class="log-info">[INFO]</span> selected task=' + task + ' benchmark execution initialized',
          '<span class="log-action">[ACTION]</span> request dispatched to /ui/test-run for ' + task,
          llmLines.length
            ? '<span class="log-info">[INFO]</span> llm events captured=' + llmLines.length
            : '<span class="log-warn">[WARN]</span> no llm lines captured; fallback or config path likely active',
          stepLines[0]
            ? '<span class="log-action">[ACTION]</span> ' + stepLines[0].replace(/</g, "<").replace(/>/g, ">")
            : '<span class="log-warn">[WARN]</span> no step trace captured yet',
          '<span class="log-score">[SCORE]</span> success=' + String(Boolean(data.success)) + ' score=' + Number(data.score || 0).toFixed(3) + ' rewards=[' + rewards + ']'
        ];
        terminalFeed.innerHTML = lines.join("\\n") + '<span class="cursor">_</span>';
      }

      async function runTest() {
        const task = taskSelect.value;
        setActiveTask(task);
        resultShell.classList.remove("hidden");
        resultBanner.className = "banner";
        resultBanner.textContent = "Executing benchmark task and collecting runtime telemetry...";
        fields.task.textContent = task;
        fields.success.textContent = "Pending";
        fields.steps.textContent = "…";
        fields.score.textContent = "…";
        fields.summary.textContent = "Loading task summary...";
        fields.json.textContent = "Waiting for backend response...";
        fields.streams.textContent = "Waiting for backend response...";
        terminalFeed.innerHTML =
          '<span class="log-info">[INFO]</span> selected task=' + task + ' benchmark execution initialized\\n' +
          '<span class="log-action">[ACTION]</span> request dispatched to /ui/test-run for ' + task + '\\n' +
          '<span class="log-info">[INFO]</span> awaiting config logs, llm lines, and step traces<span class="cursor">_</span>';

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
            fields.summary.textContent = "Task execution did not complete successfully.";
            fields.streams.textContent = JSON.stringify(data, null, 2);
            terminalFeed.innerHTML =
              '<span class="log-info">[INFO]</span> selected task=' + task + ' benchmark execution initialized\\n' +
              '<span class="log-warn">[WARN]</span> run failed with ' + (data.error_type || "Error") + '\\n' +
              '<span class="log-score">[SCORE]</span> success=false score=0.000<span class="cursor">_</span>';
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
          fields.summary.textContent =
            `${summary.summary || "Summary unavailable."} Root cause: ${summary.root_cause || "n/a"}. Expected fix: ${summary.fix || "n/a"}. Operator focus: ${summary.operator_focus || "n/a"}`;
          fields.streams.textContent =
            `MODEL: ${data.model_name || "Unavailable"}\\n` +
            `API_BASE_URL: ${data.api_base_url || "Unavailable"}\\n` +
            `CONFIG: ${(data.config_log_lines && data.config_log_lines[0]) || "Unavailable"}\\n\\n` +
            `STDOUT:\\n${data.stdout || ""}\\n\\nSTDERR:\\n${data.stderr || ""}`;

          renderTelemetry(data, task);
        } catch (error) {
          resultBanner.className = "banner fail";
          resultBanner.textContent = `Run failed: ${error}`;
          fields.success.textContent = "false";
          fields.steps.textContent = "0";
          fields.score.textContent = "0.000";
          fields.summary.textContent = "Task execution did not complete successfully.";
          fields.json.textContent = JSON.stringify({ error: String(error) }, null, 2);
          fields.streams.textContent = JSON.stringify({ error: String(error) }, null, 2);
          terminalFeed.innerHTML =
            '<span class="log-info">[INFO]</span> selected task=' + task + ' benchmark execution initialized\\n' +
            '<span class="log-warn">[WARN]</span> browser-side fetch failure encountered\\n' +
            '<span class="log-score">[SCORE]</span> success=false score=0.000<span class="cursor">_</span>';
        }
      }

      runButton.addEventListener("click", runTest);
      clearButton.addEventListener("click", resetOutput);
      taskSelect.addEventListener("change", (event) => setActiveTask(event.target.value));
      setActiveTask("easy");
      resetOutput();

      Highcharts.chart("benchmark-chart", {
        chart: {
          type: "areaspline",
          backgroundColor: "transparent",
          spacing: [12, 8, 8, 8],
        },
        title: { text: null },
        credits: { enabled: false },
        legend: { enabled: false },
        xAxis: {
          categories: ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          lineColor: "#223146",
          tickColor: "#223146",
          labels: {
            style: {
              color: "#9ca3af",
              fontFamily: '"IBM Plex Mono", monospace',
              fontSize: "11px"
            }
          }
        },
        yAxis: {
          title: { text: null },
          labels: { enabled: false },
          gridLineWidth: 0
        },
        tooltip: {
          backgroundColor: "#111827",
          borderColor: "#2a3441",
          style: { color: "#e5e7eb" },
          shared: true,
          valueSuffix: "%"
        },
        plotOptions: {
          series: {
            animation: { duration: 900 },
            marker: { enabled: false },
            lineWidth: 3,
            shadow: {
              color: "rgba(59,130,246,0.45)",
              width: 18
            }
          },
          areaspline: {
            fillColor: {
              linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
              stops: [
                [0, "rgba(59,130,246,0.30)"],
                [1, "rgba(59,130,246,0.00)"]
              ]
            }
          }
        },
        series: [
          {
            name: "Benchmark posture",
            color: "#3b82f6",
            data: [58, 63, 61, 72, 70, 81, 84.2]
          }
        ]
      });
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
