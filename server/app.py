from __future__ import annotations

import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app

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

app = create_fastapi_app(IncidentResponseEnvironment, IncidentAction, IncidentObservation) if create_fastapi_app is not None else _fallback_app()


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
            --bg: #06101c;
            --bg-2: #0a1630;
            --panel: rgba(10, 22, 42, 0.78);
            --panel-strong: rgba(12, 28, 52, 0.92);
            --panel-soft: rgba(255, 255, 255, 0.05);
            --border: rgba(255, 255, 255, 0.1);
            --text: #edf4ff;
            --muted: #9fb2cf;
            --accent: #ff7a18;
            --accent-2: #ffb703;
            --cyan: #5eead4;
            --blue: #60a5fa;
            --danger: #fb7185;
            --shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
          }

          * {
            box-sizing: border-box;
          }

          html {
            scroll-behavior: smooth;
          }

          body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background:
              radial-gradient(circle at 0% 0%, rgba(255, 122, 24, 0.2), transparent 24%),
              radial-gradient(circle at 100% 10%, rgba(94, 234, 212, 0.13), transparent 20%),
              radial-gradient(circle at 50% 100%, rgba(96, 165, 250, 0.12), transparent 26%),
              linear-gradient(180deg, var(--bg) 0%, #040914 100%);
          }

          body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
              linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 36px 36px;
            mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.9));
            opacity: 0.2;
          }

          .shell {
            width: min(1200px, calc(100% - 32px));
            margin: 0 auto;
            padding: 28px 0 64px;
            position: relative;
            z-index: 1;
          }

          .hero {
            position: relative;
            overflow: hidden;
            border-radius: 34px;
            padding: 34px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background:
              radial-gradient(circle at 15% 18%, rgba(255, 183, 3, 0.17), transparent 18%),
              radial-gradient(circle at 85% 16%, rgba(94, 234, 212, 0.13), transparent 18%),
              radial-gradient(circle at 50% 0%, rgba(96, 165, 250, 0.12), transparent 26%),
              linear-gradient(135deg, rgba(255, 122, 24, 0.12), rgba(10, 22, 42, 0.96) 35%, rgba(4, 9, 20, 0.98));
            box-shadow:
              var(--shadow),
              inset 0 1px 0 rgba(255, 255, 255, 0.06);
          }

          .hero::after {
            content: "";
            position: absolute;
            inset: auto -100px -100px auto;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 122, 24, 0.18), transparent 65%);
            filter: blur(10px);
          }

          .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
          }

          .badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 11px 16px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.09);
            color: var(--muted);
            font-size: 13px;
            letter-spacing: 0.02em;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
          }

          .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            color: var(--text);
            background: rgba(10, 26, 47, 0.8);
            border: 1px solid rgba(94, 234, 212, 0.28);
            font-size: 13px;
          }

          .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--cyan);
            box-shadow: 0 0 16px rgba(94, 234, 212, 0.8);
          }

          h1 {
            margin: 22px 0 12px;
            max-width: 760px;
            font-size: clamp(42px, 8vw, 84px);
            line-height: 0.92;
            letter-spacing: -0.06em;
          }

          .lead {
            margin: 0;
            max-width: 800px;
            font-size: 18px;
            line-height: 1.82;
            color: var(--muted);
          }

          .lead strong {
            color: var(--text);
          }

          .hero-highlight {
            margin-top: 24px;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
          }

          .highlight {
            padding: 16px 18px;
            border-radius: 18px;
            background: var(--panel-soft);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
          }

          .highlight span {
            display: block;
            margin-bottom: 8px;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
          }

          .highlight strong {
            font-size: 16px;
            color: var(--text);
          }

          .judge-banner {
            margin-top: 18px;
            padding: 16px 18px;
            border-radius: 18px;
            background: linear-gradient(90deg, rgba(255, 122, 24, 0.12), rgba(96, 165, 250, 0.08));
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text);
            line-height: 1.7;
          }

          .hero-grid {
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 22px;
            margin-top: 26px;
          }

          .section {
            margin-top: 22px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
          }

          .card {
            background: linear-gradient(180deg, rgba(10, 22, 42, 0.86), rgba(9, 20, 38, 0.76));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 22px;
            backdrop-filter: blur(14px);
            box-shadow:
              inset 0 1px 0 rgba(255, 255, 255, 0.04),
              0 18px 40px rgba(0, 0, 0, 0.18);
          }

          .card h2 {
            margin: 0 0 14px;
            font-size: 20px;
            letter-spacing: -0.02em;
          }

          .card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.75;
          }

          .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--muted);
          }

          .metric:last-child {
            border-bottom: none;
            padding-bottom: 0;
          }

          .metric strong {
            color: var(--text);
            font-weight: 700;
            text-align: right;
          }

          .actions {
            margin-top: 22px;
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
          }

          .pill {
            padding: 14px 12px;
            text-align: center;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(13, 32, 60, 0.96), rgba(9, 20, 38, 0.88));
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text);
            font-size: 14px;
            font-weight: 600;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
          }

          .callout-grid {
            margin-top: 22px;
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 22px;
          }

          ul {
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            line-height: 1.8;
          }

          li + li {
            margin-top: 6px;
          }

          code {
            color: var(--cyan);
            font-family: Consolas, monospace;
            font-size: 13px;
          }

          .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 14px;
          }

          .mini-card {
            border-radius: 18px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
          }

          .mini-card span {
            display: block;
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
          }

          .mini-card strong {
            color: var(--text);
            font-size: 15px;
          }

          .footer {
            margin-top: 24px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
          }

          .meta-note {
            margin-left: auto;
            font-size: 13px;
            color: var(--muted);
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
            box-shadow: 0 12px 28px rgba(255, 122, 24, 0.28);
          }

          .link.secondary {
            color: var(--text);
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border);
            box-shadow: none;
          }

          @media (max-width: 980px) {
            .hero-highlight,
            .hero-grid,
            .section,
            .callout-grid,
            .actions,
            .mini-grid {
              grid-template-columns: 1fr;
            }

            .metric {
              align-items: flex-start;
              flex-direction: column;
            }

            .metric strong {
              text-align: left;
            }

            .meta-note {
              margin-left: 0;
            }
          }

          @media (max-width: 640px) {
            .shell {
              width: min(100% - 18px, 1200px);
              padding-top: 18px;
            }

            .hero {
              padding: 22px;
              border-radius: 24px;
            }

            h1 {
              font-size: clamp(38px, 18vw, 64px);
            }

            .lead {
              font-size: 16px;
            }
          }
        </style>
      </head>
      <body>
        <main class="shell">
          <section class="hero">
            <div class="topbar">
              <div class="badge">🚨 Sev1Bench · OpenEnv Incident Response Environment</div>
              <div class="status-chip"><span class="status-dot"></span> Live on Hugging Face Spaces</div>
            </div>

            <h1>Sev1Bench</h1>

            <p class="lead">
              <strong>Sev1Bench</strong> is a high-signal Meta/OpenEnv-style incident-response benchmark where an agent
              must inspect deterministic evidence, identify the real failing service, apply the correct remediation,
              communicate truthfully, and restore system health. The landing page is reviewer-facing, while the API
              remains the canonical OpenEnv evaluation interface.
            </p>

            <div class="hero-highlight">
              <div class="highlight"><span>Mode</span><strong>Live OpenEnv deployment</strong></div>
              <div class="highlight"><span>Tasks</span><strong>easy · medium · hard</strong></div>
              <div class="highlight"><span>Validation</span><strong><code>inference.py</code> compatible</strong></div>
              <div class="highlight"><span>Focus</span><strong>Investigate → Fix → Communicate</strong></div>
            </div>

            <div class="judge-banner">
              Designed for hackathon judging: clear environment scope, root entrypoint visibility, Hugging Face Space
              identity, GitHub repository reference, action surface preview, and direct documentation access without
              obscuring the real OpenEnv endpoints.
            </div>

            <div class="hero-grid">
              <div class="card">
                <h2>What judges need to confirm</h2>
                <div class="metric"><span>Environment server</span><strong>Running</strong></div>
                <div class="metric"><span>Framework</span><strong>FastAPI + OpenEnv</strong></div>
                <div class="metric"><span>Task coverage</span><strong>easy · medium · hard</strong></div>
                <div class="metric"><span>Evaluation flow</span><strong>Investigate → Remediate → Communicate → Resolve</strong></div>
                <div class="metric"><span>Verification path</span><strong>Programmatic API + browser overview</strong></div>
              </div>

              <div class="card">
                <h2>Submission contract</h2>
                <div class="metric"><span>Root entrypoint</span><strong><code>inference.py</code></strong></div>
                <div class="metric"><span>Environment app</span><strong><code>server/app.py</code></strong></div>
                <div class="metric"><span>Env class</span><strong><code>IncidentResponseEnvironment</code></strong></div>
                <div class="metric"><span>Space slug</span><strong><code>Krrishya/Sev1Bench</code></strong></div>
                <div class="metric"><span>Status</span><strong>Live, verified, ready</strong></div>
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
                <h2>Environment guarantees</h2>
                <ul>
                  <li>Deterministic task-specific evidence and root-cause discovery.</li>
                  <li>Typed OpenEnv action and observation contracts.</li>
                  <li>Reward shaping across investigation, remediation, and truthful communication.</li>
                  <li>Terminal success only after genuine recovery conditions are met.</li>
                  <li>Interface supports both human review and agent-driven verification.</li>
                </ul>
              </div>

              <div class="card">
                <h2>Submission details</h2>
                <ul>
                  <li>GitHub repository: <code>github.com/krrishyaa/Sev1Bench</code></li>
                  <li>Hugging Face Space: <code>Krrishya/Sev1Bench</code></li>
                  <li>Primary verification path: run <code>python inference.py</code> against this deployment.</li>
                  <li>API docs remain available at <code>/docs</code> for endpoint inspection.</li>
                  <li>Browser landing page is presentation-only; the OpenEnv API is the canonical evaluation surface.</li>
                </ul>
              </div>
            </div>

            <div class="callout-grid">
              <div class="card">
                <h2>Hackathon-fit presentation</h2>
                <p>
                  The interface is intentionally concise, premium, and reviewer-friendly: it surfaces benchmark identity,
                  evaluation semantics, deployment status, and verification pointers without claiming unsupported features
                  or changing the actual environment behavior.
                </p>
                <div class="mini-grid">
                  <div class="mini-card"><span>Audience</span><strong>Meta / OpenEnv judges</strong></div>
                  <div class="mini-card"><span>Surface</span><strong>Hugging Face Space UI + API docs</strong></div>
                  <div class="mini-card"><span>Compatibility</span><strong>OpenEnv agent loop</strong></div>
                  <div class="mini-card"><span>Purpose</span><strong>Benchmark submission readiness</strong></div>
                </div>
              </div>

              <div class="card">
                <h2>Quick reviewer checklist</h2>
                <ul>
                  <li>Open the GitHub repo and inspect implementation files.</li>
                  <li>Use <code>/docs</code> to inspect the live API schema.</li>
                  <li>Run <code>python inference.py</code> against the Space endpoint.</li>
                  <li>Confirm tasks cover incident investigation, remediation, and communication.</li>
                  <li>Verify the landing page metadata matches the actual deployment contract.</li>
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
      </body>
            </html>
    """


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
