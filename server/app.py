from __future__ import annotations

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
            --bg: #07111f;
            --panel: rgba(12, 24, 44, 0.82);
            --panel-strong: rgba(18, 34, 61, 0.95);
            --border: rgba(255, 255, 255, 0.1);
            --text: #edf4ff;
            --muted: #a9bbd6;
            --accent: #ff6b35;
            --accent-2: #ffbe0b;
            --success: #5eead4;
            --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
          }

          * {
            box-sizing: border-box;
          }

          body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background:
              radial-gradient(circle at top, rgba(255, 107, 53, 0.22), transparent 30%),
              radial-gradient(circle at 80% 20%, rgba(255, 190, 11, 0.18), transparent 24%),
              linear-gradient(180deg, #08111f 0%, #040914 100%);
          }

          .shell {
            width: min(1120px, calc(100% - 32px));
            margin: 0 auto;
            padding: 40px 0 56px;
          }

          .hero {
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(255, 107, 53, 0.18), rgba(18, 34, 61, 0.96));
            box-shadow: var(--shadow);
            border-radius: 28px;
            padding: 32px;
            overflow: hidden;
            position: relative;
          }

          .badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 14px;
            color: var(--muted);
          }

          h1 {
            margin: 18px 0 12px;
            font-size: clamp(36px, 7vw, 64px);
            line-height: 0.98;
            letter-spacing: -0.04em;
          }

          .lead {
            max-width: 760px;
            margin: 0;
            font-size: 18px;
            line-height: 1.7;
            color: var(--muted);
          }

          .hero-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 22px;
            margin-top: 28px;
          }

          .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 22px;
            backdrop-filter: blur(12px);
          }

          .card h2 {
            margin: 0 0 12px;
            font-size: 20px;
          }

          .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
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
          }

          .actions {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-top: 24px;
          }

          .pill {
            padding: 14px 12px;
            text-align: center;
            border-radius: 16px;
            background: var(--panel-strong);
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 14px;
          }

          .section {
            margin-top: 24px;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 22px;
          }

          ul {
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            line-height: 1.8;
          }

          code {
            color: var(--success);
            font-family: Consolas, monospace;
            font-size: 13px;
          }

          .footer {
            margin-top: 24px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
          }

          .link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 12px 16px;
            border-radius: 14px;
            text-decoration: none;
            color: #06101f;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            font-weight: 700;
          }

          .link.secondary {
            color: var(--text);
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border);
          }

          @media (max-width: 860px) {
            .hero-grid,
            .section,
            .actions {
              grid-template-columns: 1fr;
            }

            .shell {
              width: min(100% - 20px, 1120px);
              padding-top: 20px;
            }

            .hero {
              padding: 22px;
              border-radius: 22px;
            }
          }
        </style>
      </head>
      <body>
        <main class="shell">
          <section class="hero">
            <div class="badge">🚨 Sev1Bench · OpenEnv Incident Response Environment</div>
            <h1>Production incident war room, deployed live.</h1>
            <p class="lead">
              Sev1Bench is a Meta/OpenEnv-style incident-response environment where an agent must investigate logs,
              identify the real failing service, apply the correct remediation, post a truthful status update, and
              restore system health before the episode ends.
            </p>

            <div class="hero-grid">
              <div class="card">
                <h2>What this Space demonstrates</h2>
                <div class="metric"><span>Environment server</span><strong>Running</strong></div>
                <div class="metric"><span>Framework</span><strong>FastAPI + OpenEnv</strong></div>
                <div class="metric"><span>Tasks</span><strong>easy · medium · hard</strong></div>
                <div class="metric"><span>Primary flow</span><strong>Investigate → Remediate → Communicate → Resolve</strong></div>
              </div>

              <div class="card">
                <h2>Hackathon contract</h2>
                <div class="metric"><span>Root entrypoint</span><strong><code>inference.py</code></strong></div>
                <div class="metric"><span>Environment app</span><strong><code>server/app.py</code></strong></div>
                <div class="metric"><span>Env class</span><strong><code>IncidentResponseEnvironment</code></strong></div>
                <div class="metric"><span>Status</span><strong>Live and ready</strong></div>
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
                <h2>Episode expectations</h2>
                <ul>
                  <li>Discover the real root cause service from deterministic evidence.</li>
                  <li>Choose the exact task-specific remediation.</li>
                  <li>Avoid misleading status claims before recovery.</li>
                  <li>Drive system health back to a healthy terminal state.</li>
                </ul>
              </div>

              <div class="card">
                <h2>Submission links</h2>
                <ul>
                  <li>GitHub repository: <code>github.com/krrishyaa/Sev1Bench</code></li>
                  <li>Space slug: <code>Krrishya/Sev1Bench</code></li>
                  <li>Recommended verification: run <code>python inference.py</code> against this Space URL.</li>
                </ul>
              </div>
            </div>

            <div class="footer">
              <a class="link" href="https://github.com/krrishyaa/Sev1Bench" target="_blank" rel="noreferrer">Open GitHub Repo</a>
              <a class="link secondary" href="/docs">Open API Docs</a>
            </div>
          </section>
        </main>
      </body>
    </html>
    """
