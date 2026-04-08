from __future__ import annotations

from fastapi import FastAPI
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
