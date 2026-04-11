from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from graders import GRADER_REGISTRY
from models import IncidentAction
from server.environment import IncidentResponseEnvironment


ROOT = Path(__file__).resolve().parent
TASK_IDS = ["easy", "medium", "hard", "expert"]


def load_grade_function(task_id: str, path: Path) -> Callable[[Any], dict[str, Any]]:
    grade = GRADER_REGISTRY.get(task_id)
    if callable(grade):
        return grade

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load grader module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grade = getattr(module, "grade", None)
    if not callable(grade):
        raise RuntimeError(f"Missing callable grade() in {path}")
    return grade


def run_task(task_id: str) -> dict[str, Any]:
    env = IncidentResponseEnvironment(task_id=task_id, max_steps=30)
    observation = env.reset()

    remediation_by_task = {
        "easy": "rollback",
        "medium": "restart_service",
        "hard": "scale_up",
        "expert": "restart_service",
    }

    candidate_services = observation.metadata.get("candidate_services", [])
    if not candidate_services:
        raise RuntimeError(f"No candidate_services exposed for task {task_id}")

    root_target = candidate_services[0]
    final_observation = env.step(IncidentAction(action_type="read_logs", target=root_target))
    if final_observation.done:
        raise RuntimeError(f"Task {task_id} terminated too early after read_logs")

    final_observation = env.step(
        IncidentAction(
            action_type=remediation_by_task[task_id],
            target=root_target,
        )
    )
    if final_observation.done and not final_observation.metadata.get("resolved", False):
        raise RuntimeError(f"Task {task_id} terminated unexpectedly after remediation")

    status_message = (
        "resolved, service restored and healthy"
        if final_observation.system_health >= 0.99
        else "mitigating incident, service still degraded while restoring capacity"
    )
    final_observation = env.step(
        IncidentAction(
            action_type="post_status_update",
            target="",
            message=status_message,
        )
    )

    while not final_observation.done:
        status_message = (
            "resolved, service restored and healthy"
            if final_observation.system_health >= 0.99
            else "mitigating incident, service still degraded while restoring capacity"
        )
        final_observation = env.step(
            IncidentAction(
                action_type="post_status_update",
                target="",
                message=status_message,
            )
        )

    grader_path = ROOT / "graders" / f"{task_id}_grader.py"
    grade = load_grade_function(task_id, grader_path)
    graded = grade(final_observation)

    reward = float(graded.get("reward", 0.0))
    if reward < 0.0 or reward > 1.0:
        raise RuntimeError(f"Reward out of range for task {task_id}: {reward}")

    return {
        "task_id": task_id,
        "done": bool(final_observation.done),
        "resolved": bool(final_observation.metadata.get("resolved", False)),
        "reward": reward,
        "grader_result": graded,
    }


def main() -> int:
    required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    print("[VALIDATION] starting")

    env_presence = {name: bool(os.environ.get(name)) for name in required_vars}
    print("[VALIDATION] env_vars=" + json.dumps(env_presence, sort_keys=True))

    required_files = [
        ROOT / "openenv.yaml",
        ROOT / "inference.py",
        ROOT / "models.py",
        ROOT / "Dockerfile",
        ROOT / "requirements.txt",
        ROOT / "pyproject.toml",
        ROOT / "server" / "app.py",
        ROOT / "server" / "environment.py",
        ROOT / "tasks" / "easy.yaml",
        ROOT / "tasks" / "medium.yaml",
        ROOT / "tasks" / "hard.yaml",
        ROOT / "tasks" / "expert.yaml",
        ROOT / "graders" / "easy_grader.py",
        ROOT / "graders" / "medium_grader.py",
        ROOT / "graders" / "hard_grader.py",
        ROOT / "graders" / "expert_grader.py",
        ROOT / "graders" / "__init__.py",
    ]

    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        print("[VALIDATION] missing_files=" + json.dumps(missing))
        return 1

    summaries = [run_task(task_id) for task_id in TASK_IDS]
    print("[VALIDATION] task_summaries=" + json.dumps(summaries, indent=2))
    print("[VALIDATION] success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
