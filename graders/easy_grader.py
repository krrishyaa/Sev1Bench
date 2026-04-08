from __future__ import annotations

from typing import Any, Dict


def grade(final_observation: Any) -> Dict[str, Any]:
    reward = float(getattr(final_observation, "reward", 0.0))
    metadata = getattr(final_observation, "metadata", {}) or {}

    return {
        "task_id": "easy",
        "resolved": bool(metadata.get("resolved", False)),
        "root_cause_found": bool(getattr(final_observation, "root_cause_found", False)),
        "correct_fix_applied": bool(getattr(final_observation, "correct_fix_applied", False)),
        "truthful_status_posted": bool(getattr(final_observation, "truthful_status_posted", False)),
        "reward": max(0.0, min(1.0, reward)),
        "passed": bool(metadata.get("resolved", False)),
    }
