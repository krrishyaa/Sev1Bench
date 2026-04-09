"""Scoring logic for the hard incident response benchmark task."""

from __future__ import annotations

from typing import Any, Dict


def _clamp_score(value: float) -> float:
    """Clamp a numeric score to the inclusive range [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def grade(final_observation: Any) -> Dict[str, Any]:
    """Compute the final grading payload for the hard task."""
    metadata = getattr(final_observation, "metadata", {}) or {}
    step_count = int(metadata.get("step_count", 0))
    resolved = bool(metadata.get("resolved", False))
    root_cause_found = bool(getattr(final_observation, "root_cause_found", False))
    correct_fix_applied = bool(getattr(final_observation, "correct_fix_applied", False))
    truthful_status_posted = bool(getattr(final_observation, "truthful_status_posted", False))
    users_affected = int(getattr(final_observation, "users_affected", 0))
    system_health = float(getattr(final_observation, "system_health", 0.0))

    investigation_score = 1.0 if root_cause_found else 0.0
    remediation_score = 1.0 if correct_fix_applied else 0.0
    communication_score = 1.0 if truthful_status_posted else 0.0
    if resolved and system_health >= 0.99:
        recovery_score = 1.0
    elif system_health >= 0.95:
        recovery_score = 0.7
    elif system_health >= 0.90:
        recovery_score = 0.4
    else:
        recovery_score = 0.0
    efficiency_score = _clamp_score(1.0 - (0.05 * max(0, step_count - 5)))
    impact_containment_score = _clamp_score(1.0 - (users_affected / 1500.0))

    composite_score = (
        0.24 * investigation_score
        + 0.22 * remediation_score
        + 0.20 * communication_score
        + 0.20 * recovery_score
        + 0.07 * efficiency_score
        + 0.07 * impact_containment_score
    )

    final_score = _clamp_score(composite_score)

    return {
        "task_id": "hard",
        "resolved": resolved,
        "root_cause_found": root_cause_found,
        "correct_fix_applied": correct_fix_applied,
        "truthful_status_posted": truthful_status_posted,
        "step_count": step_count,
        "system_health": round(system_health, 3),
        "users_affected": users_affected,
        "subscores": {
            "investigation": round(investigation_score, 3),
            "remediation": round(remediation_score, 3),
            "communication": round(communication_score, 3),
            "recovery": round(recovery_score, 3),
            "efficiency": round(efficiency_score, 3),
            "impact_containment": round(impact_containment_score, 3),
        },
        "reward": float(final_score),
        "passed": bool(resolved and root_cause_found and correct_fix_applied and truthful_status_posted),
    }