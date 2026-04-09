from __future__ import annotations

"""Pydantic models defining the incident response environment contract."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IncidentAction(BaseModel):
    """Action payload submitted by an agent to the environment."""

    action_type: str = Field(
        description="Action to execute. Supported actions: read_logs, restart_service, scale_up, rollback, post_status_update."
    )
    target: Optional[str] = Field(
        default=None,
        description="Target service or subsystem for the action, such as api-service, auth-service, or db-cluster.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Required only for post_status_update. Should be a truthful operator-facing status message.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured action metadata for future extensions.",
    )


class IncidentObservation(BaseModel):
    """Observation returned by the environment after each action."""

    done: bool = Field(default=False, description="Whether the episode has terminated.")
    reward: float = Field(default=0.0, ge=0.0, le=1.0, description="Bounded reward in the range [0.0, 1.0].")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured environment metadata including episode identifiers and log evidence.",
    )
    tick_count: int = Field(default=0, ge=0, description="Current incident tick count.")
    system_health: float = Field(default=1.0, ge=0.0, le=1.0, description="Aggregate system health from 0.0 to 1.0.")
    active_alerts: List[str] = Field(default_factory=list, description="Currently visible noisy alerts.")
    users_affected: int = Field(default=0, ge=0, description="Estimated impacted user count.")
    available_actions: List[str] = Field(default_factory=list, description="Actions the agent may take next.")
    root_cause_found: bool = Field(default=False, description="Whether the root cause has been positively identified.")
    correct_fix_applied: bool = Field(default=False, description="Whether the correct remediation has been applied.")
    truthful_status_posted: bool = Field(default=False, description="Whether a truthful required status update has been posted.")
    tool_output: List[str] = Field(
        default_factory=list,
        description="Human-readable tool output, especially deterministic log lines returned by read_logs.",
    )


class IncidentState(BaseModel):
    """Internal mutable state tracked across an incident episode."""

    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
    task_id: str = Field(default="easy")
    max_steps: int = Field(default=12, ge=1)
    system_health: float = Field(default=1.0, ge=0.0, le=1.0)
    active_alerts: List[str] = Field(default_factory=list)
    users_affected: int = Field(default=0, ge=0)
    root_cause_service: str = Field(default="")
    root_cause_found: bool = Field(default=False)
    correct_fix_applied: bool = Field(default=False)
    truthful_status_posted: bool = Field(default=False)
    resolved: bool = Field(default=False)
    failed: bool = Field(default=False)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)