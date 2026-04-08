from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from models import IncidentAction, IncidentObservation, IncidentState
from openenv.core.env_server import Environment


TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "root_cause_service": "api-service",
        "initial_health": 0.55,
        "alerts": [
            "critical: checkout-api latency spike observed from edge gateway",
            "warning: checkout retries increasing in frontend clients",
            "info: cache hit ratio remains nominal despite elevated latency",
        ],
        "correct_fix": "rollback",
        "wrong_targets": ["cache", "worker"],
        "impact_summary": "checkout latency is breaching the customer SLA and carts are timing out",
        "investigation_hint": "evidence points to a recent deploy-related regression in the checkout path",
        "recovery_note": "a rollback should stabilize upstream configuration and reduce retries",
        "users_base": 125,
    },
    "medium": {
        "root_cause_service": "auth-service",
        "initial_health": 0.40,
        "alerts": [
            "critical: auth-service token validation failures across regions",
            "warning: gateway 401 volume rising due to rejected sessions",
            "warning: user-service saturation increasing from retry storms",
        ],
        "correct_fix": "restart_service",
        "wrong_targets": ["gateway", "user-service"],
        "impact_summary": "users cannot log in and dependent identity flows are backing up",
        "investigation_hint": "symptoms are global but the failure originates in the token-issuing path",
        "recovery_note": "restarting the signer process should repopulate in-memory key handles",
        "users_base": 180,
    },
    "hard": {
        "root_cause_service": "db-cluster",
        "initial_health": 0.28,
        "alerts": [
            "critical: db-cluster replication lag exploding during payment writes",
            "critical: payment-api write timeouts breaching sla budget",
            "warning: frontend status page stale because upstream health checks are timing out",
            "info: batch-worker backlog increasing as queue drain slows",
        ],
        "correct_fix": "scale_up",
        "wrong_targets": ["frontend", "batch-worker"],
        "impact_summary": "payment writes are timing out and downstream settlement pipelines are stalling",
        "investigation_hint": "multiple services look degraded, but the persistence tier is the real bottleneck",
        "recovery_note": "capacity expansion is required before the write quorum can recover",
        "users_base": 260,
    },
}

LOG_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "easy": {
        "api-service": [
            "2026-04-08T09:00:00Z level=ERROR service=api-service request_id=req-7f31c2 route=/checkout latency_ms=4821 msg=\"Config mismatch after deploy\"",
            "2026-04-08T09:00:01Z level=ERROR service=api-service request_id=req-7f31c3 exception=ConfigError msg=\"PAYMENTS_BACKEND_URL points to invalid canary target\"",
            "2026-04-08T09:00:02Z level=TRACE service=api-service trace_id=trc-a91b span=rollback-check stack=\"ConfigError: invalid upstream\\n  at load_runtime_config(app/config.py:184)\\n  at bootstrap(app/main.py:61)\"",
        ],
        "cache": [
            "2026-04-08T09:00:00Z level=WARN service=cache request_id=req-c8112 msg=\"Cache miss rate elevated from slow upstream responses\"",
            "2026-04-08T09:00:02Z level=INFO service=cache trace_id=trc-c119 dependency=api-service msg=\"backend timeouts propagating to cache warm path\"",
        ],
        "worker": [
            "2026-04-08T09:00:01Z level=WARN service=worker request_id=req-w2011 queue=checkout-jobs msg=\"job retry count rising due to api-service 5xx responses\"",
            "2026-04-08T09:00:03Z level=INFO service=worker trace_id=trc-w201 dependency=api-service msg=\"downstream saturation detected, worker itself healthy\"",
        ],
    },
    "medium": {
        "auth-service": [
            "2026-04-08T09:10:00Z level=ERROR service=auth-service request_id=req-a1102 route=/token msg=\"JWT signer initialization failed after hot reload\"",
            "2026-04-08T09:10:01Z level=ERROR service=auth-service request_id=req-a1103 exception=SignerUnavailable msg=\"active signing key handle missing from process memory\"",
            "2026-04-08T09:10:03Z level=TRACE service=auth-service trace_id=trc-a77 stack=\"SignerUnavailable: signer handle missing\\n  at signer.load(runtime/keys.py:57)\\n  at issue_token(auth/service.py:212)\"",
        ],
        "gateway": [
            "2026-04-08T09:10:00Z level=WARN service=gateway request_id=req-g2201 route=/login status=401 msg=\"upstream auth-service rejected session token\"",
            "2026-04-08T09:10:02Z level=INFO service=gateway trace_id=trc-g220 dependency=auth-service msg=\"symptom observed at edge, root cause likely upstream auth\"",
        ],
        "user-service": [
            "2026-04-08T09:10:01Z level=WARN service=user-service request_id=req-u0191 msg=\"retry storm from gateway due to failed identity lookups\"",
            "2026-04-08T09:10:04Z level=INFO service=user-service trace_id=trc-u019 dependency=auth-service msg=\"local service healthy, waiting on auth restoration\"",
        ],
    },
    "hard": {
        "db-cluster": [
            "2026-04-08T09:20:00Z level=ERROR service=db-cluster request_id=req-d9911 shard=payments-primary msg=\"write quorum unavailable: replication lag exceeded 12.4s\"",
            "2026-04-08T09:20:01Z level=ERROR service=db-cluster request_id=req-d9912 exception=ReplicationTimeout msg=\"commit path blocked on overloaded replicas\"",
            "2026-04-08T09:20:03Z level=TRACE service=db-cluster trace_id=trc-d991 stack=\"ReplicationTimeout: quorum write blocked\\n  at commit(txn/replication.py:311)\\n  at persist(payment/store.py:88)\"",
        ],
        "frontend": [
            "2026-04-08T09:20:00Z level=WARN service=frontend request_id=req-f4402 page=/status msg=\"status widgets stale because payment-api health endpoint timed out\"",
            "2026-04-08T09:20:02Z level=INFO service=frontend trace_id=trc-f440 dependency=db-cluster msg=\"frontend healthy, degraded by upstream persistence path\"",
        ],
        "batch-worker": [
            "2026-04-08T09:20:01Z level=WARN service=batch-worker request_id=req-b5509 queue=settlements msg=\"consumer lag rising due to slow db commit acknowledgements\"",
            "2026-04-08T09:20:03Z level=INFO service=batch-worker trace_id=trc-b550 dependency=db-cluster msg=\"worker drain constrained by database replication lag\"",
        ],
    },
}


class IncidentResponseEnvironment(Environment):
    def __init__(self, task_id: str = "easy", max_steps: int = 30) -> None:
        self._default_task_id = task_id
        self._max_steps = max_steps
        self._state = IncidentState()
        self._reset_task(task_id)

    def reset(self) -> IncidentObservation:
        self._reset_task(self._default_task_id)
        return self._observation(reward=0.0, done=False, tool_output=["incident initialized"])

    def step(self, action: IncidentAction) -> IncidentObservation:
        if self._state.resolved or self._state.failed:
            return self._observation(reward=self._final_reward(), done=True, tool_output=["episode already finished"])

        self._state.step_count += 1
        reward_delta = 0.0
        tool_output: List[str] = []
        action_record = action.model_dump()
        self._state.action_history.append(action_record)

        action_type = action.action_type
        target = (action.target or "").strip()
        message = (action.message or "").strip().lower()
        task = TASKS[self._state.task_id]
        recovered_before_status = self._state.correct_fix_applied and self._state.system_health >= 0.99
        communication_before_fix = action_type == "post_status_update" and not self._state.correct_fix_applied

        if action_type == "read_logs":
            log_lines = self._read_logs(target)
            tool_output.extend(log_lines)
            if target == self._state.root_cause_service:
                first_identification = not self._state.root_cause_found
                self._state.root_cause_found = True
                reward_delta += 0.20 if first_identification else 0.05
                if first_identification:
                    tool_output.append(f"signal confirmed: {target} is the primary fault domain")
            elif target in task["wrong_targets"]:
                reward_delta -= 0.03
                tool_output.append(f"misleading signal: {target} is impacted but is not the root cause")
        elif action_type == "post_status_update":
            if self._is_truthful_status(message=message, recovered=recovered_before_status):
                first_truthful_update = not self._state.truthful_status_posted
                self._state.truthful_status_posted = True
                reward_delta += 0.15 if first_truthful_update else 0.04
                tool_output.append(f"status accepted: {message}")
                if communication_before_fix:
                    tool_output.append("stakeholder communication reduced uncertainty while mitigation was still in progress")
            else:
                reward_delta -= 0.20
                tool_output.append(f"status rejected as misleading: {message}")
        elif action_type in {"restart_service", "scale_up", "rollback"}:
            if action_type == task["correct_fix"] and target == self._state.root_cause_service:
                remediation_before_root_cause = not self._state.root_cause_found
                self._state.correct_fix_applied = True
                reward_delta += 0.35 if self._state.root_cause_found else 0.22
                tool_output.append(f"correct remediation applied to {target}")
                if remediation_before_root_cause:
                    tool_output.append("fix succeeded, but root-cause evidence was not explicitly gathered before mitigation")
            else:
                reward_delta -= 0.15
                tool_output.append(f"incorrect remediation: action={action_type} target={target}")
        else:
            reward_delta -= 0.10
            tool_output.append(f"unsupported action: {action_type}")

        if not self._state.correct_fix_applied:
            self._state.users_affected += self._users_increment()
            degradation = 0.08 if self._state.root_cause_found else 0.10
            self._state.system_health = max(0.0, self._state.system_health - degradation)
        else:
            recovery_gain = 0.18 if self._state.root_cause_found else 0.12
            self._state.system_health = min(1.0, self._state.system_health + recovery_gain)
            self._state.users_affected = max(0, self._state.users_affected - self._recovery_user_drop())

        if self._state.root_cause_found and self._state.correct_fix_applied and self._state.truthful_status_posted:
            self._state.system_health = min(1.0, self._state.system_health + 0.12)
            self._state.users_affected = max(0, self._state.users_affected - self._recovery_user_drop())

        if (
            self._state.system_health >= 0.99
            and self._state.correct_fix_applied
            and self._state.truthful_status_posted
        ):
            self._state.resolved = True
            self._state.active_alerts = ["info: all core services healthy and user impact contained"]
            tool_output.append("incident resolved with verified service recovery")

        if self._state.step_count >= self._state.max_steps and not self._state.resolved:
            self._state.failed = True
            tool_output.append("max tick limit reached before full restoration")

        reward = self._bounded_reward(reward_delta)
        done = self._state.resolved or self._state.failed
        if done:
            reward = self._final_reward()

        return self._observation(reward=reward, done=done, tool_output=tool_output)

    @property
    def state(self) -> IncidentState:
        return self._state

    def _reset_task(self, task_id: str) -> None:
        if task_id not in TASKS:
            task_id = "easy"

        task = TASKS[task_id]
        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=self._max_steps,
            system_health=task["initial_health"],
            active_alerts=list(task["alerts"]),
            users_affected=int(task.get("users_base", 125)),
            root_cause_service=task["root_cause_service"],
            root_cause_found=False,
            correct_fix_applied=False,
            truthful_status_posted=False,
            resolved=False,
            failed=False,
            action_history=[],
        )

    def _available_actions(self) -> List[str]:
        return [
            "read_logs",
            "restart_service",
            "scale_up",
            "rollback",
            "post_status_update",
        ]

    def _observation(self, reward: float, done: bool, tool_output: List[str]) -> IncidentObservation:
        task = TASKS[self._state.task_id]
        metadata = {
            "task_id": self._state.task_id,
            "episode_id": self._state.episode_id,
            "root_cause_service": self._state.root_cause_service if self._state.root_cause_found else None,
            "step_count": self._state.step_count,
            "resolved": self._state.resolved,
            "failed": self._state.failed,
            "candidate_services": self._candidate_services(),
            "last_read_logs": tool_output if tool_output else [],
            "impact_summary": task["impact_summary"],
            "investigation_hint": task["investigation_hint"],
            "recovery_note": task["recovery_note"],
            "timeline_pressure": self._timeline_pressure(),
            "action_history_length": len(self._state.action_history),
        }
        return IncidentObservation(
            done=done,
            reward=self._bounded_reward(reward),
            metadata=metadata,
            tick_count=self._state.step_count,
            system_health=self._state.system_health,
            active_alerts=list(self._state.active_alerts),
            users_affected=self._state.users_affected,
            available_actions=self._available_actions(),
            root_cause_found=self._state.root_cause_found,
            correct_fix_applied=self._state.correct_fix_applied,
            truthful_status_posted=self._state.truthful_status_posted,
            tool_output=tool_output,
        )

    def _bounded_reward(self, reward: float) -> float:
        return max(0.0, min(1.0, reward))

    def _final_reward(self) -> float:
        base = 0.0
        if self._state.root_cause_found:
            base += 0.25
        if self._state.correct_fix_applied:
            base += 0.40
        if self._state.truthful_status_posted:
            base += 0.15
        if self._state.resolved:
            base += 0.20

        time_decay = max(0.35, 1.0 - (0.03 * self._state.step_count))
        return max(0.0, min(1.0, base * time_decay))

    def _users_increment(self) -> int:
        return max(25, int((1.0 - self._state.system_health) * 100) + (8 * self._state.step_count))

    def _recovery_user_drop(self) -> int:
        return max(30, int(self._state.system_health * 90))

    def _timeline_pressure(self) -> str:
        if self._state.step_count <= 2:
            return "early"
        if self._state.step_count <= 5:
            return "elevated"
        if self._state.step_count <= 8:
            return "critical"
        return "severe"

    def _candidate_services(self) -> List[str]:
        services = [self._state.root_cause_service]
        services.extend(TASKS[self._state.task_id]["wrong_targets"])
        return services

    def _read_logs(self, target: str) -> List[str]:
        if not target:
            return ["no target specified for read_logs"]

        task_logs = LOG_TEMPLATES[self._state.task_id]
        if target in task_logs:
            return list(task_logs[target])

        base_time = datetime(2026, 4, 8, 9, 0, 0) + timedelta(minutes=self._state.step_count)
        timestamp = base_time.isoformat() + "Z"
        return [
            f"{timestamp} level=INFO service={target} request_id=req-generic-{self._state.step_count:04d} msg=\"service nominal; no direct evidence of root cause\""
        ]

    def _is_truthful_status(self, message: str, recovered: bool) -> bool:
        if not message:
            return False

        if recovered:
            return any(token in message for token in ["resolved", "restored", "healthy"])

        if "healthy" in message or "resolved" in message:
            return False

        return any(token in message for token in ["investigating", "mitigating", "degraded", "restoring"])
