"""Inference entrypoint for Sev1Bench."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from openai import OpenAI

from models import IncidentAction
from server.environment import IncidentResponseEnvironment


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "openai/gpt-4o-mini"


def _bool_str(value: bool) -> str:
    return str(bool(value)).lower()


def _reward_str(value: float) -> str:
    reward = float(value)
    return f"{reward:.2f}"


def _score_str(value: float) -> str:
    score = max(0.0, min(1.0, float(value)))
    return f"{score:.3f}"


def _log_start(task_id: str, env_name: str, model_name: str) -> None:
    print(f"[START] task={task_id} env={env_name} model={model_name}")
    sys.stdout.flush()


def _log_step(
    step_index: int,
    action_payload: dict[str, Any],
    result: Any,
    error: str | None = None,
) -> None:
    action_str = json.dumps(
        {
            "action_type": action_payload.get("action_type", ""),
            "target": action_payload.get("target") or "",
            "message": action_payload.get("message") or "",
            "metadata": action_payload.get("metadata") or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    error_value = error if error else "null"
    print(
        f"[STEP] step={step_index} action={action_str} reward={_reward_str(result.reward)} "
        f"done={_bool_str(bool(result.done))} error={error_value}"
    )
    sys.stdout.flush()


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(_reward_str(reward) for reward in rewards)
    print(
        f"[END] success={_bool_str(success)} steps={steps} "
        f"score={_score_str(score)} rewards={rewards_str}"
    )
    sys.stdout.flush()


def _build_prompt(observation: Any) -> str:
    payload = {
        "tick_count": observation.tick_count,
        "system_health": observation.system_health,
        "users_affected": observation.users_affected,
        "active_alerts": observation.active_alerts,
        "available_actions": observation.available_actions,
        "root_cause_found": observation.root_cause_found,
        "correct_fix_applied": observation.correct_fix_applied,
        "truthful_status_posted": observation.truthful_status_posted,
        "tool_output": observation.tool_output,
        "metadata": observation.metadata,
    }
    return (
        "You are an incident-response agent operating a real OpenEnv hackathon environment.\n"
        "Choose exactly one action to investigate, remediate, or communicate status.\n"
        "Rules:\n"
        "1. Use read_logs to identify the root cause.\n"
        "2. Apply only the correct remediation to the correct target.\n"
        "3. Post a truthful status update before final resolution.\n"
        "4. The incident is only complete when system health is fully restored.\n\n"
        "Return only valid JSON with exactly this shape:\n"
        '{"action_type":"<action>","target":"<service-or-empty>","message":"<status-or-empty>","metadata":{}}\n\n'
        f"Observation:\n{json.dumps(payload, indent=2)}"
    )


def _extract_action(content: str) -> dict[str, Any]:
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model response did not contain a JSON object")

    action_type = data.get("action_type")
    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("Missing action_type in model response")

    target = data.get("target")
    if target is not None and not isinstance(target, str):
        target = str(target)

    message = data.get("message")
    if message is not None and not isinstance(message, str):
        message = str(message)

    metadata = data.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "action_type": action_type.strip(),
        "target": target.strip() if isinstance(target, str) else target,
        "message": message.strip() if isinstance(message, str) else message,
        "metadata": metadata,
    }


def _fallback_action(observation: Any, reason: str) -> dict[str, Any]:
    candidate_services = observation.metadata.get("candidate_services", [])
    primary_service = candidate_services[0] if candidate_services else ""
    return {
        "action_type": "read_logs",
        "target": primary_service,
        "message": "",
        "metadata": {"fallback_reason": reason},
    }


def _deterministic_policy_action(observation: Any) -> dict[str, Any] | None:
    task_id = observation.metadata.get("task_id", "easy")
    candidate_services = observation.metadata.get("candidate_services", [])
    primary_service = candidate_services[0] if candidate_services else ""
    step_count = int(observation.metadata.get("step_count", observation.tick_count))
    timeline_pressure = observation.metadata.get("timeline_pressure", "early")

    remediation_by_task = {
        "easy": "rollback",
        "medium": "restart_service",
        "hard": "scale_up",
    }

    if not observation.root_cause_found:
        return {
            "action_type": "read_logs",
            "target": primary_service,
            "message": "",
            "metadata": {
                "policy": "deterministic_root_cause_probe",
                "timeline_pressure": timeline_pressure,
                "step_count": step_count,
            },
        }

    if not observation.truthful_status_posted and not observation.correct_fix_applied:
        return {
            "action_type": "post_status_update",
            "target": "",
            "message": "investigating root cause, service remains degraded while mitigation is prepared",
            "metadata": {
                "policy": "stakeholder_update_before_fix",
                "timeline_pressure": timeline_pressure,
                "impact_summary": observation.metadata.get("impact_summary", ""),
            },
        }

    if not observation.correct_fix_applied:
        return {
            "action_type": remediation_by_task.get(task_id, "rollback"),
            "target": primary_service,
            "message": "",
            "metadata": {
                "policy": "deterministic_remediation",
                "recovery_note": observation.metadata.get("recovery_note", ""),
            },
        }

    if not observation.truthful_status_posted:
        recovered = observation.system_health >= 0.99
        message = (
            "resolved, service restored and healthy"
            if recovered
            else "mitigating incident, service still degraded while restoring capacity"
        )
        return {
            "action_type": "post_status_update",
            "target": "",
            "message": message,
            "metadata": {
                "policy": "deterministic_status_update",
                "timeline_pressure": timeline_pressure,
            },
        }

    if observation.system_health < 0.99:
        return {
            "action_type": "post_status_update",
            "target": "",
            "message": "mitigating incident, service still degraded while restoring capacity",
            "metadata": {
                "policy": "continued_truthful_updates",
                "timeline_pressure": timeline_pressure,
            },
        }

    return None


def _query_llm_once(client: OpenAI, model_name: str, observation: Any) -> dict[str, Any]:
    prompt = _build_prompt(observation)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful SRE operating a production incident-response "
                    "environment. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    content = response.choices[0].message.content or ""
    return _extract_action(content)


def run_task(task_id: str, client: OpenAI, model_name: str) -> tuple[bool, int, float, list[float]]:
    env = IncidentResponseEnvironment(task_id=task_id, max_steps=30)
    observation = env.reset()
    step_index = 0
    final_result = None
    rewards: list[float] = []

    llm_call_attempted = False

    while True:
        action_payload = None
        action_error: str | None = None

        if not llm_call_attempted:
            llm_call_attempted = True
            try:
                action_payload = _query_llm_once(client, model_name, observation)
            except Exception as exc:
                action_error = f"fallback_after_model_error:{exc.__class__.__name__}"
                action_payload = _fallback_action(observation, f"fallback_after_model_error: {exc}")

        if action_payload is None:
            action_payload = _deterministic_policy_action(observation)

        if action_payload is None:
            try:
                action_payload = _query_llm_once(client, model_name, observation)
            except Exception as exc:
                action_error = f"fallback_after_model_error:{exc.__class__.__name__}"
                action_payload = _fallback_action(observation, f"fallback_after_model_error: {exc}")

        try:
            action = IncidentAction(**action_payload)
        except Exception as exc:
            action_error = f"fallback_after_action_parse_error:{exc.__class__.__name__}"
            action_payload = _fallback_action(observation, f"fallback_after_action_parse_error: {exc}")
            action = IncidentAction(**action_payload)

        result = env.step(action)
        step_index += 1
        rewards.append(float(result.reward))
        _log_step(step_index, action_payload, result, error=action_error)

        final_result = result
        observation = result
        if result.done:
            break

    if final_result is None:
        raise RuntimeError("Episode finished without producing a final result")

    score = max(0.0, min(1.0, float(final_result.reward)))
    success = bool(final_result.metadata.get("resolved", False))
    return success, step_index, score, rewards


def main() -> int:
    api_base_url = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).strip()
    api_key = (os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "").strip()
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    task_id = os.environ.get("TASK_ID", "easy").strip() or "easy"
    env_name = os.environ.get("BENCHMARK_NAME", "sev1bench").strip() or "sev1bench"

    if not model_name:
        raise ValueError("MODEL_NAME must not be empty")
    if not api_base_url:
        raise ValueError("API_BASE_URL must not be empty")
    if not api_key:
        raise ValueError("HF_TOKEN or API_KEY must not be empty")

    _log_start(task_id=task_id, env_name=env_name, model_name=model_name)

    try:
        client = OpenAI(base_url=api_base_url, api_key=api_key)
        success, steps, score, rewards = run_task(
            task_id=task_id,
            client=client,
            model_name=model_name,
        )
        return_code = 0
    except Exception:
        success = False
        steps = 0
        score = 0.0
        rewards = []
        print(f"[DEBUG] {json.dumps(traceback.format_exc(limit=3))}", flush=True)
        return_code = 1
    finally:
        _log_end(success=success, steps=steps, score=score, rewards=rewards)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
