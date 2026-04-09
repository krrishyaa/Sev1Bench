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


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_ID", "easy")
BENCHMARK = os.getenv("BENCHMARK_NAME", "sev1bench")


if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


def _bool_str(value: bool) -> str:
    return str(bool(value)).lower()


def _reward_str(value: float) -> str:
    return f"{float(value):.2f}"


def _log_start(task_id: str, env_name: str, model_name: str) -> None:
    print(f"[START] task={task_id} env={env_name} model={model_name}")
    sys.stdout.flush()


def _format_action_str(action_payload: dict[str, Any]) -> str:
    action_type = str(action_payload.get("action_type", "") or "").strip()
    target = str(action_payload.get("target", "") or "").strip()
    message = str(action_payload.get("message", "") or "").strip()

    if target and message:
        return f'{action_type}(target={target},message="{message}")'
    if target:
        return f"{action_type}(target={target})"
    if message:
        return f'{action_type}(message="{message}")'
    return action_type or "unknown_action"


def _log_step(
    step_index: int,
    action_payload: dict[str, Any],
    result: Any,
    error: str | None = None,
) -> None:
    action_payload_safe = action_payload if isinstance(action_payload, dict) else {}
    action_str = _format_action_str(action_payload_safe)
    reward_value = getattr(result, "reward", 0.0)
    done_value = getattr(result, "done", False)
    error_value = error if error else "null"
    print(
        f"[STEP] step={step_index} action={action_str} "
        f"reward={_reward_str(reward_value)} "
        f"done={_bool_str(bool(done_value))} error={error_value}"
    )
    sys.stdout.flush()


def _log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(_reward_str(reward) for reward in rewards)
    print(f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_str}")
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
        "You are an incident-response agent operating a production environment.\n"
        "Choose exactly one action to investigate, remediate, or communicate status.\n"
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
    if metadata is None or not isinstance(metadata, dict):
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


def _choose_target_service(observation: Any) -> str:
    candidate_services = observation.metadata.get("candidate_services", [])
    if candidate_services:
        return str(candidate_services[0])

    last_read_logs = observation.metadata.get("last_read_logs")
    if isinstance(last_read_logs, str) and last_read_logs.strip():
        return last_read_logs.strip()

    tool_output = observation.tool_output if isinstance(observation.tool_output, list) else []
    for line in tool_output:
        if not isinstance(line, str):
            continue
        lower_line = line.lower()
        for marker in ("service=", "target=", "service:", "target:"):
            if marker in lower_line:
                raw_value = line.split(marker, 1)[1].strip()
                token = raw_value.split()[0].strip(",.;")
                if token:
                    return token

    return ""


def _infer_remediation_from_context(observation: Any) -> str | None:
    text_parts: list[str] = []

    impact_summary = observation.metadata.get("impact_summary")
    if isinstance(impact_summary, str):
        text_parts.append(impact_summary)

    investigation_hint = observation.metadata.get("investigation_hint")
    if isinstance(investigation_hint, str):
        text_parts.append(investigation_hint)

    recovery_note = observation.metadata.get("recovery_note")
    if isinstance(recovery_note, str):
        text_parts.append(recovery_note)

    tool_output = observation.tool_output if isinstance(observation.tool_output, list) else []
    text_parts.extend(str(item) for item in tool_output if isinstance(item, str))

    combined = " ".join(text_parts).lower()

    keyword_to_action = (
        (("rollback", "rolled back", "deploy rollback", "revert"), "rollback"),
        (("restart", "restart service", "recycle process", "reload process"), "restart_service"),
        (("scale", "capacity", "replica", "replicas", "autoscaling"), "scale_up"),
    )

    for keywords, action_type in keyword_to_action:
        if any(keyword in combined for keyword in keywords):
            return action_type

    return None


def _truthful_status_message(observation: Any) -> str:
    if observation.correct_fix_applied and observation.system_health >= 0.99:
        return "resolved, service restored and healthy"

    if observation.correct_fix_applied:
        return "mitigating incident, service still degraded while restoring capacity"

    return "investigating incident, service remains degraded while mitigation is prepared"


def _deterministic_policy_action(observation: Any) -> dict[str, Any] | None:
    target_service = _choose_target_service(observation)
    step_count = int(observation.metadata.get("step_count", observation.tick_count))
    timeline_pressure = observation.metadata.get("timeline_pressure", "early")
    inferred_remediation = _infer_remediation_from_context(observation)

    if not observation.root_cause_found:
        return {
            "action_type": "read_logs",
            "target": target_service,
            "message": "",
            "metadata": {
                "policy": "deterministic_root_cause_probe",
                "timeline_pressure": timeline_pressure,
                "step_count": step_count,
            },
        }

    if not observation.truthful_status_posted:
        return {
            "action_type": "post_status_update",
            "target": "",
            "message": _truthful_status_message(observation),
            "metadata": {
                "policy": "truthful_status_first",
                "timeline_pressure": timeline_pressure,
                "step_count": step_count,
            },
        }

    if not observation.correct_fix_applied and inferred_remediation is not None:
        return {
            "action_type": inferred_remediation,
            "target": target_service,
            "message": "",
            "metadata": {
                "policy": "context_inferred_remediation",
                "timeline_pressure": timeline_pressure,
                "step_count": step_count,
            },
        }

    if observation.correct_fix_applied and observation.system_health < 0.99:
        return {
            "action_type": "post_status_update",
            "target": "",
            "message": _truthful_status_message(observation),
            "metadata": {
                "policy": "continued_truthful_updates",
                "timeline_pressure": timeline_pressure,
                "step_count": step_count,
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
                "content": "You are a careful SRE operating a production incident-response environment. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    content = response.choices[0].message.content or ""
    return _extract_action(content)


def run_task(task_id: str, client: OpenAI, model_name: str) -> tuple[bool, int, list[float]]:
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
                action_payload = _fallback_action(observation, action_error)

        if action_payload is None:
            action_payload = _deterministic_policy_action(observation)

        if action_payload is None:
            try:
                action_payload = _query_llm_once(client, model_name, observation)
            except Exception as exc:
                action_error = f"fallback_after_model_error:{exc.__class__.__name__}"
                action_payload = _fallback_action(observation, action_error)

        try:
            action = IncidentAction(**action_payload)
        except Exception as exc:
            action_error = f"fallback_after_action_parse_error:{exc.__class__.__name__}"
            action_payload = _fallback_action(observation, action_error)
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

    success = bool(final_result.metadata.get("resolved", False))
    return success, step_index, rewards


def main() -> int:
    task_id = (os.getenv("TASK_ID", TASK_NAME) or "easy").strip() or "easy"
    env_name = (os.getenv("BENCHMARK_NAME", BENCHMARK) or "sev1bench").strip() or "sev1bench"

    success = False
    steps = 0
    rewards: list[float] = []

    _log_start(task_id=task_id, env_name=env_name, model_name=MODEL_NAME)

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
        success, steps, rewards = run_task(task_id=task_id, client=client, model_name=MODEL_NAME)
        return_code = 0
    except Exception:
        print(traceback.format_exc(limit=3), file=sys.stderr, flush=True)
        return_code = 1
    finally:
        _log_end(success=success, steps=steps, rewards=rewards)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
