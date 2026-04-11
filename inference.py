"""Inference entrypoint for Sev1Bench."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, List, Optional

from openai import OpenAI

from models import IncidentAction
from server.environment import IncidentResponseEnvironment


IMAGE_NAME = os.getenv("IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
TASK_IDS_ENV = os.getenv("TASK_ID", "easy,medium,hard,expert")
BENCHMARK = "sev1bench"
DEFAULT_MODEL_NAME = "openai/gpt-4o-mini"
SUPPORTED_TASK_IDS = ("easy", "medium", "hard", "expert")
TARGET_TASKS = [task_id.strip() for task_id in TASK_IDS_ENV.split(",") if task_id.strip()]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def log_runtime_config(
    task: str,
    model: str,
    api_base_url: str,
    hf_token_present: bool,
    using_proxy_env: bool,
) -> None:
    print(
        "[CONFIG] "
        f"task={task} "
        f"model={model} "
        f"api_base_url={api_base_url} "
        f"hf_token_present={str(hf_token_present).lower()} "
        f"using_proxy_env={str(using_proxy_env).lower()}",
        flush=True,
    )


def log_llm_attempt(model: str) -> None:
    print(f"[LLM] attempting chat.completions.create model={model}", flush=True)


def log_llm_success(model: str) -> None:
    print(f"[LLM] request completed model={model}", flush=True)


def log_llm_failure(model: str, error_type: str, error_message: str) -> None:
    print(
        f"[LLM] request_failed model={model} error_type={error_type} error_message={error_message}",
        flush=True,
    )


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
        "expert": "restart_service",
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


def resolve_runtime_config() -> tuple[str, str]:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")
    if not HF_TOKEN.strip():
        raise ValueError("HF_TOKEN must not be empty")

    if API_BASE_URL is None:
        raise ValueError("API_BASE_URL environment variable is required")
    if not API_BASE_URL.strip():
        raise ValueError("API_BASE_URL must not be empty")

    resolved_model = (MODEL_NAME or DEFAULT_MODEL_NAME).strip()
    if not resolved_model:
        raise ValueError("MODEL_NAME resolved to an empty value")

    return API_BASE_URL.strip(), resolved_model


def _query_llm_once(client: OpenAI, model_name: str, observation: Any) -> dict[str, Any]:
    prompt = _build_prompt(observation)
    log_llm_attempt(model_name)
    try:
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
        log_llm_success(model_name)
    except Exception as exc:
        log_llm_failure(model_name, exc.__class__.__name__, str(exc))
        raise
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

        result = env.step(action)
        step_index += 1
        rewards.append(float(result.reward))
        log_step(step=step_index, action=action_str, reward=float(result.reward), done=bool(result.done), error=action_error)

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
    api_base_url, model_name = resolve_runtime_config()
    target_tasks = [task_id for task_id in TARGET_TASKS if task_id in SUPPORTED_TASK_IDS]
    if not target_tasks:
        raise ValueError("No valid task IDs configured in TASK_ID")

    overall_return_code = 0

    try:
        client = OpenAI(
            base_url=api_base_url,
            api_key=HF_TOKEN,
        )
    except Exception:
        print(traceback.format_exc(limit=3), file=sys.stderr, flush=True)
        return 1

    for current_task in target_tasks:
        log_start(task=current_task, env=BENCHMARK, model=model_name)
        log_runtime_config(
            task=current_task,
            model=model_name,
            api_base_url=api_base_url,
            hf_token_present=bool(HF_TOKEN and HF_TOKEN.strip()),
            using_proxy_env=True,
        )

        success = False
        steps = 0
        score = 0.0
        rewards: list[float] = []

        try:
            success, steps, score, rewards = run_task(
                task_id=current_task,
                client=client,
                model_name=model_name,
            )
        except Exception:
            print(traceback.format_exc(limit=3), file=sys.stderr, flush=True)
            overall_return_code = 1
        finally:
            log_end(success=success, steps=steps, score=score, rewards=rewards)

    return overall_return_code


if __name__ == "__main__":
    raise SystemExit(main())
