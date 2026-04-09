"""Run local baseline agents against Sev1Bench tasks and print markdown summaries."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

from models import IncidentAction
from server.environment import IncidentResponseEnvironment


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-latest"
DEFAULT_EPISODES = 3
TASK_IDS = ("easy", "medium", "hard")


@dataclass
class EpisodeTrace:
    """Stores the outcome and action history for a single benchmark episode."""

    task_id: str
    agent_name: str
    episode_index: int
    passed: bool
    resolved: bool
    step_count: int
    reward: float
    truthful_status_posted: bool
    root_cause_found: bool
    correct_fix_applied: bool
    system_health: float
    users_affected: int
    grader_payload: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)


def _call_state(env: Any) -> Any:
    """Return the environment state, supporting property or callable access patterns."""

    state_attr = getattr(env, "state", None)
    if callable(state_attr):
        return state_attr()
    return state_attr


def _normalize_observation(observation: Any) -> Any:
    """Validate and return the observation object expected by the baselines."""

    if hasattr(observation, "model_dump"):
        return observation
    raise TypeError("Environment returned an unsupported observation type")


def _extract_root_service(observation: Any, env: Any) -> str:
    """Infer the root-cause service from observation metadata or environment state."""

    metadata = getattr(observation, "metadata", {}) or {}
    root_service = metadata.get("root_cause_service")
    if root_service:
        return str(root_service)

    state = _call_state(env)
    root_service = getattr(state, "root_cause_service", "")
    return str(root_service or "")


def _infer_correct_fix(task_id: str) -> str:
    """Map each task identifier to its expected remediation action."""

    mapping = {
        "easy": "rollback",
        "medium": "restart_service",
        "hard": "scale_up",
    }
    return mapping.get(task_id, "rollback")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float, returning a fallback on conversion failure."""

    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int, returning a fallback on conversion failure."""

    try:
        return int(value)
    except Exception:
        return default


def _load_grader(task_id: str) -> Callable[[Any], Dict[str, Any]]:
    """Import and return the grader function for a given task identifier."""

    module = importlib.import_module(f"graders.{task_id}_grader")
    return getattr(module, "grade")


def _render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a simple markdown table from header and row values."""

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


class BaselineAgent:
    """Interface for agents that can propose the next action in an episode."""

    name: str

    def next_action(self, observation: Any, env: Any) -> Dict[str, Any]:
        """Return the next action payload for the current observation."""

        raise NotImplementedError


class HeuristicBaselineAgent(BaselineAgent):
    """Deterministic agent that follows a straightforward benchmark playbook."""

    name = "heuristic"

    def next_action(self, observation: Any, env: Any) -> Dict[str, Any]:
        """Choose the next action using simple state-based heuristics."""

        metadata = getattr(observation, "metadata", {}) or {}
        candidate_services = metadata.get("candidate_services", []) or []
        primary_service = candidate_services[0] if candidate_services else _extract_root_service(observation, env)
        task_id = metadata.get("task_id", "easy")
        correct_fix = _infer_correct_fix(str(task_id))

        if not getattr(observation, "root_cause_found", False):
            return {
                "action_type": "read_logs",
                "target": primary_service,
                "message": "",
                "metadata": {"policy": "heuristic_probe"},
            }

        if not getattr(observation, "truthful_status_posted", False):
            recovered = _safe_float(getattr(observation, "system_health", 0.0)) >= 0.99
            return {
                "action_type": "post_status_update",
                "target": "",
                "message": (
                    "resolved, service restored and healthy"
                    if recovered
                    else "investigating incident, service remains degraded while mitigation is in progress"
                ),
                "metadata": {"policy": "heuristic_truthful_status"},
            }

        if not getattr(observation, "correct_fix_applied", False):
            return {
                "action_type": correct_fix,
                "target": primary_service,
                "message": "",
                "metadata": {"policy": "heuristic_fix"},
            }

        if _safe_float(getattr(observation, "system_health", 0.0)) < 0.99:
            return {
                "action_type": "post_status_update",
                "target": "",
                "message": "mitigating incident, service is restoring but remains degraded",
                "metadata": {"policy": "heuristic_wait"},
            }

        return {
            "action_type": "post_status_update",
            "target": "",
            "message": "resolved, service restored and healthy",
            "metadata": {"policy": "heuristic_closeout"},
        }


class ReactiveMockAgent(BaselineAgent):
    """Rule-based agent that reacts to tool output before applying a remediation."""

    name = "reactive-mock"

    def next_action(self, observation: Any, env: Any) -> Dict[str, Any]:
        """Choose the next action based on observed tool output and state flags."""

        metadata = getattr(observation, "metadata", {}) or {}
        candidate_services = metadata.get("candidate_services", []) or []
        target = candidate_services[0] if candidate_services else _extract_root_service(observation, env)
        task_id = str(metadata.get("task_id", "easy"))

        if getattr(observation, "tool_output", None):
            joined = " ".join(observation.tool_output).lower()
            if "primary fault domain" in joined or "config mismatch" in joined or "signer" in joined or "replication lag" in joined:
                if not getattr(observation, "truthful_status_posted", False):
                    return {
                        "action_type": "post_status_update",
                        "target": "",
                        "message": "investigating and mitigating; service remains degraded",
                        "metadata": {"policy": "reactive_status"},
                    }
                if not getattr(observation, "correct_fix_applied", False):
                    return {
                        "action_type": _infer_correct_fix(task_id),
                        "target": target,
                        "message": "",
                        "metadata": {"policy": "reactive_fix"},
                    }

        if not getattr(observation, "root_cause_found", False):
            return {
                "action_type": "read_logs",
                "target": target,
                "message": "",
                "metadata": {"policy": "reactive_probe"},
            }

        if not getattr(observation, "correct_fix_applied", False):
            return {
                "action_type": _infer_correct_fix(task_id),
                "target": target,
                "message": "",
                "metadata": {"policy": "reactive_fix_direct"},
            }

        return {
            "action_type": "post_status_update",
            "target": "",
            "message": (
                "resolved, service restored and healthy"
                if _safe_float(getattr(observation, "system_health", 0.0)) >= 0.99
                else "mitigating incident, service still degraded while restoring capacity"
            ),
            "metadata": {"policy": "reactive_recovery_status"},
        }


class LLMJudgeAgent(BaselineAgent):
    """Baseline agent that delegates action selection to an LLM."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: Optional[str] = None) -> None:
        """Initialize the LLM-backed baseline with provider-specific client settings."""

        self.provider = provider
        self.model = model
        self.name = f"{provider}:{model}"
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def _build_prompt(self, observation: Any, env: Any) -> str:
        """Construct the prompt payload describing the current incident state."""

        state = _call_state(env)
        payload = {
            "task_id": getattr(state, "task_id", None),
            "tick_count": getattr(observation, "tick_count", 0),
            "system_health": getattr(observation, "system_health", 0.0),
            "users_affected": getattr(observation, "users_affected", 0),
            "active_alerts": getattr(observation, "active_alerts", []),
            "available_actions": getattr(observation, "available_actions", []),
            "root_cause_found": getattr(observation, "root_cause_found", False),
            "correct_fix_applied": getattr(observation, "correct_fix_applied", False),
            "truthful_status_posted": getattr(observation, "truthful_status_posted", False),
            "tool_output": getattr(observation, "tool_output", []),
            "metadata": getattr(observation, "metadata", {}),
        }
        return (
            "You are operating the Sev1Bench incident response benchmark.\n"
            "Choose exactly one next action.\n"
            "Priorities:\n"
            "1. Find the root cause with read_logs.\n"
            "2. Post a truthful status update while the incident is still active.\n"
            "3. Apply the correct remediation for the real failing service.\n"
            "4. Continue until the incident is actually restored.\n\n"
            "Return JSON only with exactly these keys:\n"
            '{"action_type":"...", "target":"...", "message":"...", "metadata":{}}\n\n'
            f"Observation:\n{payload}"
        )

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract and normalize an action payload from model output text."""

        text = (content or "").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Model output was not a JSON object")

        return {
            "action_type": str(data.get("action_type", "")).strip(),
            "target": str(data.get("target", "") or "").strip(),
            "message": str(data.get("message", "") or "").strip(),
            "metadata": data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        }

    def next_action(self, observation: Any, env: Any) -> Dict[str, Any]:
        """Request the next action from the configured LLM and parse the response."""

        prompt = self._build_prompt(observation, env)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful incident response agent. Return valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=250,
        )
        content = response.choices[0].message.content or ""
        return self._extract_json(content)


def _select_agents(mode: str, provider: str, model: Optional[str], base_url: Optional[str]) -> List[BaselineAgent]:
    """Build the list of baseline agents requested by the CLI arguments."""

    agents: List[BaselineAgent] = [HeuristicBaselineAgent(), ReactiveMockAgent()]

    if mode == "mock":
        return agents

    resolved_provider = provider.lower()
    if resolved_provider == "openai":
        api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "").strip()
        resolved_model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("API_BASE_URL")
    elif resolved_provider == "anthropic":
        api_key = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("API_KEY") or "").strip()
        resolved_model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1/"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not api_key:
        raise ValueError(f"Missing API key for provider '{provider}'")

    agents.append(
        LLMJudgeAgent(
            provider=resolved_provider,
            model=resolved_model,
            api_key=api_key,
            base_url=resolved_base_url,
        )
    )
    return agents


def _run_episode(task_id: str, agent: BaselineAgent, episode_index: int, max_steps: int) -> EpisodeTrace:
    """Run a single episode for one task-agent pair and grade the final observation."""

    env = IncidentResponseEnvironment(task_id=task_id, max_steps=max_steps)
    observation = _normalize_observation(env.reset())
    actions: List[Dict[str, Any]] = []

    while not getattr(observation, "done", False):
        action_payload = agent.next_action(observation, env)
        action = IncidentAction(**action_payload)
        actions.append(action.model_dump())
        observation = _normalize_observation(env.step(action))

    grader = _load_grader(task_id)
    grade_payload = grader(observation)

    return EpisodeTrace(
        task_id=task_id,
        agent_name=agent.name,
        episode_index=episode_index,
        passed=bool(grade_payload.get("passed", False)),
        resolved=bool(grade_payload.get("resolved", False)),
        step_count=_safe_int(grade_payload.get("step_count", getattr(observation, "tick_count", 0))),
        reward=_safe_float(grade_payload.get("reward", getattr(observation, "reward", 0.0))),
        truthful_status_posted=bool(grade_payload.get("truthful_status_posted", getattr(observation, "truthful_status_posted", False))),
        root_cause_found=bool(grade_payload.get("root_cause_found", getattr(observation, "root_cause_found", False))),
        correct_fix_applied=bool(grade_payload.get("correct_fix_applied", getattr(observation, "correct_fix_applied", False))),
        system_health=_safe_float(grade_payload.get("system_health", getattr(observation, "system_health", 0.0))),
        users_affected=_safe_int(grade_payload.get("users_affected", getattr(observation, "users_affected", 0))),
        grader_payload=grade_payload,
        actions=actions,
    )


def _aggregate_task_metrics(traces: Iterable[EpisodeTrace]) -> Dict[str, str]:
    """Compute aggregate display metrics for a collection of episode traces."""

    items = list(traces)
    if not items:
        return {
            "Episodes": "0",
            "Success Rate": "0.0%",
            "Avg Steps": "0.00",
            "Truthful Comm Rate": "0.0%",
            "Avg Final Score": "0.000",
            "Root Cause Rate": "0.0%",
            "Correct Fix Rate": "0.0%",
        }

    success_rate = 100.0 * sum(1 for item in items if item.passed) / len(items)
    truthful_rate = 100.0 * sum(1 for item in items if item.truthful_status_posted) / len(items)
    root_cause_rate = 100.0 * sum(1 for item in items if item.root_cause_found) / len(items)
    correct_fix_rate = 100.0 * sum(1 for item in items if item.correct_fix_applied) / len(items)
    avg_steps = statistics.mean(item.step_count for item in items)
    avg_score = statistics.mean(item.reward for item in items)

    return {
        "Episodes": str(len(items)),
        "Success Rate": f"{success_rate:.1f}%",
        "Avg Steps": f"{avg_steps:.2f}",
        "Truthful Comm Rate": f"{truthful_rate:.1f}%",
        "Avg Final Score": f"{avg_score:.3f}",
        "Root Cause Rate": f"{root_cause_rate:.1f}%",
        "Correct Fix Rate": f"{correct_fix_rate:.1f}%",
    }


def _build_results_markdown(all_traces: Sequence[EpisodeTrace]) -> str:
    """Build the README-ready markdown table summarizing baseline performance."""

    rows: List[List[str]] = []
    grouped: Dict[Tuple[str, str], List[EpisodeTrace]] = {}
    for trace in all_traces:
        grouped.setdefault((trace.agent_name, trace.task_id), []).append(trace)

    ordered_keys = sorted(grouped.keys(), key=lambda item: (item[0], TASK_IDS.index(item[1]) if item[1] in TASK_IDS else 999))
    for agent_name, task_id in ordered_keys:
        metrics = _aggregate_task_metrics(grouped[(agent_name, task_id)])
        rows.append(
            [
                agent_name,
                task_id,
                metrics["Episodes"],
                metrics["Success Rate"],
                metrics["Avg Steps"],
                metrics["Truthful Comm Rate"],
                metrics["Root Cause Rate"],
                metrics["Correct Fix Rate"],
                metrics["Avg Final Score"],
            ]
        )

    headers = [
        "Agent",
        "Task",
        "Episodes",
        "Success Rate",
        "Avg Steps",
        "Truthful Communication Rate",
        "Root Cause Rate",
        "Correct Fix Rate",
        "Avg Final Score",
    ]
    return _render_markdown_table(headers, rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the baseline runner."""

    parser = argparse.ArgumentParser(description="Run local Sev1Bench baseline agents and print README-ready markdown.")
    parser.add_argument("--mode", choices=("mock", "llm"), default="mock", help="Use only built-in baselines or also include an LLM-driven baseline.")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai", help="LLM provider when --mode llm is selected.")
    parser.add_argument("--model", default=None, help="Optional model override for the LLM baseline.")
    parser.add_argument("--base-url", default=None, help="Optional API base URL override.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Episodes to run per task per agent.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode.")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_IDS), help="Task IDs to run.")
    return parser.parse_args()


def main() -> int:
    """Execute the selected baselines and print a markdown summary to stdout."""

    args = parse_args()
    selected_tasks = [task_id for task_id in args.tasks if task_id in TASK_IDS]
    if not selected_tasks:
        raise ValueError("No valid task IDs selected")

    global TASK_IDS
    TASK_IDS = tuple(selected_tasks)

    agents = _select_agents(
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )

    traces: List[EpisodeTrace] = []
    for agent in agents:
        for task_id in selected_tasks:
            for episode_index in range(1, args.episodes + 1):
                traces.append(_run_episode(task_id=task_id, agent=agent, episode_index=episode_index, max_steps=args.max_steps))

    markdown = _build_results_markdown(traces)

    print("# Sev1Bench Baseline Results")
    print()
    print(f"- Mode: `{args.mode}`")
    print(f"- Episodes per task: `{args.episodes}`")
    print(f"- Tasks: `{', '.join(selected_tasks)}`")
    print()
    print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())