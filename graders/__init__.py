"""Central grader registry for Sev1Bench validation and runtime imports."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .easy_grader import grade as grade_easy
from .expert_grader import grade as grade_expert
from .hard_grader import grade as grade_hard
from .medium_grader import grade as grade_medium

GraderFn = Callable[[Any], Dict[str, Any]]

GRADER_REGISTRY: Dict[str, GraderFn] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "expert": grade_expert,
}

__all__ = [
    "GraderFn",
    "GRADER_REGISTRY",
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "grade_expert",
]
