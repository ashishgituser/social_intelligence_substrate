"""Deterministic grader for each task.

Every grader produces a score in [0.0, 1.0] from *only* the episode
metrics dict (no randomness, no model calls).  Weights differ per
difficulty level to emphasise what matters most for that task.
"""

from __future__ import annotations

from typing import Any, Dict

from app.models import GraderResult


def grade_episode(
    task_id: str,
    metrics: Dict[str, Any],
    task_complete: bool,
) -> GraderResult:
    """Route to the appropriate task grader and return a ``GraderResult``."""
    graders = {
        "resource_acquisition": _grade_easy,
        "coalition_building": _grade_medium,
        "adversarial_resilience": _grade_hard,
        "market_manipulation": _grade_expert,
    }
    fn = graders.get(task_id)
    if fn is None:
        return GraderResult(task_id=task_id, score=0.0, breakdown={}, passed=False)
    return fn(task_id, metrics, task_complete)


# ------------------------------------------------------------------
# Easy — Resource Acquisition
# ------------------------------------------------------------------
def _grade_easy(
    task_id: str, m: Dict[str, Any], task_complete: bool
) -> GraderResult:
    """
    60 %  task_completion  (did you acquire the resources?)
    25 %  efficiency       (fewer steps → higher)
    15 %  social_capital   (trade success rate)
    """
    tc = float(m.get("task_completion", 0))
    eff = float(m.get("efficiency", 0))
    sc = float(m.get("social_capital", 0))

    bd = {
        "task_completion": round(tc * 0.60, 4),
        "efficiency": round(eff * 0.25, 4),
        "social_capital": round(sc * 0.15, 4),
    }
    score = _clamp(sum(bd.values()))
    return GraderResult(
        task_id=task_id, score=score, breakdown=bd, passed=score >= 0.30
    )


# ------------------------------------------------------------------
# Medium — Coalition Building
# ------------------------------------------------------------------
def _grade_medium(
    task_id: str, m: Dict[str, Any], task_complete: bool
) -> GraderResult:
    """
    45 %  task_completion
    25 %  alliance_quality  (alliances formed / 2)
    20 %  efficiency
    10 %  social_capital
    """
    tc = float(m.get("task_completion", 0))
    eff = float(m.get("efficiency", 0))
    sc = float(m.get("social_capital", 0))
    af = float(m.get("alliances_formed", 0))
    aq = min(1.0, af / 2.0)

    bd = {
        "task_completion": round(tc * 0.45, 4),
        "alliance_quality": round(aq * 0.25, 4),
        "efficiency": round(eff * 0.20, 4),
        "social_capital": round(sc * 0.10, 4),
    }
    score = _clamp(sum(bd.values()))
    return GraderResult(
        task_id=task_id, score=score, breakdown=bd, passed=score >= 0.25
    )


# ------------------------------------------------------------------
# Hard — Adversarial Resilience
# ------------------------------------------------------------------
def _grade_hard(
    task_id: str, m: Dict[str, Any], task_complete: bool
) -> GraderResult:
    """
    35 %  task_completion
    30 %  robustness       (avoiding exploitation)
    20 %  efficiency
    15 %  social_capital
    """
    tc = float(m.get("task_completion", 0))
    eff = float(m.get("efficiency", 0))
    sc = float(m.get("social_capital", 0))
    rob = float(m.get("robustness", 0))

    bd = {
        "task_completion": round(tc * 0.35, 4),
        "robustness": round(rob * 0.30, 4),
        "efficiency": round(eff * 0.20, 4),
        "social_capital": round(sc * 0.15, 4),
    }
    score = _clamp(sum(bd.values()))
    return GraderResult(
        task_id=task_id, score=score, breakdown=bd, passed=score >= 0.20
    )


# ------------------------------------------------------------------
# Expert — Market Manipulation
# ------------------------------------------------------------------
def _grade_expert(
    task_id: str, m: Dict[str, Any], task_complete: bool
) -> GraderResult:
    """
    30 %  task_completion    (resource acquisition despite hostile market)
    25 %  robustness         (avoiding exploitation from colluding agents)
    20 %  efficiency         (fewer steps under heavy adversarial pressure)
    15 %  social_capital     (building genuine relationships)
    10 %  alliance_quality   (strategic alliance formation)
    """
    tc = float(m.get("task_completion", 0))
    eff = float(m.get("efficiency", 0))
    sc = float(m.get("social_capital", 0))
    rob = float(m.get("robustness", 0))
    af = float(m.get("alliances_formed", 0))
    aq = min(1.0, af / 2.0)

    bd = {
        "task_completion": round(tc * 0.30, 4),
        "robustness": round(rob * 0.25, 4),
        "efficiency": round(eff * 0.20, 4),
        "social_capital": round(sc * 0.15, 4),
        "alliance_quality": round(aq * 0.10, 4),
    }
    score = _clamp(sum(bd.values()))
    return GraderResult(
        task_id=task_id, score=score, breakdown=bd, passed=score >= 0.15
    )


# ------------------------------------------------------------------
def _clamp(v: float) -> float:
    return max(0.0, min(1.0, round(v, 4)))
