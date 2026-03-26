"""FastAPI server — OpenEnv-compliant HTTP interface.

Endpoints:
  Core:
    POST /reset   → Observation
    POST /step    → StepResult
    GET  /state   → EnvironmentState

  Required additions:
    GET  /tasks    → List[TaskInfo]
    GET  /grader   → GraderResult
    POST /baseline → List[BaselineResult]

  Health:
    GET  /        → basic status (responds 200 for automated pings)
"""

from __future__ import annotations

import time
import traceback
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.environment import SocialIntelligenceEnv
from app.grader import grade_episode
from app.models import (
    Action,
    BaselineResult,
    EnvironmentState,
    GraderResult,
    Observation,
    StepResult,
    TaskInfo,
)
from app.tasks import RESOURCE_TYPES, TASK_CONFIGS

# ==================================================================
# App
# ==================================================================

app = FastAPI(
    title="Social Intelligence Substrate",
    description=(
        "A graph-native OpenEnv environment for evaluating AI social "
        "intelligence through structured economic interactions in "
        "decentralised resource markets."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory state ----
_env: Optional[SocialIntelligenceEnv] = None
_last_grader: Optional[GraderResult] = None

# ---- Rate limiting for /baseline (protects OpenAI API key) ----
_baseline_last_call: float = 0.0
_BASELINE_COOLDOWN: int = 60  # seconds between allowed calls


# ==================================================================
# Request schemas
# ==================================================================

class ResetRequest(BaseModel):
    task_id: str = "resource_acquisition"
    seed: int = 42


# ==================================================================
# Health
# ==================================================================

@app.get("/")
def root():
    return {
        "name": "Social Intelligence Substrate",
        "version": "1.0.0",
        "status": "running",
        "spec": "openenv",
    }


# ==================================================================
# OpenEnv core
# ==================================================================

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    global _env, _last_grader
    try:
        _env = SocialIntelligenceEnv(task_id=req.task_id, seed=req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _last_grader = None
    return _env.reset()


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    global _last_grader
    if _env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialised. Call /reset first."
        )
    result = _env.step(action)
    if result.done:
        metrics = _env._compute_metrics()
        _last_grader = grade_episode(
            _env.task_id, metrics, _env._check_task_complete()
        )
    return result


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    if _env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialised. Call /reset first."
        )
    return _env.state()


# ==================================================================
# Additional required endpoints
# ==================================================================

_ACTION_SCHEMA = [
    {
        "action_type": "PROPOSE_TRADE",
        "required_fields": ["target_agent", "offer_resources", "request_resources"],
        "description": "Propose a resource trade to another agent",
        "example": {
            "action_type": "PROPOSE_TRADE",
            "target_agent": "provider_alpha",
            "offer_resources": {"compute": 2},
            "request_resources": {"storage": 3},
        },
    },
    {
        "action_type": "ACCEPT_TRADE",
        "required_fields": ["proposal_id"],
        "description": "Accept a pending trade proposal",
        "example": {
            "action_type": "ACCEPT_TRADE",
            "proposal_id": "npc_prop_0",
        },
    },
    {
        "action_type": "REJECT_TRADE",
        "required_fields": ["proposal_id"],
        "description": "Reject a pending trade proposal",
        "example": {
            "action_type": "REJECT_TRADE",
            "proposal_id": "npc_prop_0",
        },
    },
    {
        "action_type": "FORM_ALLIANCE",
        "required_fields": ["target_agent"],
        "description": "Request to form a resource-sharing alliance",
        "example": {
            "action_type": "FORM_ALLIANCE",
            "target_agent": "provider_alpha",
        },
    },
    {
        "action_type": "BREAK_ALLIANCE",
        "required_fields": ["target_agent"],
        "description": "Break an existing alliance",
        "example": {
            "action_type": "BREAK_ALLIANCE",
            "target_agent": "provider_alpha",
        },
    },
    {
        "action_type": "COMPLETE_TASK",
        "required_fields": [],
        "description": "Attempt to complete the task (checks resource requirements)",
        "example": {"action_type": "COMPLETE_TASK"},
    },
    {
        "action_type": "OBSERVE",
        "required_fields": [],
        "description": "Observe the marketplace without acting",
        "example": {"action_type": "OBSERVE"},
    },
]


@app.get("/tasks", response_model=List[TaskInfo])
def tasks() -> List[TaskInfo]:
    """Return available tasks with their action schema."""
    return [
        TaskInfo(
            task_id=tid,
            name=cfg["name"],
            difficulty=cfg["difficulty"],
            description=cfg["description"],
            max_steps=cfg["max_steps"],
            resource_types=RESOURCE_TYPES,
            action_schema=_ACTION_SCHEMA,
        )
        for tid, cfg in TASK_CONFIGS.items()
    ]


@app.get("/grader", response_model=GraderResult)
def grader() -> GraderResult:
    """Return the grader score for the most recently completed episode."""
    global _last_grader
    if _last_grader is not None:
        return _last_grader
    if _env is None:
        raise HTTPException(status_code=400, detail="No episode has been run yet.")
    if not _env.done:
        raise HTTPException(status_code=400, detail="Episode still in progress.")
    metrics = _env._compute_metrics()
    _last_grader = grade_episode(
        _env.task_id, metrics, _env._check_task_complete()
    )
    return _last_grader


@app.post("/baseline", response_model=List[BaselineResult])
def baseline() -> List[BaselineResult]:
    """Run baseline inference on all tasks and return scores."""
    global _baseline_last_call
    now = time.time()
    if now - _baseline_last_call < _BASELINE_COOLDOWN:
        wait = int(_BASELINE_COOLDOWN - (now - _baseline_last_call))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited. Try again in {wait}s.",
        )
    _baseline_last_call = now

    from app.baseline import run_baseline_all_tasks

    try:
        return run_baseline_all_tasks()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline failed: {exc}\n{traceback.format_exc()}",
        )
