"""Pydantic models for the Social Intelligence Substrate environment.

All OpenEnv-compliant typed models:
  - Observation: what the learning agent sees each step
  - Action: structured economic actions the agent can take
  - Reward: dense signal with component breakdown
  - StepResult, EnvironmentState, and API response helpers
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Available structured economic actions."""
    PROPOSE_TRADE = "PROPOSE_TRADE"
    ACCEPT_TRADE = "ACCEPT_TRADE"
    REJECT_TRADE = "REJECT_TRADE"
    FORM_ALLIANCE = "FORM_ALLIANCE"
    BREAK_ALLIANCE = "BREAK_ALLIANCE"
    COMPLETE_TASK = "COMPLETE_TASK"
    OBSERVE = "OBSERVE"


class AgentPersonality(str, Enum):
    """NPC agent personality archetypes."""
    HONEST = "honest"
    SELFISH = "selfish"
    MALICIOUS = "malicious"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class TradeProposal(BaseModel):
    """A pending trade proposal between two agents."""
    proposal_id: str
    from_agent: str
    to_agent: str
    offer: Dict[str, int] = Field(
        description="Resources being offered: {resource_name: quantity}"
    )
    request: Dict[str, int] = Field(
        description="Resources requested in return: {resource_name: quantity}"
    )
    step_created: int


class TradeRecord(BaseModel):
    """Record of a past trade with an agent."""
    step: int
    counterparty: str
    gave: Dict[str, int] = Field(default_factory=dict)
    received: Dict[str, int] = Field(default_factory=dict)
    outcome: str = Field(description="success | rejected | exploitation | failed")


class AgentInfo(BaseModel):
    """Information about a visible agent in the marketplace."""
    agent_id: str
    visible_resources: Dict[str, int] = Field(
        description="Resources this agent appears to have (may be faked by malicious agents)"
    )
    trust_score: float = Field(ge=0.0, le=1.0, description="Your trust level with this agent")
    is_allied: bool = Field(description="Whether you have an active alliance")
    interaction_count: int = Field(default=0, description="Total past interactions")
    successful_trades: int = Field(default=0, description="Number of successful trades with this agent")
    failed_trades: int = Field(default=0, description="Number of failed/rejected trades with this agent")
    exploitation_events: int = Field(default=0, description="Number of times this agent exploited you")
    reputation_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Computed reputation: ratio of successful interactions to total interactions"
    )


# ---------------------------------------------------------------------------
# Core OpenEnv Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Full observation returned to the learning agent each step.

    Contains partial-visibility into the social graph:
    visible agents (with potentially faked resources for malicious ones),
    trust scores, pending trade proposals, alliance status, and more.
    """
    agent_id: str = "player"
    step_number: int
    max_steps: int
    task_id: str
    task_description: str
    own_resources: Dict[str, int] = Field(
        description="Agent's current resource inventory"
    )
    target_resources: Dict[str, int] = Field(
        description="Resources required to complete the task"
    )
    resource_progress: Dict[str, float] = Field(
        description="Progress toward each target resource (0.0 – 1.0)"
    )
    visible_agents: List[AgentInfo] = Field(
        description="Other agents visible in the marketplace"
    )
    pending_proposals: List[TradeProposal] = Field(
        description="Incoming trade proposals awaiting your decision"
    )
    active_alliances: List[str] = Field(
        description="IDs of agents you are currently allied with"
    )
    trade_history: List[TradeRecord] = Field(
        default_factory=list,
        description="Chronological record of past trades (last 20)",
    )
    market_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate market statistics: total_trades, exploitation_rate, avg_trust",
    )
    recent_events: List[str] = Field(
        description="Chronological log of recent marketplace events"
    )


class Action(BaseModel):
    """Structured economic action submitted by the learning agent.

    Only action_type is always required.  Additional fields depend on the
    action_type chosen (see /tasks endpoint for the schema).
    """
    action_type: ActionType
    target_agent: Optional[str] = Field(
        default=None, description="ID of the agent this action targets"
    )
    offer_resources: Optional[Dict[str, int]] = Field(
        default=None, description="Resources to offer in a trade"
    )
    request_resources: Optional[Dict[str, int]] = Field(
        default=None, description="Resources to request in a trade"
    )
    proposal_id: Optional[str] = Field(
        default=None, description="ID of a pending proposal (for ACCEPT/REJECT)"
    )


class Reward(BaseModel):
    """Dense reward signal with a component-level breakdown."""
    value: float = Field(description="Total reward for this step")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Named components that sum to *value*",
    )
    message: str = Field(default="", description="Human-readable event summary")


# ---------------------------------------------------------------------------
# Composite / Response Models
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Result of a single environment step — implements OpenEnv step() return."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Full internal state snapshot — implements OpenEnv state()."""
    graph_nodes: List[Dict[str, Any]]
    graph_edges: List[Dict[str, Any]]
    current_step: int
    max_steps: int
    task_id: str
    episode_complete: bool
    cumulative_reward: float
    metrics: Dict[str, float]


class TaskInfo(BaseModel):
    """Metadata for a single task — returned by /tasks endpoint."""
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    resource_types: List[str]
    action_schema: List[Dict[str, Any]]


class BaselineResult(BaseModel):
    """Score report for one task — returned by /baseline endpoint."""
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    steps_taken: int
    cumulative_reward: float
    details: Dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    """Deterministic grader output — returned by /grader endpoint."""
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    passed: bool
