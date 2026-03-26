"""Tests for the baseline agents (heuristic)."""

import pytest
from app.baseline import HeuristicAgent, run_baseline_single


class TestHeuristicAgent:
    def test_complete_task_when_ready(self):
        obs = {
            "own_resources": {"storage": 5, "api_credits": 4},
            "target_resources": {"storage": 4, "api_credits": 3},
            "resource_progress": {"storage": 1.0, "api_credits": 1.0},
            "visible_agents": [],
            "pending_proposals": [],
            "task_id": "resource_acquisition",
        }
        agent = HeuristicAgent()
        action = agent.decide(obs)
        assert action.action_type.value == "COMPLETE_TASK"

    def test_rejects_low_trust_proposal(self):
        obs = {
            "own_resources": {"compute": 5},
            "target_resources": {"storage": 3},
            "resource_progress": {"storage": 0.0},
            "visible_agents": [
                {
                    "agent_id": "npc_x",
                    "trust_score": 0.1,
                    "reputation_score": 0.1,
                    "exploitation_events": 0,
                    "visible_resources": {"storage": 5},
                }
            ],
            "pending_proposals": [
                {
                    "proposal_id": "p1",
                    "from_agent": "npc_x",
                    "offer": {"storage": 3},
                    "request": {"compute": 1},
                }
            ],
            "task_id": "resource_acquisition",
        }
        agent = HeuristicAgent()
        action = agent.decide(obs)
        assert action.action_type.value == "REJECT_TRADE"

    def test_rejects_exploiter_proposal(self):
        obs = {
            "own_resources": {"compute": 5},
            "target_resources": {"storage": 3},
            "resource_progress": {"storage": 0.0},
            "visible_agents": [
                {
                    "agent_id": "npc_x",
                    "trust_score": 0.7,
                    "reputation_score": 0.3,
                    "exploitation_events": 1,  # has exploited before!
                    "visible_resources": {"storage": 5},
                }
            ],
            "pending_proposals": [
                {
                    "proposal_id": "p1",
                    "from_agent": "npc_x",
                    "offer": {"storage": 5},
                    "request": {"compute": 1},
                }
            ],
            "task_id": "resource_acquisition",
        }
        agent = HeuristicAgent()
        action = agent.decide(obs)
        assert action.action_type.value == "REJECT_TRADE"

    def test_proposes_trade_for_needed(self):
        obs = {
            "own_resources": {"compute": 5, "data": 0},
            "target_resources": {"compute": 2, "data": 3},
            "resource_progress": {"compute": 1.0, "data": 0.0},
            "visible_agents": [
                {
                    "agent_id": "npc_a",
                    "trust_score": 0.7,
                    "reputation_score": 0.8,
                    "exploitation_events": 0,
                    "visible_resources": {"data": 5},
                    "is_allied": False,
                }
            ],
            "pending_proposals": [],
            "task_id": "resource_acquisition",
        }
        agent = HeuristicAgent()
        action = agent.decide(obs)
        assert action.action_type.value == "PROPOSE_TRADE"
        assert action.target_agent == "npc_a"
        assert "data" in (action.request_resources or {})


class TestBaselineRunner:
    @pytest.mark.parametrize("task_id", [
        "resource_acquisition",
        "coalition_building",
        "adversarial_resilience",
        "market_manipulation",
    ])
    def test_heuristic_baseline_runs(self, task_id):
        result = run_baseline_single(task_id, seed=42, use_llm=False)
        assert 0.0 <= result.score <= 1.0
        assert result.steps_taken > 0
        assert result.task_id == task_id

    def test_heuristic_is_deterministic(self):
        r1 = run_baseline_single("resource_acquisition", seed=42)
        r2 = run_baseline_single("resource_acquisition", seed=42)
        assert r1.score == r2.score
        assert r1.steps_taken == r2.steps_taken
        assert r1.cumulative_reward == r2.cumulative_reward
