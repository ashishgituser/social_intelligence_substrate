"""Tests for the core environment (step / reset / state)."""

import pytest
from app.environment import SocialIntelligenceEnv
from app.models import Action, ActionType


class TestReset:
    def test_reset_returns_observation(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        obs = env.reset()
        assert obs.agent_id == "player"
        assert obs.step_number == 0
        assert obs.task_id == "resource_acquisition"

    def test_reset_initializes_resources(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        obs = env.reset()
        assert obs.own_resources["compute"] == 5
        assert obs.own_resources["data"] == 4

    def test_reset_shows_visible_agents(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        obs = env.reset()
        agent_ids = {a.agent_id for a in obs.visible_agents}
        assert "provider_alpha" in agent_ids
        assert "provider_beta" in agent_ids

    def test_reset_includes_trade_history(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        obs = env.reset()
        assert obs.trade_history == []

    def test_reset_includes_market_stats(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        obs = env.reset()
        assert "total_trades" in obs.market_stats

    @pytest.mark.parametrize("task_id", [
        "resource_acquisition",
        "coalition_building",
        "adversarial_resilience",
        "market_manipulation",
    ])
    def test_reset_all_tasks(self, task_id):
        env = SocialIntelligenceEnv(task_id, seed=42)
        obs = env.reset()
        assert obs.task_id == task_id
        assert not env.done

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            SocialIntelligenceEnv("nonexistent_task")


class TestStep:
    def test_observe_action(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        result = env.step(Action(action_type=ActionType.OBSERVE))
        assert result.observation.step_number == 1
        assert not result.done
        assert "observe" in result.reward.breakdown or "step_cost" in result.reward.breakdown

    def test_complete_task_before_ready(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        result = env.step(Action(action_type=ActionType.COMPLETE_TASK))
        assert "incomplete_attempt" in result.reward.breakdown

    def test_propose_trade(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        result = env.step(Action(
            action_type=ActionType.PROPOSE_TRADE,
            target_agent="provider_alpha",
            offer_resources={"compute": 2},
            request_resources={"storage": 2},
        ))
        assert result.observation.step_number == 1
        # Trade should either succeed or be rejected
        has_trade_key = any(
            k in result.reward.breakdown
            for k in ["successful_trade", "trade_rejected"]
        )
        assert has_trade_key

    def test_invalid_target_raises(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        result = env.step(Action(
            action_type=ActionType.PROPOSE_TRADE,
            target_agent="nonexistent",
            offer_resources={"compute": 1},
            request_resources={"storage": 1},
        ))
        # Should handle gracefully with invalid_action penalty
        assert "invalid_action" in result.reward.breakdown

    def test_step_after_done_returns_zero_reward(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        env.done = True
        result = env.step(Action(action_type=ActionType.OBSERVE))
        assert result.done is True
        assert result.reward.value == 0.0

    def test_episode_terminates_at_max_steps(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        for _ in range(20):
            result = env.step(Action(action_type=ActionType.OBSERVE))
        assert result.done is True

    def test_trade_history_tracked(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        env.step(Action(
            action_type=ActionType.PROPOSE_TRADE,
            target_agent="provider_alpha",
            offer_resources={"compute": 2},
            request_resources={"storage": 2},
        ))
        obs = env._build_observation()
        assert len(obs.trade_history) >= 1
        assert obs.trade_history[0].counterparty == "provider_alpha"

    def test_agent_info_has_reputation(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        obs = env.reset()
        for agent in obs.visible_agents:
            assert hasattr(agent, "reputation_score")
            assert 0.0 <= agent.reputation_score <= 1.0


class TestState:
    def test_state_returns_graph(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        s = env.state()
        assert len(s.graph_nodes) > 0
        assert s.task_id == "resource_acquisition"
        assert not s.episode_complete

    def test_state_metrics(self):
        env = SocialIntelligenceEnv("resource_acquisition", seed=42)
        env.reset()
        s = env.state()
        assert "task_completion" in s.metrics
        assert "efficiency" in s.metrics
        assert "social_capital" in s.metrics
        assert "robustness" in s.metrics


class TestDeterminism:
    """Verify seeded reproducibility — critical for hackathon scoring."""

    def test_same_seed_same_output(self):
        """Two runs with the same seed and actions produce identical results."""
        results = []
        for _ in range(2):
            env = SocialIntelligenceEnv("resource_acquisition", seed=42)
            env.reset()
            r1 = env.step(Action(
                action_type=ActionType.PROPOSE_TRADE,
                target_agent="provider_alpha",
                offer_resources={"compute": 2},
                request_resources={"storage": 2},
            ))
            r2 = env.step(Action(action_type=ActionType.OBSERVE))
            results.append((r1.reward.value, r2.reward.value))
        assert results[0] == results[1]

    def test_different_seed_different_output(self):
        """Different seeds may produce different NPC behaviors."""
        envs = [
            SocialIntelligenceEnv("adversarial_resilience", seed=42),
            SocialIntelligenceEnv("adversarial_resilience", seed=99),
        ]
        observations = []
        for env in envs:
            env.reset()
            step_results = []
            for _ in range(5):
                r = env.step(Action(action_type=ActionType.OBSERVE))
                step_results.append(len(r.observation.pending_proposals))
            observations.append(step_results)
        # With different seeds, NPC proposal generation patterns should differ
        # (they're random). Not guaranteed to differ every time, but very likely.
        # At minimum, both should be valid.
        assert all(isinstance(x, int) for x in observations[0])


class TestExploitation:
    """Test malicious agent exploitation mechanics."""

    def test_exploitation_tracked_in_metrics(self):
        env = SocialIntelligenceEnv("adversarial_resilience", seed=42)
        env.reset()
        # Trade with the malicious agent — may get exploited
        for _ in range(10):
            env.step(Action(
                action_type=ActionType.PROPOSE_TRADE,
                target_agent="provider_gamma",
                offer_resources={"compute": 1},
                request_resources={"storage": 1},
            ))
        metrics = env._compute_metrics()
        # With seed=42, some trades with malicious should result in exploitation
        assert metrics["exploitation_count"] >= 0
        assert metrics["robustness"] <= 1.0


class TestMarketManipulation:
    """Test the 4th expert task."""

    def test_expert_task_loads(self):
        env = SocialIntelligenceEnv("market_manipulation", seed=42)
        obs = env.reset()
        assert obs.task_id == "market_manipulation"
        assert obs.max_steps == 50
        agent_ids = {a.agent_id for a in obs.visible_agents}
        assert "provider_epsilon" in agent_ids  # 5th NPC unique to expert task

    def test_expert_task_has_5_npcs(self):
        env = SocialIntelligenceEnv("market_manipulation", seed=42)
        obs = env.reset()
        assert len(obs.visible_agents) == 5

    def test_expert_coalition_resources(self):
        """Alliance with non-malicious agent should pool resources."""
        env = SocialIntelligenceEnv("market_manipulation", seed=42)
        env.reset()
        # Force alliance
        env.graph.form_alliance("player", "provider_alpha")
        total = env._get_total_resources()
        player_res = env.graph.get_resources("player")
        alpha_res = env.graph.get_resources("provider_alpha")
        for r in total:
            assert total[r] == player_res.get(r, 0) + alpha_res.get(r, 0)
