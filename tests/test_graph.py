"""Tests for the SocialGraph engine."""

import pytest
from app.graph import SocialGraph


@pytest.fixture
def graph():
    g = SocialGraph()
    g.add_agent("player", "player", {"compute": 5, "data": 3})
    g.add_agent("npc_a", "honest", {"compute": 2, "storage": 4})
    g.add_agent(
        "npc_b", "malicious",
        {"compute": 1, "storage": 1},
        fake_resources={"compute": 10, "storage": 10},
    )
    return g


class TestAgentManagement:
    def test_add_agent(self, graph):
        assert "player" in graph.get_all_agents()
        assert "npc_a" in graph.get_all_agents()
        assert "npc_b" in graph.get_all_agents()

    def test_get_all_agents_exclude(self, graph):
        agents = graph.get_all_agents(exclude="player")
        assert "player" not in agents
        assert "npc_a" in agents

    def test_get_agent_type(self, graph):
        assert graph.get_agent_type("player") == "player"
        assert graph.get_agent_type("npc_a") == "honest"
        assert graph.get_agent_type("npc_b") == "malicious"


class TestResources:
    def test_get_resources(self, graph):
        res = graph.get_resources("player")
        assert res == {"compute": 5, "data": 3}

    def test_get_resources_returns_copy(self, graph):
        res = graph.get_resources("player")
        res["compute"] = 999
        assert graph.get_resources("player")["compute"] == 5

    def test_visible_resources_honest(self, graph):
        vis = graph.get_visible_resources("npc_a")
        assert vis == {"compute": 2, "storage": 4}

    def test_visible_resources_malicious_shows_fake(self, graph):
        vis = graph.get_visible_resources("npc_b")
        assert vis == {"compute": 10, "storage": 10}

    def test_has_resources_true(self, graph):
        assert graph.has_resources("player", {"compute": 3})

    def test_has_resources_false(self, graph):
        assert not graph.has_resources("player", {"compute": 99})

    def test_transfer_resources_success(self, graph):
        result = graph.transfer_resources("player", "npc_a", {"compute": 2})
        assert result is True
        assert graph.get_resources("player")["compute"] == 3
        assert graph.get_resources("npc_a")["compute"] == 4

    def test_transfer_resources_insufficient(self, graph):
        result = graph.transfer_resources("player", "npc_a", {"compute": 99})
        assert result is False
        assert graph.get_resources("player")["compute"] == 5

    def test_add_resources(self, graph):
        graph.add_resources("player", {"storage": 3})
        assert graph.get_resources("player")["storage"] == 3

    def test_remove_resources(self, graph):
        result = graph.remove_resources("player", {"compute": 2})
        assert result is True
        assert graph.get_resources("player")["compute"] == 3


class TestTrust:
    def test_initial_trust_zero(self, graph):
        assert graph.get_trust("player", "npc_a") == 0.0

    def test_set_and_get_trust(self, graph):
        graph.set_trust("player", "npc_a", 0.8)
        assert graph.get_trust("player", "npc_a") == 0.8

    def test_trust_clamped_high(self, graph):
        graph.set_trust("player", "npc_a", 1.5)
        assert graph.get_trust("player", "npc_a") == 1.0

    def test_trust_clamped_low(self, graph):
        graph.set_trust("player", "npc_a", -0.5)
        assert graph.get_trust("player", "npc_a") == 0.0

    def test_update_trust(self, graph):
        graph.set_trust("player", "npc_a", 0.5)
        new = graph.update_trust("player", "npc_a", 0.2)
        assert new == pytest.approx(0.7)
        assert graph.get_trust("player", "npc_a") == pytest.approx(0.7)


class TestAlliances:
    def test_form_alliance(self, graph):
        graph.form_alliance("player", "npc_a")
        assert graph.is_allied("player", "npc_a")
        assert graph.is_allied("npc_a", "player")

    def test_break_alliance(self, graph):
        graph.form_alliance("player", "npc_a")
        graph.break_alliance("player", "npc_a")
        assert not graph.is_allied("player", "npc_a")

    def test_get_allies(self, graph):
        graph.form_alliance("player", "npc_a")
        allies = graph.get_allies("player")
        assert "npc_a" in allies

    def test_no_alliance_by_default(self, graph):
        assert not graph.is_allied("player", "npc_a")


class TestTradeStats:
    def test_record_successful_trade(self, graph):
        graph.record_successful_trade("npc_a")
        stats = graph.get_agent_stats("npc_a")
        assert stats["successful_trades"] == 1

    def test_record_failed_trade(self, graph):
        graph.record_failed_trade("npc_a")
        stats = graph.get_agent_stats("npc_a")
        assert stats["failed_trades"] == 1

    def test_record_exploitation(self, graph):
        graph.record_exploitation("npc_b")
        stats = graph.get_agent_stats("npc_b")
        assert stats["exploitation_events"] == 1

    def test_reputation_no_interactions(self, graph):
        rep = graph.get_reputation("npc_a")
        assert rep == 0.5

    def test_reputation_after_trades(self, graph):
        graph.increment_interaction("npc_a")
        graph.increment_interaction("npc_a")
        graph.record_successful_trade("npc_a")
        graph.record_successful_trade("npc_a")
        rep = graph.get_reputation("npc_a")
        assert rep == 1.0  # 2 successful / 2 total

    def test_reputation_penalizes_exploitation(self, graph):
        graph.increment_interaction("npc_b")
        graph.record_exploitation("npc_b")
        rep = graph.get_reputation("npc_b")
        assert rep == 0.0  # exploitation heavily penalized


class TestSerialization:
    def test_get_state(self, graph):
        nodes, edges = graph.get_state()
        assert len(nodes) == 3
        ids = {n["id"] for n in nodes}
        assert ids == {"player", "npc_a", "npc_b"}
