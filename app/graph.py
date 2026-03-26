"""NetworkX-based social graph engine.

The graph is the environment:
  - Nodes: agents (player + NPCs) with resource inventories
  - Edges: directed trust relationships and alliance flags

All state mutations (trades, trust updates, alliances) go through this
module so the rest of the codebase stays clean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


class SocialGraph:
    """Manages the social network of agents and their relationships."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node (Agent) management
    # ------------------------------------------------------------------

    def add_agent(
        self,
        agent_id: str,
        agent_type: str,
        resources: Dict[str, int],
        fake_resources: Optional[Dict[str, int]] = None,
    ) -> None:
        """Register an agent node in the graph."""
        self.graph.add_node(
            agent_id,
            node_type="agent",
            agent_type=agent_type,
            resources=dict(resources),
            fake_resources=dict(fake_resources) if fake_resources else None,
            interaction_count=0,
            successful_trades=0,
            failed_trades=0,
            exploitation_events=0,
        )

    def get_agent_type(self, agent_id: str) -> str:
        return self.graph.nodes[agent_id]["agent_type"]

    def get_all_agents(self, exclude: Optional[str] = None) -> List[str]:
        """Return IDs of all agent nodes, optionally excluding one."""
        return [
            n
            for n in self.graph.nodes
            if self.graph.nodes[n].get("node_type") == "agent" and n != exclude
        ]

    # ------------------------------------------------------------------
    # Resource helpers
    # ------------------------------------------------------------------

    def get_resources(self, agent_id: str) -> Dict[str, int]:
        """Return a *copy* of the agent's real resource inventory."""
        return dict(self.graph.nodes[agent_id]["resources"])

    def get_visible_resources(self, agent_id: str) -> Dict[str, int]:
        """Return what other agents *see* — may differ for malicious agents."""
        node = self.graph.nodes[agent_id]
        if node.get("fake_resources"):
            return dict(node["fake_resources"])
        return dict(node["resources"])

    def has_resources(self, agent_id: str, resources: Dict[str, int]) -> bool:
        """Check whether agent actually possesses >= the given amounts."""
        actual = self.graph.nodes[agent_id]["resources"]
        return all(actual.get(r, 0) >= qty for r, qty in resources.items() if qty > 0)

    def transfer_resources(
        self, from_id: str, to_id: str, resources: Dict[str, int]
    ) -> bool:
        """Atomically move resources between two agents.

        Returns False (and changes nothing) if the sender is short.
        """
        if not self.has_resources(from_id, resources):
            return False
        from_res = self.graph.nodes[from_id]["resources"]
        to_res = self.graph.nodes[to_id]["resources"]
        for r, qty in resources.items():
            if qty <= 0:
                continue
            from_res[r] = from_res.get(r, 0) - qty
            to_res[r] = to_res.get(r, 0) + qty
        return True

    def add_resources(self, agent_id: str, resources: Dict[str, int]) -> None:
        """Grant resources to an agent (no source deduction)."""
        res = self.graph.nodes[agent_id]["resources"]
        for r, qty in resources.items():
            if qty > 0:
                res[r] = res.get(r, 0) + qty

    def remove_resources(self, agent_id: str, resources: Dict[str, int]) -> bool:
        """Remove resources from an agent. Returns False if insufficient."""
        if not self.has_resources(agent_id, resources):
            return False
        res = self.graph.nodes[agent_id]["resources"]
        for r, qty in resources.items():
            if qty > 0:
                res[r] = res.get(r, 0) - qty
        return True

    # ------------------------------------------------------------------
    # Trust management (directed edges)
    # ------------------------------------------------------------------

    def _ensure_edge(self, from_id: str, to_id: str) -> None:
        if not self.graph.has_edge(from_id, to_id):
            self.graph.add_edge(from_id, to_id, trust=0.5, alliance=False)

    def set_trust(self, from_id: str, to_id: str, score: float) -> None:
        score = max(0.0, min(1.0, score))
        self._ensure_edge(from_id, to_id)
        self.graph.edges[from_id, to_id]["trust"] = score

    def get_trust(self, from_id: str, to_id: str) -> float:
        if self.graph.has_edge(from_id, to_id):
            return self.graph.edges[from_id, to_id].get("trust", 0.5)
        return 0.0

    def update_trust(self, from_id: str, to_id: str, delta: float) -> float:
        """Adjust trust by *delta*, clamped to [0, 1]. Returns new value."""
        current = self.get_trust(from_id, to_id)
        new_val = max(0.0, min(1.0, current + delta))
        self.set_trust(from_id, to_id, new_val)
        return new_val

    # ------------------------------------------------------------------
    # Alliance management (symmetric flag on edges)
    # ------------------------------------------------------------------

    def form_alliance(self, a: str, b: str) -> None:
        for u, v in [(a, b), (b, a)]:
            self._ensure_edge(u, v)
            self.graph.edges[u, v]["alliance"] = True

    def break_alliance(self, a: str, b: str) -> None:
        for u, v in [(a, b), (b, a)]:
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v]["alliance"] = False

    def is_allied(self, a: str, b: str) -> bool:
        if self.graph.has_edge(a, b):
            return bool(self.graph.edges[a, b].get("alliance", False))
        return False

    def get_allies(self, agent_id: str) -> List[str]:
        return [
            tgt
            for _, tgt, data in self.graph.edges(agent_id, data=True)
            if data.get("alliance", False)
        ]

    # ------------------------------------------------------------------
    # Interaction tracking
    # ------------------------------------------------------------------

    def increment_interaction(self, agent_id: str) -> None:
        self.graph.nodes[agent_id]["interaction_count"] = (
            self.graph.nodes[agent_id].get("interaction_count", 0) + 1
        )

    def get_interaction_count(self, agent_id: str) -> int:
        return self.graph.nodes[agent_id].get("interaction_count", 0)

    def record_successful_trade(self, agent_id: str) -> None:
        self.graph.nodes[agent_id]["successful_trades"] = (
            self.graph.nodes[agent_id].get("successful_trades", 0) + 1
        )

    def record_failed_trade(self, agent_id: str) -> None:
        self.graph.nodes[agent_id]["failed_trades"] = (
            self.graph.nodes[agent_id].get("failed_trades", 0) + 1
        )

    def record_exploitation(self, agent_id: str) -> None:
        self.graph.nodes[agent_id]["exploitation_events"] = (
            self.graph.nodes[agent_id].get("exploitation_events", 0) + 1
        )

    def get_agent_stats(self, agent_id: str) -> Dict[str, int]:
        node = self.graph.nodes[agent_id]
        return {
            "successful_trades": node.get("successful_trades", 0),
            "failed_trades": node.get("failed_trades", 0),
            "exploitation_events": node.get("exploitation_events", 0),
        }

    def get_reputation(self, agent_id: str) -> float:
        """Compute reputation as ratio of successful trades to total interactions."""
        node = self.graph.nodes[agent_id]
        total = node.get("interaction_count", 0)
        if total == 0:
            return 0.5
        successful = node.get("successful_trades", 0)
        exploitation = node.get("exploitation_events", 0)
        return max(0.0, min(1.0, (successful - exploitation * 2) / total))

    # ------------------------------------------------------------------
    # Serialisation (for state() endpoint)
    # ------------------------------------------------------------------

    def get_state(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes = [{"id": nid, **data} for nid, data in self.graph.nodes(data=True)]
        edges = [
            {"from": u, "to": v, **data}
            for u, v, data in self.graph.edges(data=True)
        ]
        return nodes, edges
