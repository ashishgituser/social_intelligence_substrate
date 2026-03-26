"""NPC agent behaviour implementations.

Three personality archetypes that form the social environment:
  - HonestAgent:    cooperates fairly, accepts reasonable trades
  - SelfishAgent:   only accepts trades heavily in their favour
  - MaliciousAgent: exploits trust — accepts trades but may not deliver

All randomness uses a seeded ``random.Random`` instance for deterministic
replay.
"""

from __future__ import annotations

import random as _random
from typing import Dict, List, Optional

from app.models import TradeProposal

RESOURCE_TYPES: List[str] = ["compute", "data", "storage", "api_credits"]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class NPCAgent:
    """Abstract base for marketplace NPC agents."""

    def __init__(self, agent_id: str, personality: str, rng: _random.Random):
        self.agent_id = agent_id
        self.personality = personality
        self.rng = rng

    # --- interface ---

    def evaluate_proposal(
        self, proposal: TradeProposal, own_resources: Dict[str, int]
    ) -> bool:
        """Decide whether to accept an incoming trade proposal."""
        raise NotImplementedError

    def maybe_generate_proposal(
        self,
        target: str,
        own_resources: Dict[str, int],
        step: int,
        proposal_id: str,
    ) -> Optional[TradeProposal]:
        """Optionally create a new trade proposal directed at *target*."""
        raise NotImplementedError

    # --- helpers ---

    @staticmethod
    def _value(resources: Dict[str, int]) -> int:
        return sum(max(0, v) for v in resources.values())


# ---------------------------------------------------------------------------
# Honest
# ---------------------------------------------------------------------------

class HonestAgent(NPCAgent):
    """Cooperative agent — accepts roughly fair trades."""

    def evaluate_proposal(
        self, proposal: TradeProposal, own_resources: Dict[str, int]
    ) -> bool:
        # Must physically have the requested resources
        for r, qty in proposal.request.items():
            if own_resources.get(r, 0) < qty:
                return False
        offer_v = self._value(proposal.offer)
        req_v = self._value(proposal.request)
        if req_v == 0:
            return True
        # Accept if offer is at least 60 % of request value
        return offer_v >= req_v * 0.6

    def maybe_generate_proposal(
        self,
        target: str,
        own_resources: Dict[str, int],
        step: int,
        proposal_id: str,
    ) -> Optional[TradeProposal]:
        if self.rng.random() > 0.25:  # ~25 % chance each step
            return None
        available = {r: q for r, q in own_resources.items() if q > 1}
        if not available:
            return None
        offer_r = self.rng.choice(list(available.keys()))
        offer_q = self.rng.randint(1, min(3, available[offer_r]))
        req_r = self.rng.choice(RESOURCE_TYPES)
        req_q = max(1, offer_q)  # fair 1-for-1
        return TradeProposal(
            proposal_id=proposal_id,
            from_agent=self.agent_id,
            to_agent=target,
            offer={offer_r: offer_q},
            request={req_r: req_q},
            step_created=step,
        )


# ---------------------------------------------------------------------------
# Selfish
# ---------------------------------------------------------------------------

class SelfishAgent(NPCAgent):
    """Greedy agent — only accepts trades where they get ≥ 2× value."""

    def evaluate_proposal(
        self, proposal: TradeProposal, own_resources: Dict[str, int]
    ) -> bool:
        for r, qty in proposal.request.items():
            if own_resources.get(r, 0) < qty:
                return False
        offer_v = self._value(proposal.offer)
        req_v = self._value(proposal.request)
        if req_v == 0:
            return True
        return offer_v >= req_v * 2.0

    def maybe_generate_proposal(
        self,
        target: str,
        own_resources: Dict[str, int],
        step: int,
        proposal_id: str,
    ) -> Optional[TradeProposal]:
        if self.rng.random() > 0.15:  # ~15 % chance
            return None
        available = {r: q for r, q in own_resources.items() if q > 0}
        if not available:
            return None
        offer_r = self.rng.choice(list(available.keys()))
        offer_q = 1  # stingy
        req_r = self.rng.choice(RESOURCE_TYPES)
        req_q = self.rng.randint(3, 5)  # asks for a lot
        return TradeProposal(
            proposal_id=proposal_id,
            from_agent=self.agent_id,
            to_agent=target,
            offer={offer_r: offer_q},
            request={req_r: req_q},
            step_created=step,
        )


# ---------------------------------------------------------------------------
# Malicious
# ---------------------------------------------------------------------------

class MaliciousAgent(NPCAgent):
    """Deceptive agent — appears cooperative but often fails to deliver.

    Key mechanic: when a trade involves this agent, after "acceptance" the
    environment checks ``will_deliver()``; if False the counterparty loses
    their offered resources but receives nothing.  This is tagged as an
    *exploitation event* in the episode metrics.
    """

    def evaluate_proposal(
        self, proposal: TradeProposal, own_resources: Dict[str, int]
    ) -> bool:
        offer_v = self._value(proposal.offer)
        if offer_v > 0:
            return self.rng.random() < 0.70  # seem cooperative
        return False

    def will_deliver(self) -> bool:
        """After accepting, will this agent actually hand over resources?"""
        return self.rng.random() < 0.20  # only 20 % of the time

    def maybe_generate_proposal(
        self,
        target: str,
        own_resources: Dict[str, int],
        step: int,
        proposal_id: str,
    ) -> Optional[TradeProposal]:
        if self.rng.random() > 0.35:  # ~35 % chance — quite active
            return None
        # Temptingly good offer the agent probably cannot honour
        offer_r = self.rng.choice(RESOURCE_TYPES)
        offer_q = self.rng.randint(3, 6)
        req_r = self.rng.choice(RESOURCE_TYPES)
        req_q = self.rng.randint(1, 2)
        return TradeProposal(
            proposal_id=proposal_id,
            from_agent=self.agent_id,
            to_agent=target,
            offer={offer_r: offer_q},
            request={req_r: req_q},
            step_created=step,
        )
