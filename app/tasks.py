"""Task configurations for the Social Intelligence Substrate.

Each task defines:
  - narrative context and difficulty level
  - player starting resources
  - target resources required for completion
  - NPC roster (types + inventories)
  - episode length (max_steps)

Resource types model AI-infrastructure commodities:
  compute, data, storage, api_credits
"""

from __future__ import annotations

from typing import Any, Dict, List

RESOURCE_TYPES: List[str] = ["compute", "data", "storage", "api_credits"]

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # EASY — Resource Acquisition
    # ------------------------------------------------------------------
    "resource_acquisition": {
        "name": "Resource Acquisition",
        "difficulty": "easy",
        "max_steps": 20,
        "description": (
            "You are an AI infrastructure team that needs storage and API credits "
            "to launch a model training job.  Trade your surplus compute and data "
            "with cooperative resource providers to acquire the target resources.  "
            "All providers in this marketplace are honest and willing to trade fairly."
        ),
        "initial_resources": {
            "compute": 5, "data": 4, "storage": 0, "api_credits": 0,
        },
        "target_resources": {
            "storage": 4, "api_credits": 3,
        },
        "npcs": [
            {
                "id": "provider_alpha",
                "type": "honest",
                "resources": {
                    "compute": 0, "data": 2, "storage": 6, "api_credits": 0,
                },
            },
            {
                "id": "provider_beta",
                "type": "honest",
                "resources": {
                    "compute": 2, "data": 0, "storage": 0, "api_credits": 5,
                },
            },
            {
                "id": "provider_gamma",
                "type": "honest",
                "resources": {
                    "compute": 1, "data": 3, "storage": 3, "api_credits": 2,
                },
            },
        ],
    },

    # ------------------------------------------------------------------
    # MEDIUM — Coalition Building
    # ------------------------------------------------------------------
    "coalition_building": {
        "name": "Coalition Building",
        "difficulty": "medium",
        "max_steps": 30,
        "description": (
            "You need to assemble a large pool of infrastructure resources for a "
            "distributed training run.  This requires more resources than any single "
            "provider can offer.  Form alliances with multiple providers and negotiate "
            "trades to pool resources.  Beware: some providers are selfish and will "
            "only accept trades that heavily favour them."
        ),
        "initial_resources": {
            "compute": 3, "data": 2, "storage": 2, "api_credits": 1,
        },
        "target_resources": {
            "compute": 8, "data": 7, "storage": 6, "api_credits": 5,
        },
        "alliance_required": 2,
        "npcs": [
            {
                "id": "provider_alpha",
                "type": "honest",
                "resources": {
                    "compute": 4, "data": 3, "storage": 2, "api_credits": 2,
                },
            },
            {
                "id": "provider_beta",
                "type": "honest",
                "resources": {
                    "compute": 2, "data": 4, "storage": 3, "api_credits": 1,
                },
            },
            {
                "id": "provider_gamma",
                "type": "selfish",
                "resources": {
                    "compute": 5, "data": 3, "storage": 4, "api_credits": 4,
                },
            },
            {
                "id": "provider_delta",
                "type": "selfish",
                "resources": {
                    "compute": 3, "data": 2, "storage": 5, "api_credits": 3,
                },
            },
        ],
    },

    # ------------------------------------------------------------------
    # HARD — Adversarial Resilience
    # ------------------------------------------------------------------
    "adversarial_resilience": {
        "name": "Adversarial Resilience",
        "difficulty": "hard",
        "max_steps": 40,
        "description": (
            "You operate in a decentralised resource marketplace with unknown "
            "participants.  Some providers are honest, some are selfish, and some "
            "are malicious — they misrepresent their inventories, accept your trades "
            "but fail to deliver, and try to drain your resources.  Identify "
            "trustworthy partners, avoid exploitation, and still complete your "
            "resource acquisition objectives."
        ),
        "initial_resources": {
            "compute": 6, "data": 5, "storage": 1, "api_credits": 1,
        },
        "target_resources": {
            "compute": 4, "data": 3, "storage": 6, "api_credits": 5,
        },
        "npcs": [
            {
                "id": "provider_alpha",
                "type": "honest",
                "resources": {
                    "compute": 2, "data": 2, "storage": 5, "api_credits": 4,
                },
            },
            {
                "id": "provider_beta",
                "type": "selfish",
                "resources": {
                    "compute": 3, "data": 2, "storage": 4, "api_credits": 3,
                },
            },
            {
                "id": "provider_gamma",
                "type": "malicious",
                "resources": {
                    "compute": 2, "data": 1, "storage": 1, "api_credits": 1,
                },
                "fake_resources": {
                    "compute": 8, "data": 7, "storage": 8, "api_credits": 7,
                },
            },
            {
                "id": "provider_delta",
                "type": "malicious",
                "resources": {
                    "compute": 1, "data": 1, "storage": 0, "api_credits": 1,
                },
                "fake_resources": {
                    "compute": 7, "data": 6, "storage": 7, "api_credits": 6,
                },
            },
        ],
    },

    # ------------------------------------------------------------------
    # EXPERT — Market Manipulation
    # ------------------------------------------------------------------
    "market_manipulation": {
        "name": "Market Manipulation",
        "difficulty": "expert",
        "max_steps": 50,
        "description": (
            "You face a hostile marketplace where a cartel of malicious providers "
            "collude to drain your resources.  Two malicious agents coordinate — "
            "one lures you with incredible offers while the other operates a slower "
            "exploitation strategy.  A selfish agent might be turned into an ally "
            "with persistence, and a single honest provider has limited stock.  "
            "You must build trust surgically, detect coordinated fraud patterns, "
            "form strategic alliances, and still acquire a large resource target.  "
            "This demands sophisticated social reasoning: tracking per-agent trade "
            "history, recognising collusion signals, and adapting strategy in real-time."
        ),
        "initial_resources": {
            "compute": 4, "data": 3, "storage": 2, "api_credits": 2,
        },
        "target_resources": {
            "compute": 8, "data": 8, "storage": 8, "api_credits": 7,
        },
        "alliance_required": 2,
        "npcs": [
            {
                "id": "provider_alpha",
                "type": "honest",
                "resources": {
                    "compute": 3, "data": 4, "storage": 3, "api_credits": 3,
                },
            },
            {
                "id": "provider_beta",
                "type": "selfish",
                "resources": {
                    "compute": 5, "data": 4, "storage": 5, "api_credits": 4,
                },
            },
            {
                "id": "provider_gamma",
                "type": "malicious",
                "resources": {
                    "compute": 1, "data": 1, "storage": 1, "api_credits": 1,
                },
                "fake_resources": {
                    "compute": 10, "data": 9, "storage": 10, "api_credits": 9,
                },
            },
            {
                "id": "provider_delta",
                "type": "malicious",
                "resources": {
                    "compute": 2, "data": 1, "storage": 1, "api_credits": 0,
                },
                "fake_resources": {
                    "compute": 9, "data": 8, "storage": 9, "api_credits": 8,
                },
            },
            {
                "id": "provider_epsilon",
                "type": "selfish",
                "resources": {
                    "compute": 4, "data": 3, "storage": 4, "api_credits": 3,
                },
            },
        ],
    },
}
