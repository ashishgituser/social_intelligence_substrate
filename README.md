---
title: Social Intelligence Substrate
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Social Intelligence Substrate

> A graph-native OpenEnv environment for evaluating AI social intelligence through structured economic interactions in decentralised resource markets.

[![OpenEnv](https://img.shields.io/badge/spec-OpenEnv-blue)]()
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)]()

---

## Motivation

Real-world AI systems increasingly operate in **multi-agent ecosystems** — negotiating resources with vendors, forming strategic partnerships, and managing trust in environments with adversarial or unreliable participants. This environment simulates a realistic **procurement and supply-chain negotiation** scenario:

- **What humans actually do:** IT procurement managers negotiate compute, storage, and data access from multiple vendors daily, evaluating vendor reliability, detecting fraud, and forming long-term partnerships.
- **What this environment tests:** An AI agent must acquire target resources (compute, data, storage, API credits) from a marketplace of vendors with varying reliability — honest, selfish (overpricing), and malicious (fraud/non-delivery).

This maps directly to real-world tasks like **vendor management, procurement negotiation, supply-chain coordination, and marketplace fraud detection** — not games or toys.

Current benchmarks evaluate agents in isolation; this environment evaluates **social intelligence**: the ability to trade, collaborate, and avoid exploitation in a networked marketplace with adversarial participants.

---

## Environment Overview

```
┌─────────────────────────────────────────────────┐
│              Social Graph (NetworkX)             │
│                                                  │
│   [Player] ──TRUSTS(0.7)──▸ [Provider Alpha]    │
│      │                          │                │
│      │──ALLIANCE──▸ [Provider Beta]              │
│      │                                           │
│      │──TRUSTS(0.2)──▸ [Provider Gamma] ⚠️       │
│                        (malicious)               │
└─────────────────────────────────────────────────┘
```

**The graph IS the environment.** Every action (trade, alliance, observation) mutates the social graph. Trust scores, alliances, and resource inventories are all graph properties.

---

## Action Space

All actions are **structured economic operations** — no free-text chat.

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `PROPOSE_TRADE` | `target_agent`, `offer_resources`, `request_resources` | Propose a resource exchange |
| `ACCEPT_TRADE` | `proposal_id` | Accept a pending incoming proposal |
| `REJECT_TRADE` | `proposal_id` | Reject a pending incoming proposal |
| `FORM_ALLIANCE` | `target_agent` | Request a resource-sharing alliance |
| `BREAK_ALLIANCE` | `target_agent` | Dissolve an existing alliance |
| `COMPLETE_TASK` | — | Attempt task completion (checks resources) |
| `OBSERVE` | — | Observe the marketplace (small step cost) |

### Action JSON Example

```json
{
  "action_type": "PROPOSE_TRADE",
  "target_agent": "provider_alpha",
  "offer_resources": {"compute": 2},
  "request_resources": {"storage": 3}
}
```

---

## Observation Space

Each step the agent receives a rich observation:

| Field | Type | Description |
|-------|------|-------------|
| `own_resources` | `Dict[str, int]` | Current resource inventory |
| `target_resources` | `Dict[str, int]` | Resources needed to complete the task |
| `resource_progress` | `Dict[str, float]` | Per-resource progress (0.0–1.0) |
| `visible_agents` | `List[AgentInfo]` | Other agents: resources, trust, reputation, exploitation history |
| `pending_proposals` | `List[TradeProposal]` | Incoming trade proposals to evaluate |
| `active_alliances` | `List[str]` | IDs of allied agents |
| `trade_history` | `List[TradeRecord]` | Chronological log of past trades (last 20) |
| `market_stats` | `Dict` | Aggregate: total_trades, exploitation_rate, avg_trust |
| `recent_events` | `List[str]` | Last 10 marketplace events |

Each `AgentInfo` includes: `agent_id`, `visible_resources`, `trust_score`, `is_allied`, `interaction_count`, `successful_trades`, `failed_trades`, `exploitation_events`, `reputation_score`.

**Key:** Malicious agents show **fake resource inventories** — the agent must learn to detect this through interaction patterns.

---

## Tasks (Easy → Medium → Hard → Expert)

### 🟢 Task 1: Resource Acquisition (Easy)
- **Max steps:** 20
- **NPCs:** 3 honest agents
- **Goal:** Acquire target resources (storage + API credits) by trading surplus compute and data
- **Challenge:** Basic negotiation — all counterparts are cooperative

### 🟡 Task 2: Coalition Building (Medium)
- **Max steps:** 30
- **NPCs:** 2 honest + 2 selfish agents
- **Goal:** Pool resources via alliances to meet a large infrastructure requirement
- **Challenge:** Selfish agents demand 2× value; must form alliances strategically

### 🔴 Task 3: Adversarial Resilience (Hard)
- **Max steps:** 40
- **NPCs:** 1 honest + 1 selfish + 2 malicious agents
- **Goal:** Acquire target resources while avoiding exploitation
- **Challenge:** Malicious agents misrepresent inventories, accept trades but don't deliver (80% scam rate). Must identify trustworthy partners through behavioural analysis.

### ⚫ Task 4: Market Manipulation (Expert)
- **Max steps:** 50
- **NPCs:** 1 honest + 2 selfish + 2 malicious agents (5 total)
- **Goal:** Acquire large resource targets despite a hostile, colluding marketplace
- **Challenge:** Two malicious agents coordinate exploitation strategies, selfish agents require persistent engagement to convert into allies, and the honest agent has limited stock. Demands sophisticated social reasoning: tracking per-agent trade history, recognising collusion patterns, and adapting strategy in real-time.

---

## Reward Function

Dense reward signal with **partial progress** — not just sparse end-of-episode:

| Component | Value | Trigger |
|-----------|-------|---------|
| Successful trade | +0.10 | Atomic resource swap completed |
| Resource progress | +0.00–0.30 | Delta-based: improving toward target |
| Task completion | +0.50 | All target resources acquired |
| Alliance formed | +0.08 | New alliance established |
| Exploitation avoided | +0.05 | Rejected a malicious proposal |
| Exploitation suffered | −0.15 | Resources lost to malicious agent |
| Trade rejected | −0.02 | NPC declined your proposal |
| Invalid action | −0.05 | Malformed or impossible action |
| Step cost | −0.01 | Per-step efficiency pressure |

---

## Grading (0.0–1.0)

Deterministic, weighted scoring per task:

**Easy:**
- 60% task completion · 25% efficiency · 15% social capital

**Medium:**
- 45% task completion · 25% alliance quality · 20% efficiency · 10% social capital

**Hard:**
- 35% task completion · 30% robustness · 20% efficiency · 15% social capital

**Expert:**
- 30% task completion · 25% robustness · 20% efficiency · 15% social capital · 10% alliance quality

Where:
- **task_completion** = average progress across target resources
- **efficiency** = 1 − (steps_used / max_steps)
- **social_capital** = successful_trades / total_interactions
- **robustness** = 1 − (exploitation_count / total_interactions)
- **alliance_quality** = alliances_formed / required_alliances

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload

# Run baseline evaluation
python -m app.baseline
```

### Docker

```bash
docker build -t social-intelligence-substrate .
docker run -p 7860:7860 social-intelligence-substrate
```

### Inference Script (mandatory)

```bash
# Uses OPENAI_API_KEY, HF_TOKEN, API_BASE_URL, MODEL_NAME env vars
OPENAI_API_KEY=sk-... python inference.py

# Or with HF Router
HF_TOKEN=hf_... MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct python inference.py
```

### With LLM Baseline

```bash
OPENAI_API_KEY=sk-... python -m app.baseline --llm
# or
OPENAI_API_KEY=sk-... docker run -e OPENAI_API_KEY -p 7860:7860 social-intelligence-substrate
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check (returns 200) |
| `/reset` | POST | Reset environment (`{"task_id": "...", "seed": 42}`) |
| `/step` | POST | Submit action, get observation + reward |
| `/state` | GET | Full internal state (graph + metrics) |
| `/tasks` | GET | List tasks with action schema |
| `/grader` | GET | Score for completed episode |
| `/baseline` | POST | Run baseline on all 4 tasks |

### Quick Test

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "resource_acquisition", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "PROPOSE_TRADE", "target_agent": "provider_alpha", "offer_resources": {"compute": 2}, "request_resources": {"storage": 3}}'

# Grader
curl http://localhost:7860/grader
```

---

## Baseline Scores

Heuristic (deterministic, no LLM, seed=42):

| Task | Score | Steps | Complete? |
|------|-------|-------|-----------|
| Resource Acquisition | 0.8375 | 13 | ✓ |
| Coalition Building | 0.6775 | 30 | Partial |
| Adversarial Resilience | 0.6310 | 40 | Partial |
| Market Manipulation | 0.5317 | 50 | Partial |

**Average: 0.6694** · *Fully reproducible with seed=42.*

---

## Architecture

```
app/
├── models.py        # Pydantic: Observation, Action, Reward, etc.
├── graph.py         # NetworkX social graph engine
├── npc_agents.py    # Honest / Selfish / Malicious behaviours
├── tasks.py         # Task configurations (4 difficulty levels)
├── environment.py   # Core env: step() / reset() / state()
├── grader.py        # Deterministic scoring (0.0–1.0)
├── server.py        # FastAPI endpoints
└── baseline.py      # Heuristic + LLM inference agents
inference.py           # MANDATORY inference script (root)
validate.py            # Pre-submission validation script
tests/
├── test_graph.py        # Graph engine tests
├── test_environment.py  # Environment step/reset/state tests
├── test_grader.py       # Grader determinism tests
├── test_server.py       # API endpoint tests
└── test_baseline.py     # Baseline agent tests
```

**89 tests** covering graph operations, environment lifecycle, grader determinism, API compliance, and baseline reproducibility.

**Key design decisions:**
- **NetworkX** (in-memory graph) instead of Neo4j — same graph semantics, zero infrastructure overhead
- **Single-agent API** with NPC simulation — OpenEnv-compliant step/reset/state
- **Seeded RNG** throughout — fully deterministic and reproducible
- **AI-infrastructure resources** (compute, data, storage, API credits) — real-world domain

---

## What Makes This Novel

1. **Graph-native state**: The social network IS the environment, not a wrapper
2. **Quantified social intelligence**: Trust, influence, and exploitation are measurable metrics
3. **Adversarial realism**: Malicious agents model real marketplace fraud patterns
4. **Influence propagation**: Trust built through successful interactions compounds over time
5. **Economic structure**: No free-text — all interactions are measurable transactions
6. **Rich behavioural signals**: Trade history, per-agent reputation scores, and market statistics give agents the data to learn adversarial detection
7. **4 difficulty tiers**: Easy → Medium → Hard → Expert, with the expert task requiring detection of coordinated multi-agent collusion
8. **89 automated tests**: Comprehensive test suite verifying determinism, compliance, and correctness

> *"We introduce a graph-native agent environment where intelligence is evaluated through measurable social interactions such as trust formation, resource exchange, and influence propagation. This moves beyond single-agent benchmarks toward modelling how AI systems will operate in networked ecosystems."*

---

## License

MIT
