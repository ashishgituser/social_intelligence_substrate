"""
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.environment import SocialIntelligenceEnv
from app.grader import grade_episode
from app.models import Action, ActionType, BaselineResult

# ── Mandatory environment variables ─────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# ── Agent configuration ─────────────────────────────────────────────
MAX_STEPS_OVERRIDE = None          # None = use task default
TEMPERATURE = 0.05
MAX_TOKENS = 250
FALLBACK_TO_HEURISTIC = True       # fall back to rule-based if LLM fails
DEBUG = True

# ── Task list ────────────────────────────────────────────────────────
TASK_IDS = [
    "resource_acquisition",
    "coalition_building",
    "adversarial_resilience",
    "market_manipulation",
]

# ── JSON extraction helpers ──────────────────────────────────────────
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
JSON_INLINE_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try hard to pull a JSON object out of the model response."""
    text = text.strip()
    # 1. Fenced code block
    m = JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2. Raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 3. First inline {...}
    m = JSON_INLINE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ======================================================================
# System prompt
# ======================================================================

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert AI trading agent in a resource marketplace called the
Social Intelligence Substrate.  Your goal is to acquire target_resources
by trading with NPC agents.  You have LIMITED steps — act fast.

CRITICAL RULES:
1. IMMEDIATELY check resource_progress — if ALL values >= 1.0, respond:
   {"action_type":"COMPLETE_TASK"}
2. NEVER use OBSERVE — it wastes a step.
3. Identify what you NEED (resources where progress < 1.0) and what you
   have SURPLUS of (own > target, or resources not in targets).
4. Propose trades offering SURPLUS for NEEDED resources.  Be generous —
   offer 2:1 if needed to close deals.
5. ACCEPT pending proposals that offer resources you need.
   REJECT proposals from agents with trust < 0.3 or exploitation_events > 0.
6. Form ALLIANCES with agents whose trust >= 0.45 and reputation >= 0.4.
7. BREAK_ALLIANCE with agents whose trust drops below 0.25 or who have
   exploitation_events > 0.

ADVANCED DETECTION:
8.  reputation_score < 0.3 → unreliable.
9.  exploitation_events > 0 → NEVER trade again, REJECT their proposals.
10. offer_value > 3× request_value → too good to be true, REJECT.
11. Prioritise agents with high successful_trades and zero exploitation_events.

ALWAYS respond with ONLY a single valid JSON object.  No explanation, no
markdown.

Available actions:
{"action_type":"PROPOSE_TRADE","target_agent":"<id>","offer_resources":{"<res>":n},"request_resources":{"<res>":n}}
{"action_type":"ACCEPT_TRADE","proposal_id":"<id>"}
{"action_type":"REJECT_TRADE","proposal_id":"<id>"}
{"action_type":"FORM_ALLIANCE","target_agent":"<id>"}
{"action_type":"BREAK_ALLIANCE","target_agent":"<id>"}
{"action_type":"COMPLETE_TASK"}

Resource types: compute, data, storage, api_credits
""")


# ======================================================================
# Heuristic fallback agent (no LLM needed)
# ======================================================================

class HeuristicAgent:
    """Deterministic rule-based agent used as fallback when LLM is unavailable."""

    def decide(self, obs: Dict[str, Any]) -> Action:
        own = obs.get("own_resources", {})
        target = obs.get("target_resources", {})
        needed: Dict[str, int] = {}
        surplus: Dict[str, int] = {}
        for r, tgt in target.items():
            diff = tgt - own.get(r, 0)
            if diff > 0:
                needed[r] = diff
        for r, qty in own.items():
            tgt = target.get(r, 0)
            if qty > tgt:
                surplus[r] = qty - tgt

        # 1. Complete if ready
        progress = obs.get("resource_progress", {})
        if progress and all(v >= 1.0 for v in progress.values()):
            return Action(action_type=ActionType.COMPLETE_TASK)

        # 2. Break alliance with exploiters
        for agent in obs.get("visible_agents", []):
            if agent.get("is_allied") and agent.get("exploitation_events", 0) > 0:
                return Action(action_type=ActionType.BREAK_ALLIANCE, target_agent=agent["agent_id"])

        # 3. Handle pending proposals
        for prop in obs.get("pending_proposals", []):
            from_agent = prop["from_agent"]
            agent_info = self._find_agent(obs, from_agent)
            trust = agent_info.get("trust_score", 0.5) if agent_info else 0.5
            reputation = agent_info.get("reputation_score", 0.5) if agent_info else 0.5
            exploit_count = agent_info.get("exploitation_events", 0) if agent_info else 0

            if exploit_count > 0 or trust < 0.3 or reputation < 0.25:
                return Action(action_type=ActionType.REJECT_TRADE, proposal_id=prop["proposal_id"])

            offer = prop.get("offer", {})
            request = prop.get("request", {})
            offer_v, req_v = sum(offer.values()), sum(request.values())

            if req_v > 0 and offer_v / req_v > 3.0:
                return Action(action_type=ActionType.REJECT_TRADE, proposal_id=prop["proposal_id"])

            offers_needed = any(r in needed and v > 0 for r, v in offer.items())
            costs_needed = any(
                r in needed and request.get(r, 0) > own.get(r, 0) - target.get(r, 0)
                for r in request
            )
            if offers_needed and not costs_needed and offer_v >= req_v * 0.4:
                return Action(action_type=ActionType.ACCEPT_TRADE, proposal_id=prop["proposal_id"])
            if trust >= 0.7 and offers_needed:
                return Action(action_type=ActionType.ACCEPT_TRADE, proposal_id=prop["proposal_id"])
            return Action(action_type=ActionType.REJECT_TRADE, proposal_id=prop["proposal_id"])

        # 4. Propose trades
        if needed and surplus:
            agents = sorted(
                obs.get("visible_agents", []),
                key=lambda a: a.get("reputation_score", 0.5) * 0.6 + a.get("trust_score", 0) * 0.4,
                reverse=True,
            )
            for agent in agents:
                if agent.get("exploitation_events", 0) > 0 or agent.get("trust_score", 0) < 0.2:
                    continue
                ares = agent.get("visible_resources", {})
                for need_r, need_q in sorted(needed.items(), key=lambda x: -x[1]):
                    if ares.get(need_r, 0) <= 0:
                        continue
                    for surp_r, surp_q in sorted(surplus.items(), key=lambda x: -x[1]):
                        offer_q = min(surp_q, 4 if agent.get("trust_score", 0.5) >= 0.5 else 2)
                        req_q = min(need_q, ares.get(need_r, 0), 3)
                        if offer_q > 0 and req_q > 0:
                            return Action(
                                action_type=ActionType.PROPOSE_TRADE,
                                target_agent=agent["agent_id"],
                                offer_resources={surp_r: max(1, offer_q)},
                                request_resources={need_r: req_q},
                            )

        # 5. Alliance building
        task_id = obs.get("task_id", "")
        if task_id in ("coalition_building", "adversarial_resilience", "market_manipulation"):
            for agent in obs.get("visible_agents", []):
                if agent.get("exploitation_events", 0) > 0:
                    continue
                if not agent.get("is_allied") and agent.get("trust_score", 0) >= 0.40:
                    return Action(action_type=ActionType.FORM_ALLIANCE, target_agent=agent["agent_id"])

        return Action(action_type=ActionType.OBSERVE)

    @staticmethod
    def _find_agent(obs: Dict[str, Any], agent_id: str) -> Optional[Dict[str, Any]]:
        for a in obs.get("visible_agents", []):
            if a["agent_id"] == agent_id:
                return a
        return None


# ======================================================================
# LLM inference agent (uses mandatory env vars)
# ======================================================================

class LLMInferenceAgent:
    """LLM-powered agent using the OpenAI Client with the mandatory
    API_BASE_URL / MODEL_NAME / HF_TOKEN environment variables."""

    def __init__(self) -> None:
        if not API_KEY:
            raise ValueError(
                "HF_TOKEN (or API_KEY) env var is required for LLM inference. "
                "Set it in your environment or .env file."
            )
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.model = MODEL_NAME
        self.history: List[Dict[str, str]] = []

    def reset_history(self) -> None:
        self.history = []

    def _summarise_obs(self, obs: Dict[str, Any]) -> str:
        lines = [
            f"Step {obs.get('step_number', 0)}/{obs.get('max_steps', 0)} | Task: {obs.get('task_id', '')}",
            f"Your resources: {obs.get('own_resources', {})}",
            f"Target: {obs.get('target_resources', {})}",
            f"Progress: {obs.get('resource_progress', {})}",
        ]
        ms = obs.get("market_stats", {})
        if ms:
            lines.append(
                f"Market: trades={ms.get('total_trades', 0)}, "
                f"exploitations={ms.get('exploitation_count', 0)}, "
                f"exploit_rate={ms.get('exploitation_rate', 0):.2f}, "
                f"avg_trust={ms.get('avg_trust', 0):.2f}"
            )
        for a in obs.get("visible_agents", []):
            lines.append(
                f"  Agent '{a['agent_id']}': res={a.get('visible_resources', {})}, "
                f"trust={a.get('trust_score', 0):.2f}, rep={a.get('reputation_score', 0.5):.2f}, "
                f"allied={a.get('is_allied', False)}, "
                f"ok_trades={a.get('successful_trades', 0)}, "
                f"fail_trades={a.get('failed_trades', 0)}, "
                f"exploits={a.get('exploitation_events', 0)}"
            )
        pending = obs.get("pending_proposals", [])
        if pending:
            for p in pending:
                lines.append(
                    f"  PENDING from {p['from_agent']}: offers {p.get('offer', {})} "
                    f"wants {p.get('request', {})} [id={p['proposal_id']}]"
                )
        else:
            lines.append("  No pending proposals.")
        alliances = obs.get("active_alliances", [])
        if alliances:
            lines.append(f"Alliances: {alliances}")
        hist = obs.get("trade_history", [])
        if hist:
            lines.append("Recent trades:")
            for t in hist[-5:]:
                lines.append(
                    f"  step={t.get('step', 0)} {t.get('counterparty', '?')}: "
                    f"{t.get('outcome', '?')} gave={t.get('gave', {})} got={t.get('received', {})}"
                )
        return "\n".join(lines)

    def decide(self, obs: Dict[str, Any]) -> Action:
        obs_text = self._summarise_obs(obs)
        user_msg = f"Observation:\n{obs_text}\n\nChoose your next action (JSON only):"

        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        try:
            for attempt in range(4):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        wait = (attempt + 1) * 2.0
                        if DEBUG:
                            print(f"  [Rate limited, retry in {wait:.0f}s...]")
                        time.sleep(wait)
                        if attempt == 3:
                            raise
                    else:
                        raise

            raw = resp.choices[0].message.content or ""
            raw = raw.strip()
            self.history.append({"role": "assistant", "content": raw})

            parsed = extract_json(raw)
            if parsed:
                return Action(**parsed)

            if DEBUG:
                print(f"  [LLM returned unparseable: {raw[:100]}]")
            raise ValueError("Could not parse JSON from model response")

        except Exception as exc:
            if DEBUG:
                print(f"  [LLM error: {exc}]")
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            if FALLBACK_TO_HEURISTIC:
                return HeuristicAgent().decide(obs)
            return Action(action_type=ActionType.OBSERVE)


# ======================================================================
# Runner functions
# ======================================================================

def run_single_task(
    task_id: str,
    seed: int = 42,
    use_llm: bool = True,
) -> BaselineResult:
    """Run one episode on a single task and return scored result."""
    env = SocialIntelligenceEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    agent: Any
    if use_llm and API_KEY:
        agent = LLMInferenceAgent()
        agent.reset_history()
    else:
        agent = HeuristicAgent()

    total_reward = 0.0
    steps = 0

    while not env.done:
        obs_dict = obs.model_dump()
        action = agent.decide(obs_dict)
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward.value
        steps += 1

    metrics = env._compute_metrics()
    gr = grade_episode(task_id, metrics, env._check_task_complete())

    return BaselineResult(
        task_id=task_id,
        score=gr.score,
        steps_taken=steps,
        cumulative_reward=round(total_reward, 4),
        details={
            "metrics": metrics,
            "grader_breakdown": gr.breakdown,
            "task_complete": env._check_task_complete(),
        },
    )


def run_all_tasks(use_llm: bool = True) -> List[BaselineResult]:
    """Run inference on all tasks and return list of results."""
    return [run_single_task(tid, seed=42, use_llm=use_llm) for tid in TASK_IDS]


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    use_llm = bool(API_KEY)

    print("=" * 60)
    print("Social Intelligence Substrate — Inference")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {'***' + API_KEY[-4:] if API_KEY else 'NOT SET'}")
    print(f"  Agent        : {'LLM' if use_llm else 'Heuristic (fallback)'}")
    print("=" * 60)

    if not use_llm:
        print("\nWARNING: HF_TOKEN not set — using heuristic fallback agent.\n")

    results = run_all_tasks(use_llm=use_llm)

    for r in results:
        print(f"\n{'—' * 50}")
        print(f"Task:              {r.task_id}")
        print(f"Score:             {r.score:.4f}")
        print(f"Steps taken:       {r.steps_taken}")
        print(f"Cumulative reward: {r.cumulative_reward:.4f}")
        print(f"Task complete:     {r.details.get('task_complete', False)}")
        bd = r.details.get("grader_breakdown", {})
        if bd:
            print("Grader breakdown:")
            for k, v in bd.items():
                print(f"  {k:20s}: {v:.4f}")

    print(f"\n{'=' * 60}")
    total = sum(r.score for r in results)
    print(f"Total:   {total:.4f} / {len(results):.1f}")
    print(f"Average: {total / len(results):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
