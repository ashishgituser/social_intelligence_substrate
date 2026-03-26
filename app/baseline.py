"""Baseline inference script for the Social Intelligence Substrate.

Two agent implementations:
  1. HeuristicAgent — deterministic rule-based (always available)
  2. LLMAgent       — OpenAI API (used when OPENAI_API_KEY is set)

Usage (CLI):
    python -m app.baseline          # heuristic only
    OPENAI_API_KEY=sk-... python -m app.baseline   # LLM baseline

The /baseline endpoint calls ``run_baseline_all_tasks()`` directly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from app.environment import SocialIntelligenceEnv
from app.grader import grade_episode
from app.models import Action, ActionType, BaselineResult


# ======================================================================
# Heuristic agent (no LLM required)
# ======================================================================

class HeuristicAgent:
    """Advanced rule-based agent for reproducible baseline scoring.

    Strategy (priority order):
      1. If all target resources met → COMPLETE_TASK immediately.
      2. Handle pending proposals: reject from suspicious agents,
         accept profitable ones that move us toward the goal.
      3. Proactively propose trades: offer surplus for needed resources,
         targeting highest-reputation agents first.
      4. Form alliances when beneficial (coalition / adversarial / expert tasks).
      5. Break alliances with agents that exploited us.
      6. Fallback: OBSERVE (last resort).
    """

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

        # ------- 1. Check if task can be completed -------
        progress = obs.get("resource_progress", {})
        if progress and all(v >= 1.0 for v in progress.values()):
            return Action(action_type=ActionType.COMPLETE_TASK)

        # ------- 2. Break alliance with exploiters -------
        for agent in obs.get("visible_agents", []):
            if (
                agent.get("is_allied")
                and agent.get("exploitation_events", 0) > 0
            ):
                return Action(
                    action_type=ActionType.BREAK_ALLIANCE,
                    target_agent=agent["agent_id"],
                )

        # ------- 3. Handle pending proposals (smart filtering) -------
        pending = obs.get("pending_proposals", [])
        for prop in pending:
            from_agent = prop["from_agent"]
            agent_info = self._find_agent_info(obs, from_agent)
            trust = agent_info.get("trust_score", 0.5) if agent_info else 0.5
            reputation = agent_info.get("reputation_score", 0.5) if agent_info else 0.5
            exploit_count = agent_info.get("exploitation_events", 0) if agent_info else 0

            # Reject from anyone who has exploited us before
            if exploit_count > 0:
                return Action(
                    action_type=ActionType.REJECT_TRADE,
                    proposal_id=prop["proposal_id"],
                )

            # Reject if agent has low trust OR low reputation
            if trust < 0.3 or reputation < 0.25:
                return Action(
                    action_type=ActionType.REJECT_TRADE,
                    proposal_id=prop["proposal_id"],
                )

            offer = prop.get("offer", {})
            request = prop.get("request", {})
            offer_v = sum(offer.values())
            req_v = sum(request.values())

            # Suspicion heuristic: if offer is too good (>3x request), likely malicious
            if req_v > 0 and offer_v / req_v > 3.0:
                return Action(
                    action_type=ActionType.REJECT_TRADE,
                    proposal_id=prop["proposal_id"],
                )

            # Accept if it offers resources we need and doesn't hurt our progress
            offers_needed = any(
                r in needed and v > 0 for r, v in offer.items()
            )
            costs_needed = any(
                r in needed and request.get(r, 0) > own.get(r, 0) - target.get(r, 0)
                for r in request
            )
            # Also accept if it offers something we need even if not surplus-neutral
            if offers_needed and not costs_needed and offer_v >= req_v * 0.4:
                return Action(
                    action_type=ActionType.ACCEPT_TRADE,
                    proposal_id=prop["proposal_id"],
                )

            # Accept even marginal trades if from very trusted agents
            if trust >= 0.7 and offers_needed:
                return Action(
                    action_type=ActionType.ACCEPT_TRADE,
                    proposal_id=prop["proposal_id"],
                )

            # Reject proposals that don't help
            return Action(
                action_type=ActionType.REJECT_TRADE,
                proposal_id=prop["proposal_id"],
            )

        # ------- 4. Propose trades for needed resources -------
        if needed and surplus:
            agents = sorted(
                obs.get("visible_agents", []),
                key=lambda a: (
                    a.get("reputation_score", 0.5) * 0.6
                    + a.get("trust_score", 0) * 0.4
                ),
                reverse=True,
            )
            for agent in agents:
                # Skip agents who exploited us
                if agent.get("exploitation_events", 0) > 0:
                    continue
                if agent.get("trust_score", 0) < 0.2:
                    continue
                aid = agent["agent_id"]
                ares = agent.get("visible_resources", {})
                for need_r, need_q in sorted(needed.items(), key=lambda x: -x[1]):
                    avail = ares.get(need_r, 0)
                    if avail <= 0:
                        continue
                    for surp_r, surp_q in sorted(surplus.items(), key=lambda x: -x[1]):
                        # Be generous with high-trust agents, stingy with low
                        trust = agent.get("trust_score", 0.5)
                        max_offer = min(surp_q, 4) if trust >= 0.5 else min(surp_q, 2)
                        offer_q = max(1, max_offer)
                        req_q = min(need_q, avail, 3)
                        if offer_q > 0 and req_q > 0:
                            return Action(
                                action_type=ActionType.PROPOSE_TRADE,
                                target_agent=aid,
                                offer_resources={surp_r: offer_q},
                                request_resources={need_r: req_q},
                            )

        # If no surplus but still need things, try offering non-target resources
        if needed and not surplus:
            non_target = {
                r: q for r, q in own.items()
                if r not in target and q > 0
            }
            if non_target:
                agents = sorted(
                    obs.get("visible_agents", []),
                    key=lambda a: a.get("reputation_score", 0.5),
                    reverse=True,
                )
                for agent in agents:
                    if agent.get("exploitation_events", 0) > 0:
                        continue
                    if agent.get("trust_score", 0) < 0.3:
                        continue
                    ares = agent.get("visible_resources", {})
                    for need_r in needed:
                        if ares.get(need_r, 0) > 0:
                            for offer_r, offer_q in non_target.items():
                                return Action(
                                    action_type=ActionType.PROPOSE_TRADE,
                                    target_agent=agent["agent_id"],
                                    offer_resources={offer_r: min(offer_q, 2)},
                                    request_resources={need_r: min(needed[need_r], 2)},
                                )

        # ------- 5. Alliance building (broader applicability) -------
        task_id = obs.get("task_id", "")
        if task_id in ("coalition_building", "adversarial_resilience", "market_manipulation"):
            for agent in obs.get("visible_agents", []):
                if agent.get("exploitation_events", 0) > 0:
                    continue
                if (
                    not agent.get("is_allied")
                    and agent.get("trust_score", 0) >= 0.40
                    and agent.get("reputation_score", 0.5) >= 0.4
                ):
                    return Action(
                        action_type=ActionType.FORM_ALLIANCE,
                        target_agent=agent["agent_id"],
                    )

        # ------- 6. Fallback -------
        return Action(action_type=ActionType.OBSERVE)

    @staticmethod
    def _find_agent_info(obs: Dict[str, Any], agent_id: str) -> Optional[Dict[str, Any]]:
        for a in obs.get("visible_agents", []):
            if a["agent_id"] == agent_id:
                return a
        return None

    @staticmethod
    def _find_trust(obs: Dict[str, Any], agent_id: str) -> float:
        for a in obs.get("visible_agents", []):
            if a["agent_id"] == agent_id:
                return a.get("trust_score", 0.5)
        return 0.5


# ======================================================================
# LLM agent (OpenAI API)
# ======================================================================

class LLMAgent:
    """GPT-powered agent with conversation history for coherent multi-step play."""

    SYSTEM_PROMPT = """\
You are an expert AI trading agent in a resource marketplace. You MUST acquire \
target_resources by trading with other agents. You have LIMITED steps — act fast.

CRITICAL RULES:
1. IMMEDIATELY check resource_progress — if ALL values >= 1.0, respond: {"action_type":"COMPLETE_TASK"}
2. NEVER use OBSERVE — it wastes a step.
3. Identify what you NEED (resources where progress < 1.0) and what you have SURPLUS of \
   (own_resources that exceed your target for that resource, or resources not in your targets).
4. Propose trades offering SURPLUS for NEEDED resources. Be generous — offer 2:1 if needed to close deals.
5. If there are pending_proposals: ACCEPT if they offer resources you need. REJECT if from agents with trust < 0.3.
6. Agents with trust_score < 0.3 may be malicious — avoid trading with them.
7. For coalition/adversarial/expert tasks, FORM_ALLIANCE with agents whose trust_score >= 0.45.
8. If an agent's visible_resources look suspiciously high relative to others, they may be malicious.
9. BREAK_ALLIANCE with agents whose trust drops below 0.25 or who have exploitation_events > 0.

ADVANCED DETECTION STRATEGIES:
10. Check each agent's reputation_score — below 0.3 means unreliable.
11. Check exploitation_events — if > 0, NEVER trade with that agent again. REJECT their proposals.
12. Check trade_history for patterns: agents who gave nothing (exploitation) are malicious.
13. If an offer seems too good to be true (offer_value > 3× request_value), it's likely a scam — REJECT.
14. Prioritise agents with high successful_trades and zero exploitation_events.
15. Form alliances only with agents who have reputation_score >= 0.4.

ALWAYS respond with ONLY a single valid JSON object. No explanation, no markdown.

Available actions (JSON format):
{"action_type":"PROPOSE_TRADE","target_agent":"<id>","offer_resources":{"<resource>":qty},"request_resources":{"<resource>":qty}}
{"action_type":"ACCEPT_TRADE","proposal_id":"<id>"}
{"action_type":"REJECT_TRADE","proposal_id":"<id>"}
{"action_type":"FORM_ALLIANCE","target_agent":"<id>"}
{"action_type":"BREAK_ALLIANCE","target_agent":"<id>"}
{"action_type":"COMPLETE_TASK"}

Resource types: compute, data, storage, api_credits
"""

    def __init__(self) -> None:
        self.client = None
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.history: List[Dict[str, str]] = []
        try:
            from openai import OpenAI

            key = os.environ.get("OPENAI_API_KEY")
            if key:
                self.client = OpenAI(api_key=key)
        except Exception:
            pass

    def reset_history(self) -> None:
        self.history = []

    def _summarise_obs(self, obs: Dict[str, Any]) -> str:
        """Create a compact observation summary to save tokens."""
        lines = []
        lines.append(f"Step {obs.get('step_number',0)}/{obs.get('max_steps',0)} | Task: {obs.get('task_id','')}")
        lines.append(f"Your resources: {obs.get('own_resources',{})}")
        lines.append(f"Target: {obs.get('target_resources',{})}")
        lines.append(f"Progress: {obs.get('resource_progress',{})}")

        # Market stats
        ms = obs.get("market_stats", {})
        if ms:
            lines.append(
                f"Market: trades={ms.get('total_trades',0)}, "
                f"exploitations={ms.get('exploitation_count',0)}, "
                f"exploit_rate={ms.get('exploitation_rate',0):.2f}, "
                f"avg_trust={ms.get('avg_trust',0):.2f}"
            )

        for a in obs.get("visible_agents", []):
            lines.append(
                f"  Agent '{a['agent_id']}': resources={a.get('visible_resources',{})}, "
                f"trust={a.get('trust_score',0):.2f}, reputation={a.get('reputation_score',0.5):.2f}, "
                f"allied={a.get('is_allied',False)}, "
                f"trades_ok={a.get('successful_trades',0)}, "
                f"trades_fail={a.get('failed_trades',0)}, "
                f"exploits={a.get('exploitation_events',0)}"
            )
        pending = obs.get("pending_proposals", [])
        if pending:
            for p in pending:
                lines.append(
                    f"  PENDING from {p['from_agent']}: offers {p.get('offer',{})} "
                    f"wants {p.get('request',{})} [id={p['proposal_id']}]"
                )
        else:
            lines.append("  No pending proposals.")
        alliances = obs.get("active_alliances", [])
        if alliances:
            lines.append(f"Alliances: {alliances}")

        # Trade history summary (last 5)
        history = obs.get("trade_history", [])
        if history:
            lines.append("Recent trades:")
            for t in history[-5:]:
                lines.append(
                    f"  step={t.get('step',0)} {t.get('counterparty','?')}: "
                    f"{t.get('outcome','?')} gave={t.get('gave',{})} got={t.get('received',{})}"
                )

        events = obs.get("recent_events", [])
        if events:
            lines.append(f"Events: {events[-3:]}")
        return "\n".join(lines)

    def decide(self, obs: Dict[str, Any]) -> Action:
        if self.client is None:
            return HeuristicAgent().decide(obs)

        obs_text = self._summarise_obs(obs)
        user_msg = f"Observation:\n{obs_text}\n\nChoose your next action (JSON only):"

        # Keep conversation history for context (last 10 exchanges to stay within limits)
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.history

        try:
            # Retry with backoff for rate limits (429)
            for attempt in range(4):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        wait = (attempt + 1) * 1.5  # 1.5s, 3s, 4.5s, 6s
                        print(f"[Rate limited, retrying in {wait:.1f}s...]")
                        time.sleep(wait)
                        if attempt == 3:
                            raise
                    else:
                        raise
            raw = resp.choices[0].message.content or ""
            raw = raw.strip()
            # Store assistant response in history
            self.history.append({"role": "assistant", "content": raw})
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return Action(**json.loads(raw))
        except Exception as exc:
            print(f"[LLM fallback] {exc}")
            # Remove the failed user message from history
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            return HeuristicAgent().decide(obs)


# ======================================================================
# Runner
# ======================================================================

TASK_IDS = ["resource_acquisition", "coalition_building", "adversarial_resilience", "market_manipulation"]


def run_baseline_single(
    task_id: str, seed: int = 42, use_llm: bool = False
) -> BaselineResult:
    env = SocialIntelligenceEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    agent: Any
    if use_llm:
        agent = LLMAgent()
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


def run_baseline_all_tasks(use_llm: bool = False) -> List[BaselineResult]:
    """Run baseline on all tasks. The /baseline endpoint always uses heuristic
    for reproducibility. CLI uses LLM if OPENAI_API_KEY is set."""
    return [run_baseline_single(tid, seed=42, use_llm=use_llm) for tid in TASK_IDS]


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import sys
    use_llm = "--llm" in sys.argv or "--LLM" in sys.argv
    if use_llm and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: --llm flag set but OPENAI_API_KEY not found in environment")
        sys.exit(1)
    agent_label = "LLM (OpenAI)" if use_llm else "Heuristic (deterministic)"

    print("=" * 60)
    print("Social Intelligence Substrate — Baseline Evaluation")
    print(f"Agent: {agent_label}")
    print("=" * 60)

    results = run_baseline_all_tasks(use_llm=use_llm)

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
