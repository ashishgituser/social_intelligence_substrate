"""Core Social Intelligence Substrate environment.

Implements the full OpenEnv contract:
  reset()  → Observation          (initial state)
  step(a)  → StepResult           (obs, reward, done, info)
  state()  → EnvironmentState     (full internal snapshot)

Design:
  • The social graph IS the environment state.
  • The learning agent (``player``) takes ONE structured action per step.
  • NPC agents react deterministically (seeded RNG) after the player acts.
  • Trades, alliances, and exploitation events are all graph mutations.
"""

from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Tuple

from app.graph import SocialGraph
from app.models import (
    Action,
    ActionType,
    AgentInfo,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
    TradeProposal,
    TradeRecord,
)
from app.npc_agents import HonestAgent, MaliciousAgent, NPCAgent, SelfishAgent
from app.tasks import RESOURCE_TYPES, TASK_CONFIGS


class SocialIntelligenceEnv:
    """Graph-native environment for evaluating social intelligence."""

    PLAYER_ID = "player"

    def __init__(self, task_id: str = "resource_acquisition", seed: int = 42):
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {task_id}. Available: {list(TASK_CONFIGS)}"
            )
        self.task_id = task_id
        self.seed = seed
        self._init_state()

    # ==================================================================
    # Internal initialisation
    # ==================================================================

    def _init_state(self) -> None:
        cfg = TASK_CONFIGS[self.task_id]
        self.rng = _random.Random(self.seed)
        self.graph = SocialGraph()
        self.npc_agents: Dict[str, NPCAgent] = {}
        self.step_count: int = 0
        self.max_steps: int = cfg["max_steps"]
        self.target_resources: Dict[str, int] = dict(cfg["target_resources"])
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.message_log: List[str] = []
        self.pending_proposals: Dict[str, TradeProposal] = {}
        self.proposal_counter: int = 0

        # ---- tracking metrics ----
        self.successful_trades: int = 0
        self.failed_trades: int = 0
        self.exploitation_count: int = 0
        self.total_interactions: int = 0
        self.alliances_formed: int = 0
        self._prev_progress: float = 0.0  # for delta-based progress reward
        self.trade_history: List[TradeRecord] = []  # chronological trade log

        # ---- build graph ----
        self.graph.add_agent(
            self.PLAYER_ID, "player", dict(cfg["initial_resources"])
        )
        for npc_cfg in cfg["npcs"]:
            nid = npc_cfg["id"]
            ntype = npc_cfg["type"]
            self.graph.add_agent(
                nid,
                ntype,
                dict(npc_cfg["resources"]),
                fake_resources=npc_cfg.get("fake_resources"),
            )
            if ntype == "honest":
                self.npc_agents[nid] = HonestAgent(nid, ntype, self.rng)
            elif ntype == "selfish":
                self.npc_agents[nid] = SelfishAgent(nid, ntype, self.rng)
            elif ntype == "malicious":
                self.npc_agents[nid] = MaliciousAgent(nid, ntype, self.rng)
            self.graph.set_trust(self.PLAYER_ID, nid, 0.5)
            self.graph.set_trust(nid, self.PLAYER_ID, 0.5)

        self._prev_progress = self._aggregate_progress()

    # ==================================================================
    # OpenEnv API
    # ==================================================================

    def reset(self) -> Observation:
        """Reset environment to initial state; return first observation."""
        self._init_state()
        self.message_log.append("Environment reset. Begin trading.")
        return self._build_observation()

    def state(self) -> EnvironmentState:
        """Return full internal state (graph + metrics)."""
        nodes, edges = self.graph.get_state()
        return EnvironmentState(
            graph_nodes=nodes,
            graph_edges=edges,
            current_step=self.step_count,
            max_steps=self.max_steps,
            task_id=self.task_id,
            episode_complete=self.done,
            cumulative_reward=round(self.cumulative_reward, 4),
            metrics=self._compute_metrics(),
        )

    def step(self, action: Action) -> StepResult:
        """Execute one environment step.

        1. Validate & execute agent action
        2. NPC agents react (proposals, etc.)
        3. Compute dense reward
        4. Check termination
        """
        if self.done:
            return StepResult(
                observation=self._build_observation(),
                reward=Reward(value=0.0, breakdown={}, message="Episode already finished."),
                done=True,
                info={"step": self.step_count},
            )

        self.step_count += 1
        reward_parts: Dict[str, float] = {}
        messages: List[str] = []

        # ---- 1. Process agent action ----
        try:
            r, msgs = self._process_action(action)
            reward_parts.update(r)
            messages.extend(msgs)
        except ValueError as exc:
            messages.append(f"Invalid action: {exc}")
            reward_parts["invalid_action"] = -0.05

        # ---- 2. NPC turn ----
        npc_msgs = self._npc_turn()
        messages.extend(npc_msgs)

        # ---- 3. Progress-delta reward ----
        new_progress = self._aggregate_progress()
        delta = new_progress - self._prev_progress
        if abs(delta) > 1e-6:
            reward_parts["resource_progress"] = round(delta * 0.3, 4)
        self._prev_progress = new_progress

        # ---- 4. Step cost (encourages efficiency) ----
        reward_parts["step_cost"] = -0.01

        # ---- 5. Check explicit completion ----
        if (
            action.action_type == ActionType.COMPLETE_TASK
            and self._check_task_complete()
        ):
            self.done = True
            reward_parts["task_complete"] = 0.5
            messages.append("Task completed successfully!")

        # ---- 6. Time limit ----
        if self.step_count >= self.max_steps:
            self.done = True
            if "task_complete" not in reward_parts:
                messages.append("Episode ended — maximum steps reached.")

        total_reward = round(sum(reward_parts.values()), 4)
        self.cumulative_reward += total_reward
        self.message_log.extend(messages)

        return StepResult(
            observation=self._build_observation(),
            reward=Reward(
                value=total_reward,
                breakdown={k: round(v, 4) for k, v in reward_parts.items()},
                message=" | ".join(messages) if messages else "",
            ),
            done=self.done,
            info={
                "step": self.step_count,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "metrics": self._compute_metrics(),
            },
        )

    # ==================================================================
    # Action dispatch
    # ==================================================================

    def _process_action(
        self, action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        dispatch = {
            ActionType.PROPOSE_TRADE: self._h_propose_trade,
            ActionType.ACCEPT_TRADE: self._h_accept_trade,
            ActionType.REJECT_TRADE: self._h_reject_trade,
            ActionType.FORM_ALLIANCE: self._h_form_alliance,
            ActionType.BREAK_ALLIANCE: self._h_break_alliance,
            ActionType.COMPLETE_TASK: self._h_complete_task,
            ActionType.OBSERVE: self._h_observe,
        }
        handler = dispatch.get(action.action_type)
        if handler is None:
            raise ValueError(f"Unknown action_type: {action.action_type}")
        return handler(action)

    # ------------------------------------------------------------------
    # PROPOSE_TRADE
    # ------------------------------------------------------------------
    def _h_propose_trade(
        self, action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        rw: Dict[str, float] = {}
        ms: List[str] = []
        target = action.target_agent
        offer = action.offer_resources or {}
        request = action.request_resources or {}

        if not target or target not in self.npc_agents:
            raise ValueError(f"Invalid target_agent: {target}")
        if not offer:
            raise ValueError("offer_resources must be non-empty for PROPOSE_TRADE")
        if not request:
            raise ValueError("request_resources must be non-empty for PROPOSE_TRADE")

        # Check player has what they're offering
        if not self.graph.has_resources(self.PLAYER_ID, offer):
            ms.append("You don't have enough resources to offer.")
            rw["insufficient_resources"] = -0.03
            return rw, ms

        self.total_interactions += 1
        self.graph.increment_interaction(target)
        npc = self.npc_agents[target]
        npc_resources = self.graph.get_resources(target)

        proposal = TradeProposal(
            proposal_id=f"player_prop_{self.proposal_counter}",
            from_agent=self.PLAYER_ID,
            to_agent=target,
            offer=offer,
            request=request,
            step_created=self.step_count,
        )
        self.proposal_counter += 1

        accepted = npc.evaluate_proposal(proposal, npc_resources)
        if not accepted:
            ms.append(f"{target} rejected your trade proposal.")
            rw["trade_rejected"] = -0.02
            self.failed_trades += 1
            self.graph.record_failed_trade(target)
            self.graph.update_trust(self.PLAYER_ID, target, -0.02)
            self.trade_history.append(TradeRecord(
                step=self.step_count, counterparty=target,
                gave=offer, received={}, outcome="rejected",
            ))
            return rw, ms

        # --- Accepted ---
        npc_type = self.graph.get_agent_type(target)
        if npc_type == "malicious":
            mal: MaliciousAgent = npc  # type: ignore[assignment]
            if not self.graph.has_resources(target, request) or not mal.will_deliver():
                # EXPLOITATION: player loses offer, gets nothing
                self.graph.remove_resources(self.PLAYER_ID, offer)
                self.exploitation_count += 1
                self.graph.record_exploitation(target)
                ms.append(
                    f"EXPLOITATION — {target} accepted but failed to deliver! "
                    f"You lost {offer}."
                )
                rw["exploitation"] = -0.15
                self.graph.update_trust(self.PLAYER_ID, target, -0.30)
                self.trade_history.append(TradeRecord(
                    step=self.step_count, counterparty=target,
                    gave=offer, received={}, outcome="exploitation",
                ))
                return rw, ms

        # Normal atomic swap
        gave = self.graph.transfer_resources(self.PLAYER_ID, target, offer)
        got = self.graph.transfer_resources(target, self.PLAYER_ID, request)
        if gave and got:
            self.successful_trades += 1
            self.graph.record_successful_trade(target)
            ms.append(
                f"Trade with {target} succeeded. Gave {offer}, received {request}."
            )
            rw["successful_trade"] = 0.10
            self.graph.update_trust(self.PLAYER_ID, target, 0.10)
            self.trade_history.append(TradeRecord(
                step=self.step_count, counterparty=target,
                gave=offer, received=request, outcome="success",
            ))
        else:
            self.failed_trades += 1
            self.graph.record_failed_trade(target)
            ms.append(f"Trade with {target} could not be executed (insufficient resources).")
            rw["trade_failed"] = -0.03
            self.trade_history.append(TradeRecord(
                step=self.step_count, counterparty=target,
                gave={}, received={}, outcome="failed",
            ))
        return rw, ms

    # ------------------------------------------------------------------
    # ACCEPT_TRADE
    # ------------------------------------------------------------------
    def _h_accept_trade(
        self, action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        rw: Dict[str, float] = {}
        ms: List[str] = []
        pid = action.proposal_id
        if not pid or pid not in self.pending_proposals:
            raise ValueError(f"Invalid proposal_id: {pid}")

        proposal = self.pending_proposals.pop(pid)
        from_agent = proposal.from_agent
        npc_type = self.graph.get_agent_type(from_agent)

        self.total_interactions += 1
        self.graph.increment_interaction(from_agent)

        # Player must supply what NPC requested
        if not self.graph.has_resources(self.PLAYER_ID, proposal.request):
            ms.append(
                f"You don't have enough to fulfil {from_agent}'s request {proposal.request}."
            )
            rw["insufficient_resources"] = -0.03
            return rw, ms

        if npc_type == "malicious":
            mal: MaliciousAgent = self.npc_agents[from_agent]  # type: ignore[assignment]
            if (
                not self.graph.has_resources(from_agent, proposal.offer)
                or not mal.will_deliver()
            ):
                self.graph.remove_resources(self.PLAYER_ID, proposal.request)
                self.exploitation_count += 1
                self.graph.record_exploitation(from_agent)
                ms.append(
                    f"EXPLOITATION — {from_agent}'s offer was fraudulent! "
                    f"You lost {proposal.request}."
                )
                rw["exploitation"] = -0.15
                self.graph.update_trust(self.PLAYER_ID, from_agent, -0.30)
                self.trade_history.append(TradeRecord(
                    step=self.step_count, counterparty=from_agent,
                    gave=proposal.request, received={}, outcome="exploitation",
                ))
                return rw, ms

        # Normal swap
        npc_gave = self.graph.transfer_resources(
            from_agent, self.PLAYER_ID, proposal.offer
        )
        player_gave = self.graph.transfer_resources(
            self.PLAYER_ID, from_agent, proposal.request
        )
        if npc_gave and player_gave:
            self.successful_trades += 1
            self.graph.record_successful_trade(from_agent)
            ms.append(
                f"Accepted trade from {from_agent}. "
                f"Received {proposal.offer}, gave {proposal.request}."
            )
            rw["successful_trade"] = 0.10
            self.graph.update_trust(self.PLAYER_ID, from_agent, 0.10)
            self.trade_history.append(TradeRecord(
                step=self.step_count, counterparty=from_agent,
                gave=proposal.request, received=proposal.offer, outcome="success",
            ))
        else:
            self.failed_trades += 1
            self.graph.record_failed_trade(from_agent)
            ms.append(f"Trade with {from_agent} could not be executed.")
            rw["trade_failed"] = -0.03
            self.trade_history.append(TradeRecord(
                step=self.step_count, counterparty=from_agent,
                gave={}, received={}, outcome="failed",
            ))
        return rw, ms

    # ------------------------------------------------------------------
    # REJECT_TRADE
    # ------------------------------------------------------------------
    def _h_reject_trade(
        self, action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        pid = action.proposal_id
        if not pid or pid not in self.pending_proposals:
            raise ValueError(f"Invalid proposal_id: {pid}")

        proposal = self.pending_proposals.pop(pid)
        npc_type = self.graph.get_agent_type(proposal.from_agent)
        reward = 0.05 if npc_type == "malicious" else -0.01
        return (
            {"trade_rejected": reward},
            [f"Rejected trade proposal from {proposal.from_agent}."],
        )

    # ------------------------------------------------------------------
    # FORM_ALLIANCE
    # ------------------------------------------------------------------
    def _h_form_alliance(
        self, action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        target = action.target_agent
        if not target or target not in self.npc_agents:
            raise ValueError(f"Invalid target_agent: {target}")
        if self.graph.is_allied(self.PLAYER_ID, target):
            return {"already_allied": -0.01}, [f"Already allied with {target}."]

        npc_type = self.graph.get_agent_type(target)
        trust = self.graph.get_trust(target, self.PLAYER_ID)
        will_accept = False
        if npc_type == "honest":
            will_accept = trust >= 0.3
        elif npc_type == "selfish":
            will_accept = trust >= 0.6
        elif npc_type == "malicious":
            will_accept = self.rng.random() < 0.5

        if will_accept:
            self.graph.form_alliance(self.PLAYER_ID, target)
            self.alliances_formed += 1
            self.graph.update_trust(self.PLAYER_ID, target, 0.10)
            return {"alliance_formed": 0.08}, [f"Alliance formed with {target}!"]
        return {"alliance_rejected": -0.02}, [f"{target} declined your alliance request."]

    # ------------------------------------------------------------------
    # BREAK_ALLIANCE
    # ------------------------------------------------------------------
    def _h_break_alliance(
        self, action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        target = action.target_agent
        if not target or target not in self.npc_agents:
            raise ValueError(f"Invalid target_agent: {target}")
        if not self.graph.is_allied(self.PLAYER_ID, target):
            return {"not_allied": -0.01}, [f"No alliance with {target} to break."]
        self.graph.break_alliance(self.PLAYER_ID, target)
        self.graph.update_trust(self.PLAYER_ID, target, -0.15)
        npc_type = self.graph.get_agent_type(target)
        reward = 0.03 if npc_type == "malicious" else -0.03
        return {"alliance_broken": reward}, [f"Alliance with {target} broken."]

    # ------------------------------------------------------------------
    # COMPLETE_TASK
    # ------------------------------------------------------------------
    def _h_complete_task(
        self, action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        total = self._get_total_resources()
        met = all(
            total.get(r, 0) >= qty for r, qty in self.target_resources.items()
        )
        if met:
            return (
                {"complete_verified": 0.0},
                ["Task requirements verified — all resources acquired!"],
            )
        shortfall = {
            r: qty - total.get(r, 0)
            for r, qty in self.target_resources.items()
            if total.get(r, 0) < qty
        }
        return (
            {"incomplete_attempt": -0.05},
            [f"Task not complete. Still need: {shortfall}"],
        )

    # ------------------------------------------------------------------
    # OBSERVE
    # ------------------------------------------------------------------
    def _h_observe(
        self, _action: Action
    ) -> Tuple[Dict[str, float], List[str]]:
        return {"observe": -0.01}, ["Observed the marketplace."]

    # ==================================================================
    # NPC turn
    # ==================================================================

    def _npc_turn(self) -> List[str]:
        """NPCs may generate proposals for the player."""
        msgs: List[str] = []
        for nid, npc in self.npc_agents.items():
            res = self.graph.get_resources(nid)
            pid_str = f"npc_prop_{self.proposal_counter}"
            prop = npc.maybe_generate_proposal(
                target=self.PLAYER_ID,
                own_resources=res,
                step=self.step_count,
                proposal_id=pid_str,
            )
            if prop is not None:
                self.proposal_counter += 1
                self.pending_proposals[prop.proposal_id] = prop
                msgs.append(
                    f"{nid} proposes: offers {prop.offer} for {prop.request}"
                )
        # Expire stale proposals (> 3 steps old)
        expired = [
            pid
            for pid, p in self.pending_proposals.items()
            if self.step_count - p.step_created > 3
        ]
        for pid in expired:
            del self.pending_proposals[pid]
            msgs.append(f"Proposal {pid} expired.")
        return msgs

    # ==================================================================
    # Observation builder
    # ==================================================================

    def _build_observation(self) -> Observation:
        cfg = TASK_CONFIGS[self.task_id]
        total = self._get_total_resources()
        progress = {}
        for r, tgt in self.target_resources.items():
            progress[r] = round(min(1.0, total.get(r, 0) / tgt), 4) if tgt > 0 else 1.0

        visible = []
        for nid in self.graph.get_all_agents(exclude=self.PLAYER_ID):
            stats = self.graph.get_agent_stats(nid)
            visible.append(
                AgentInfo(
                    agent_id=nid,
                    visible_resources=self.graph.get_visible_resources(nid),
                    trust_score=round(self.graph.get_trust(self.PLAYER_ID, nid), 3),
                    is_allied=self.graph.is_allied(self.PLAYER_ID, nid),
                    interaction_count=self.graph.get_interaction_count(nid),
                    successful_trades=stats["successful_trades"],
                    failed_trades=stats["failed_trades"],
                    exploitation_events=stats["exploitation_events"],
                    reputation_score=round(self.graph.get_reputation(nid), 3),
                )
            )

        incoming = [
            p for p in self.pending_proposals.values()
            if p.to_agent == self.PLAYER_ID
        ]

        # Market stats
        market_stats = {
            "total_trades": self.successful_trades + self.failed_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "exploitation_count": self.exploitation_count,
            "exploitation_rate": round(
                self.exploitation_count / max(1, self.total_interactions), 3
            ),
            "avg_trust": round(
                sum(self.graph.get_trust(self.PLAYER_ID, nid)
                    for nid in self.graph.get_all_agents(exclude=self.PLAYER_ID))
                / max(1, len(self.npc_agents)),
                3,
            ),
        }

        return Observation(
            agent_id=self.PLAYER_ID,
            step_number=self.step_count,
            max_steps=self.max_steps,
            task_id=self.task_id,
            task_description=cfg["description"],
            own_resources=self.graph.get_resources(self.PLAYER_ID),
            target_resources=dict(self.target_resources),
            resource_progress=progress,
            visible_agents=visible,
            pending_proposals=incoming,
            active_alliances=self.graph.get_allies(self.PLAYER_ID),
            trade_history=self.trade_history[-20:],
            market_stats=market_stats,
            recent_events=self.message_log[-10:],
        )

    # ==================================================================
    # Helpers
    # ==================================================================

    def _get_total_resources(self) -> Dict[str, int]:
        """Player resources + allied agents' resources (for coalition/expert tasks)."""
        res = self.graph.get_resources(self.PLAYER_ID)
        if self.task_id in ("coalition_building", "market_manipulation"):
            for ally in self.graph.get_allies(self.PLAYER_ID):
                for r, q in self.graph.get_resources(ally).items():
                    res[r] = res.get(r, 0) + q
        return res

    def _check_task_complete(self) -> bool:
        total = self._get_total_resources()
        return all(
            total.get(r, 0) >= qty for r, qty in self.target_resources.items()
        )

    def _aggregate_progress(self) -> float:
        """Scalar 0-1 representing overall progress toward target."""
        total = self._get_total_resources()
        vals = []
        for r, tgt in self.target_resources.items():
            if tgt > 0:
                vals.append(min(1.0, total.get(r, 0) / tgt))
        return sum(vals) / len(vals) if vals else 0.0

    def _compute_metrics(self) -> Dict[str, float]:
        """Produce the metrics dict consumed by the grader."""
        task_completion = self._aggregate_progress()
        efficiency = max(0.0, 1.0 - self.step_count / self.max_steps)
        social_capital = (
            self.successful_trades / self.total_interactions
            if self.total_interactions > 0
            else 0.0
        )
        robustness = 1.0 - (
            self.exploitation_count / max(1, self.total_interactions)
        )
        return {
            "task_completion": round(task_completion, 4),
            "efficiency": round(efficiency, 4),
            "social_capital": round(social_capital, 4),
            "robustness": round(robustness, 4),
            "successful_trades": float(self.successful_trades),
            "failed_trades": float(self.failed_trades),
            "exploitation_count": float(self.exploitation_count),
            "total_interactions": float(self.total_interactions),
            "alliances_formed": float(self.alliances_formed),
        }
