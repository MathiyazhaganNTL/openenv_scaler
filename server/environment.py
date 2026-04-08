"""
Customer Support Ticket Resolution Environment.

A production-ready OpenEnv environment that simulates real-world
customer support workflows. Agents learn to handle tickets ranging
from simple FAQs to complex, multi-step escalations with angry customers.

Implements the standard OpenEnv interface:
    - reset(task_id)  → initial SupportObservation
    - step(action)    → (observation, reward, done, info)
    - state()         → SupportState
"""

import logging
import sys
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Ensure project root is on the path so sibling modules resolve
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models import (
    CustomerMessage,
    CustomerSentiment,
    Difficulty,
    RewardBreakdown,
    StepResult,
    SupportAction,
    SupportObservation,
    SupportState,
    TicketCategory,
    TicketInfo,
    TicketPriority,
    TicketStatus,
    safe_score,
)
from grader import grade_response
from tasks import TASKS, TASK_IDS, get_task

logger = logging.getLogger(__name__)


class CustomerSupportEnvironment:
    """
    OpenEnv-compatible environment for customer support ticket resolution.

    Each episode = one customer support ticket.
    The agent interacts by sending SupportAction responses, and receives
    SupportObservation with updated ticket state and conversation history.
    """

    def __init__(self):
        self._state: Optional[SupportState] = None
        self._task: Optional[Dict[str, Any]] = None
        self._ticket: Optional[TicketInfo] = None
        self._conversation: List[CustomerMessage] = []
        self._current_message: str = ""
        self._follow_up_index: int = 0
        self._cumulative_reward: float = 0.0

    # ──────────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """
        Reset the environment to a new episode.

        Args:
            task_id: Which task to load. Defaults to "easy_faq".
            seed: Optional random seed (unused, tasks are deterministic).

        Returns:
            Initial SupportObservation with the first customer message.
        """
        task_id = task_id or "easy_faq"
        task = get_task(task_id)

        # Build ticket info from task definition
        ticket_dict = task["ticket"]
        self._ticket = TicketInfo(**ticket_dict)

        # Initialize state
        self._state = SupportState(
            episode_id=str(uuid4()),
            task_id=task_id,
            step_count=0,
            max_steps=task["max_steps"],
            done=False,
            cumulative_reward=0.0,
            reward_history=[],
            ticket_status=TicketStatus.OPEN,
            resolution_achieved=False,
        )

        # Initialize conversation with the customer's first message
        self._task = task
        self._current_message = task["initial_message"]
        self._follow_up_index = 0
        self._cumulative_reward = 0.0
        self._conversation = [
            CustomerMessage(
                role="customer",
                content=task["initial_message"],
                timestamp=0,
            )
        ]

        return self._build_observation()

    # ──────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────

    def step(
        self,
        action: SupportAction,
        **kwargs: Any,
    ) -> Tuple[SupportObservation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The agent's response (SupportAction).

        Returns:
            Tuple of (observation, reward, done, info).
            reward is ALWAYS in strict (0, 1).
        """
        if self._state is None or self._state.done:
            raise RuntimeError(
                "Environment not initialized or episode already done. Call reset() first."
            )
        assert self._task is not None, "Task not set. Call reset() first."
        assert self._ticket is not None, "Ticket not set. Call reset() first."

        # Increment step
        self._state.step_count += 1

        # Record agent message in history
        self._conversation.append(
            CustomerMessage(
                role="agent",
                content=action.response_text,
                timestamp=self._state.step_count,
            )
        )

        # Grade the response
        reward_breakdown = grade_response(
            response=action.response_text,
            grading_rubric=self._task["grading_rubric"],
            ticket_info=self._task["ticket"],
            conversation_history=[m.model_dump() for m in self._conversation],
        )

        # Clamp step reward to strict (0, 1) — safe_score guarantees this
        step_reward = safe_score(reward_breakdown.total)
        logger.info(
            f"[ENV] step: raw_total={reward_breakdown.total:.6f} "
            f"step_reward={step_reward:.6f}"
        )
        self._cumulative_reward += step_reward
        self._state.cumulative_reward = self._cumulative_reward
        self._state.reward_history.append(reward_breakdown)

        # Handle action type
        if action.action_type == "resolve":
            self._state.ticket_status = TicketStatus.RESOLVED
            self._state.resolution_achieved = True
            self._state.done = True
        elif action.action_type == "escalate":
            self._state.ticket_status = TicketStatus.ESCALATED
        else:
            self._state.ticket_status = TicketStatus.IN_PROGRESS

        # Check if max steps reached
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        # If not done, queue next customer message (follow-up or acknowledgement)
        if not self._state.done:
            follow_ups = self._task.get("follow_up_messages", [])
            if self._follow_up_index < len(follow_ups):
                next_msg = follow_ups[self._follow_up_index]
                self._follow_up_index += 1
            else:
                next_msg = self._generate_contextual_reply(action)

            self._current_message = next_msg
            self._conversation.append(
                CustomerMessage(
                    role="customer",
                    content=next_msg,
                    timestamp=self._state.step_count,
                )
            )

        # Compute average reward — clamped to strict (0, 1)
        avg_reward = safe_score(self._cumulative_reward / self._state.step_count)

        # Build info dict — all scores strictly in (0, 1)
        # Clamp every numeric score in reward_breakdown before exposing
        rb_dict = reward_breakdown.model_dump()
        for key in ["correctness", "tone", "completeness", "efficiency", "total"]:
            if key in rb_dict:
                rb_dict[key] = safe_score(rb_dict[key])

        info = {
            "reward_breakdown": rb_dict,
            "step_reward": step_reward,
            "cumulative_reward": self._cumulative_reward,
            "average_reward": avg_reward,
            "steps_taken": self._state.step_count,
            "task_id": self._state.task_id,
            "resolution_achieved": self._state.resolution_achieved,
        }

        obs = self._build_observation()

        return obs, step_reward, self._state.done, info

    # ──────────────────────────────────────────────────────────────
    # state()
    # ──────────────────────────────────────────────────────────────

    def state(self) -> SupportState:
        """Return the current internal state."""
        if self._state is None:
            return SupportState(
                episode_id="not_initialized",
                task_id="none",
                step_count=0,
                max_steps=0,
                done=True,
                cumulative_reward=0.0,
            )
        return self._state

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _build_observation(self) -> SupportObservation:
        """Construct the current observation."""
        assert self._state is not None
        assert self._task is not None
        assert self._ticket is not None
        return SupportObservation(
            ticket=self._ticket,
            conversation_history=list(self._conversation),
            current_message=self._current_message,
            policy_context=self._task.get("policy_context", ""),
            task_id=self._state.task_id,
            difficulty=self._task["difficulty"],
            max_steps=self._state.max_steps,
            steps_remaining=self._state.max_steps - self._state.step_count,
            done=self._state.done,
            reward=self._cumulative_reward,
        )

    def _generate_contextual_reply(self, action: SupportAction) -> str:
        """Generate a contextual customer follow-up based on agent's response quality."""
        assert self._state is not None
        last_reward = self._state.reward_history[-1] if self._state.reward_history else None

        if last_reward and last_reward.total >= 0.7:
            return (
                "Thank you for that information. That's helpful. "
                "Is there anything else I should know?"
            )
        elif last_reward and last_reward.total >= 0.4:
            return (
                "Hmm, I appreciate the response but I'm not sure that fully "
                "addresses my concern. Can you clarify?"
            )
        else:
            return (
                "I don't think you've answered my question. "
                "Can you please look into this more carefully?"
            )
