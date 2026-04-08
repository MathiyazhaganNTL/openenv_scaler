"""
Deterministic grading engine for the Customer Support Environment.

Follows the reference additive scoring pattern:
  - Category/keyword correctness  (+0.3)
  - Empathy detection             (+0.1 / +0.2)
  - Angry customer strict rule    (-0.25)
  - Anti-generic response penalty (-0.1)
  - Helpfulness detection         (+0.3)
  - Repetition penalty            (-0.2)
  - Escalation penalty            (-0.1)
  - Resolution bonus              (+0.2)
  - Efficiency bonus              (+0.1 * remaining steps)

Returns a RewardBreakdown with a total score in (0.0, 1.0) — strict open interval.

IMPORTANT — Every numeric score produced by this module is passed through
``safe_score`` before it leaves the grader so that the evaluator NEVER
receives a boundary value (0.0 or 1.0).
"""

import logging
import re
from typing import Any, Dict, List

from models import RewardBreakdown, safe_score

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    """Lower-case and strip extra whitespace for matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def grade_response(
    response: str,
    grading_rubric: Dict[str, Any],
    ticket_info: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
    action_type: str = "respond",
    step_count: int = 0,
    max_steps: int = 5,
) -> RewardBreakdown:
    """
    Grade an agent response using the reference additive scoring pattern.

    Args:
        response: The agent's response text
        grading_rubric: Task-specific grading criteria
        ticket_info: Ticket metadata
        conversation_history: Previous messages
        action_type: 'respond', 'escalate', or 'resolve'
        step_count: Current step number (1-indexed, already incremented)
        max_steps: Maximum allowed steps for this task

    Returns:
        RewardBreakdown with ALL scores in strict (0.0, 1.0) open interval.
    """
    score = 0.0
    metrics: Dict[str, float] = {}
    norm = _normalise(response)

    # ── 1. Correct category / keyword extraction (+0.3) ──
    correctness_criteria = grading_rubric.get("correctness", {}).get("criteria", [])
    correctness_hit = False
    for criterion in correctness_criteria:
        kw_group: List[str] = criterion.get("keyword_group", [])
        if any(kw.lower() in norm for kw in kw_group):
            correctness_hit = True
            break
    if correctness_hit:
        score += 0.3
        metrics["category_correct"] = 0.3

    # ── 2. Empathy check (+0.1 neutral, +0.2 angry/frustrated) ──
    sentiment = ticket_info.get("customer_sentiment", "neutral")
    empathy_words = ["sorry", "apologize", "understand", "help"]
    if any(word in norm for word in empathy_words):
        empathy_score = 0.2 if sentiment in ["angry", "frustrated"] else 0.1
        score += empathy_score
        metrics["empathy"] = empathy_score

    # ── 3. Angry customer strict rule (-0.25) ──
    if sentiment == "angry" and not any(
        w in norm for w in ["sorry", "apologize", "understand"]
    ):
        score -= 0.25
        metrics["angry_penalty"] = -0.25

    # ── 4. Anti-generic response penalty (-0.1) ──
    generic_phrases = ["i will help you", "let me help", "i understand your issue"]
    if any(phrase in norm for phrase in generic_phrases) and len(response) < 60:
        score -= 0.1
        metrics["generic_penalty"] = -0.1

    # ── 5. Helpfulness check (+0.3) ──
    helpful_words = [
        "step", "fix", "update", "here is", "resolved",
        "refund", "replacement", "process", "ship", "send",
        "return", "credit", "track", "label",
    ]
    if any(word in norm for word in helpful_words):
        score += 0.3
        metrics["helpfulness"] = 0.3

    # ── 6. Repetition penalty (-0.2) ──
    past_responses = [
        msg.get("content", "").lower()
        for msg in conversation_history
        if msg.get("role") == "agent"
    ]
    if norm in past_responses:
        score -= 0.2
        metrics["repetition_penalty"] = -0.2

    # ── 7. Escalation penalty (-0.1) ──
    if action_type == "escalate":
        score -= 0.1
        metrics["escalation_penalty"] = -0.1

    # ── 8. Resolution bonus (+0.2) & Efficiency bonus ──
    if action_type == "resolve":
        score += 0.2
        metrics["resolution_bonus"] = 0.2

        # Efficiency bonus: reward resolving in fewer steps
        if step_count < max_steps:
            efficiency_bonus = round(0.1 * (max_steps - step_count), 4)
            score += efficiency_bonus
            metrics["efficiency_bonus"] = efficiency_bonus

    # ── Final score — STRICT (0, 1) via safe_score ──
    final_score = safe_score(score)

    # Map metrics to RewardBreakdown fields
    correctness_val = safe_score(metrics.get("category_correct", 0.0))
    tone_val = safe_score(
        metrics.get("empathy", 0.0)
        + metrics.get("angry_penalty", 0.0)
        + metrics.get("generic_penalty", 0.0)
        + 0.3  # base tone
    )
    completeness_val = safe_score(
        metrics.get("helpfulness", 0.0)
        + metrics.get("resolution_bonus", 0.0)
    )
    efficiency_val = safe_score(
        metrics.get("efficiency_bonus", 0.0) + 0.2
    )
    penalties_total = sum(v for v in metrics.values() if v < 0)

    # Build explanation
    parts = [f"{k}: {v:.4f}" for k, v in sorted(metrics.items())]
    parts.append(f"Total: {final_score:.4f}")

    logger.info(f"[GRADER] score={final_score:.4f} metrics={metrics}")

    return RewardBreakdown(
        correctness=correctness_val,
        tone=tone_val,
        completeness=completeness_val,
        efficiency=efficiency_val,
        penalties=round(max(-1.0, min(0.0, penalties_total)), 4),
        total=final_score,
        explanation=" | ".join(parts),
    )
