"""
Deterministic grading engine for the Customer Support Environment.

Evaluates agent responses on three axes:
  - Correctness  (keyword / concept matching)
  - Tone         (positive vs. negative signal detection)
  - Completeness (checklist of required response elements)

Returns a RewardBreakdown with a total score in (0.0, 1.0) — strict open interval.

IMPORTANT — Every numeric score produced by this module is passed through
``normalize_score`` before it leaves the grader so that the evaluator NEVER
receives a boundary value (0.0 or 1.0).
"""

import re
from typing import Any, Dict, List

from models import RewardBreakdown


# ──────────────────────────────────────────────────────────────────
# Central score normaliser — THE single source of truth
# ──────────────────────────────────────────────────────────────────

# Strict open-interval bounds: scores must never be exactly 0.0 or 1.0
_SCORE_FLOOR = 0.0001
_SCORE_CEIL  = 0.9999


def normalize_score(value: Any) -> float:
    """Clamp *value* into the strict open interval (0, 1).

    * ``None``  → 0.5
    * anything that cannot be converted to float → 0.5
    * values ≤ 0 → ``_SCORE_FLOOR``
    * values ≥ 1 → ``_SCORE_CEIL``
    """
    if value is None:
        return 0.5
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    # Guard against NaN / Inf
    if v != v or v == float('inf') or v == float('-inf'):
        return 0.5
    return max(_SCORE_FLOOR, min(_SCORE_CEIL, v))


def _normalise(text: str) -> str:
    """Lower-case and strip extra whitespace for matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ──────────────────────────────────────────────────────────────────
# Correctness scorer
# ──────────────────────────────────────────────────────────────────

def _score_correctness(
    response: str,
    rubric: Dict[str, Any],
) -> float:
    """Score based on presence of expected keyword groups.

    Returns a value in (0, 1) — never 0.0 or 1.0.
    """
    norm = _normalise(response)
    criteria = rubric.get("criteria", [])
    if not criteria:
        # No rubric → return a safe neutral score, never 0.0
        return normalize_score(0.1)

    total = 0.0
    for criterion in criteria:
        kw_group: List[str] = criterion.get("keyword_group", [])
        points: float = criterion.get("points", 0.0)
        # Award points if ANY keyword in the group is found
        if any(kw.lower() in norm for kw in kw_group):
            total += points

    return normalize_score(total)


# ──────────────────────────────────────────────────────────────────
# Tone scorer
# ──────────────────────────────────────────────────────────────────

def _score_tone(
    response: str,
    rubric: Dict[str, Any],
) -> float:
    """
    Score tone based on positive and negative signal presence.
    Start at 0.5, boost for positive signals, penalize for negative signals.

    Returns a value in (0, 1) — never 0.0 or 1.0.
    """
    norm = _normalise(response)
    criteria = rubric.get("criteria", {})

    positive_signals: List[str] = criteria.get("positive_signals", [])
    negative_signals: List[str] = criteria.get("negative_signals", [])

    # Count matches
    pos_count = sum(1 for sig in positive_signals if sig.lower() in norm)
    neg_count = sum(1 for sig in negative_signals if sig.lower() in norm)

    # Base score: 0.5 (neutral)
    score = 0.5

    # Each positive signal adds points (diminishing returns)
    if positive_signals:
        pos_ratio = pos_count / len(positive_signals)
        score += pos_ratio * 0.4  # max +0.4 from positives (keeps below 1.0)

    # Each negative signal deducts heavily
    if neg_count > 0:
        score -= min(neg_count * 0.2, 0.4)  # max -0.4 from negatives (keeps above 0.0)

    # Additional length/quality checks
    word_count = len(norm.split())
    if word_count < 10:
        score -= 0.1  # Too terse is often rude

    # Check if response uses ALL CAPS excessively
    upper_ratio = sum(1 for c in response if c.isupper()) / max(len(response), 1)
    if upper_ratio > 0.4 and len(response) > 20:
        score -= 0.05  # Shouting in response

    return normalize_score(score)


# ──────────────────────────────────────────────────────────────────
# Completeness scorer
# ──────────────────────────────────────────────────────────────────

def _score_completeness(
    response: str,
    rubric: Dict[str, Any],
    ticket_info: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
) -> float:
    """Score based on completeness checklist.

    Returns a value in (0, 1) — never 0.0 or 1.0.
    """
    norm = _normalise(response)
    criteria = rubric.get("criteria", [])
    if not criteria:
        # No rubric → return a safe neutral score, never 0.0
        return normalize_score(0.1)

    total = 0.0
    for criterion in criteria:
        check = criterion.get("check", "")
        points = criterion.get("points", 0.0)

        if check == "addresses_question" or check == "addresses_defect":
            # Check if response directly addresses the main issue
            subject = _normalise(ticket_info.get("subject", ""))
            subject_words = [w for w in subject.split() if len(w) > 3]
            if any(w in norm for w in subject_words) or len(norm.split()) > 20:
                total += points

        elif check == "provides_next_steps":
            # Check for actionable next steps
            step_indicators = [
                "will", "can", "please", "next step", "process",
                "we'll", "i'll", "going to", "let me", "i can",
                "here's what", "here is what", "follow up",
            ]
            if any(ind in norm for ind in step_indicators):
                total += points

        elif check == "references_order":
            # Check if the specific order ID is referenced
            order_id = ticket_info.get("order_id", "")
            if order_id and order_id.lower() in norm:
                total += points
            elif "order" in norm:
                total += points * 0.5  # Partial credit for mentioning order

        elif check == "explains_policy":
            # Check if relevant policy details are mentioned
            policy_terms = [
                "policy", "within", "days", "eligible", "qualify",
                "terms", "condition", "guideline",
            ]
            if sum(1 for t in policy_terms if t in norm) >= 2:
                total += points

        elif check == "provides_process":
            # Check if return/refund process is outlined
            process_terms = [
                "step", "first", "then", "send", "ship", "return",
                "label", "process", "receive", "refund",
            ]
            if sum(1 for t in process_terms if t in norm) >= 3:
                total += points

        elif check == "offers_options":
            # Check if multiple options are presented
            option_indicators = ["or", "option", "alternative", "either", "choose", "prefer"]
            if any(ind in norm for ind in option_indicators):
                total += points

        elif check == "acknowledges_all_issues":
            # For hard task: must address multiple issues
            issues_to_address = ["wrong", "late", "delay", "rude", "staff", "agent"]
            addressed = sum(1 for iss in issues_to_address if iss in norm)
            if addressed >= 3:
                total += points
            elif addressed >= 2:
                total += points * 0.6
            elif addressed >= 1:
                total += points * 0.3

        elif check == "concrete_resolution":
            # Check for concrete actions, not just apologies
            concrete_terms = [
                "refund", "replacement", "ship", "send", "credit",
                "discount", "expedite", "priority", "immediately",
                "right away", "today",
            ]
            if sum(1 for t in concrete_terms if t in norm) >= 2:
                total += points

        elif check == "timeline":
            # Check if specific timelines are given
            time_patterns = [
                r"\d+\s*(hour|day|week|business day)",
                r"within\s+\d+",
                r"by\s+(end of|tomorrow|today)",
                r"immediately",
                r"right away",
                r"asap",
                r"as soon as",
            ]
            if any(re.search(pat, norm) for pat in time_patterns):
                total += points

        elif check == "empathy":
            # Check for empathetic language
            empathy_terms = [
                "understand", "frustrat", "sorry", "apologize",
                "inconvenience", "disappoint", "concern",
                "appreciate your patience", "I hear you",
            ]
            if sum(1 for t in empathy_terms if t in norm) >= 2:
                total += points

        elif check == "follow_up_plan":
            # Check for follow-up commitments
            follow_up_terms = [
                "follow up", "follow-up", "check back", "update you",
                "keep you informed", "contact you", "reach out",
                "email you", "confirmation",
            ]
            if any(t in norm for t in follow_up_terms):
                total += points

    return normalize_score(total)


# ──────────────────────────────────────────────────────────────────
# Penalty computation
# ──────────────────────────────────────────────────────────────────

def _compute_penalties(
    response: str,
    conversation_history: List[Dict[str, Any]],
) -> float:
    """
    Compute penalties for bad behaviours.
    Returns a negative value in [-0.5, 0.0].
    """
    norm = _normalise(response)
    penalty = 0.0

    # Penalty: empty or near-empty response
    if len(norm.split()) < 5:
        penalty -= 0.2

    # Penalty: repeated response (copy-paste from previous)
    if conversation_history:
        prev_agent_msgs = [
            _normalise(m.get("content", ""))
            for m in conversation_history
            if m.get("role") == "agent"
        ]
        for prev in prev_agent_msgs:
            if prev and norm == prev:
                penalty -= 0.2
                break
            elif prev and len(prev) > 20 and prev in norm:
                penalty -= 0.1
                break

    # Penalty: harmful/inappropriate content
    harmful_patterns = [
        "kill", "die", "hate you", "shut up", "idiot",
        "moron", "loser", "go away",
    ]
    if any(pat in norm for pat in harmful_patterns):
        penalty -= 0.3

    # Penalty: completely irrelevant response
    irrelevant_signals = [
        "weather", "recipe", "joke", "game score",
        "political", "stock market",
    ]
    if sum(1 for s in irrelevant_signals if s in norm) >= 2:
        penalty -= 0.3

    return max(-0.5, penalty)


# ──────────────────────────────────────────────────────────────────
# Main grading function
# ──────────────────────────────────────────────────────────────────

def grade_response(
    response: str,
    grading_rubric: Dict[str, Any],
    ticket_info: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
) -> RewardBreakdown:
    """
    Grade an agent response and return a detailed RewardBreakdown.

    Args:
        response: The agent's response text
        grading_rubric: Task-specific grading criteria
        ticket_info: Ticket metadata
        conversation_history: Previous messages

    Returns:
        RewardBreakdown with ALL scores in strict (0.0, 1.0) open interval
    """
    # Score each axis — normalize_score guarantees (0, 1)
    correctness = normalize_score(_score_correctness(
        response,
        grading_rubric.get("correctness", {}),
    ))
    tone = normalize_score(_score_tone(
        response,
        grading_rubric.get("tone", {}),
    ))
    completeness = normalize_score(_score_completeness(
        response,
        grading_rubric.get("completeness", {}),
        ticket_info,
        conversation_history,
    ))

    # Get weights
    w_correctness = grading_rubric.get("correctness", {}).get("weight", 0.33)
    w_tone = grading_rubric.get("tone", {}).get("weight", 0.33)
    w_completeness = grading_rubric.get("completeness", {}).get("weight", 0.34)

    # Compute penalties (capped at -0.5)
    penalties = _compute_penalties(response, conversation_history)

    # Weighted total (before penalties)
    weighted = (
        correctness * w_correctness
        + tone * w_tone
        + completeness * w_completeness
    )

    # Apply penalties — normalize_score guarantees strict (0, 1)
    total = normalize_score(weighted + penalties)

    # The efficiency field re-uses the weighted pre-penalty score
    efficiency = normalize_score(weighted)

    # Debug logging
    print(f"[DEBUG] correctness={correctness:.4f} tone={tone:.4f} "
          f"completeness={completeness:.4f} weighted={weighted:.4f} "
          f"penalties={penalties:.4f} total={total:.4f}")

    # Build explanation
    parts = []
    parts.append(f"Correctness: {correctness:.4f} (weight={w_correctness:.2f})")
    parts.append(f"Tone: {tone:.4f} (weight={w_tone:.2f})")
    parts.append(f"Completeness: {completeness:.4f} (weight={w_completeness:.2f})")
    if penalties < 0:
        parts.append(f"Penalties: {penalties:.4f}")
    parts.append(f"Total: {total:.4f}")

    return RewardBreakdown(
        correctness=normalize_score(correctness),
        tone=normalize_score(tone),
        completeness=normalize_score(completeness),
        efficiency=normalize_score(efficiency),
        penalties=round(penalties, 4),
        total=normalize_score(total),
        explanation=" | ".join(parts),
    )
