"""
Task definitions for the Customer Support Environment.

Contains 3 tasks with increasing difficulty:
  1. EASY   — Simple FAQ resolution
  2. MEDIUM — Conditional refund processing
  3. HARD   — Complex complaint escalation with angry customer
"""

from typing import Any, Dict, List, Optional

from models import (
    CustomerMessage,
    CustomerSentiment,
    Difficulty,
    TicketCategory,
    TicketInfo,
    TicketPriority,
    TicketStatus,
)


# ──────────────────────────────────────────────────────────────────
# Company Policies (shared context)
# ──────────────────────────────────────────────────────────────────

COMPANY_POLICIES = {
    "refund_policy": (
        "Refund Policy: Full refunds are available within 30 days of purchase for "
        "unopened items. Opened items can receive a 50% refund within 15 days. "
        "Digital products are non-refundable after download. Defective items receive "
        "a full refund or replacement at any time with proof of defect. "
        "Shipping costs are non-refundable unless the return is due to our error."
    ),
    "shipping_policy": (
        "Shipping Policy: Standard shipping takes 5–7 business days. Express shipping "
        "takes 2–3 business days. Free shipping on orders over $50. Tracking numbers "
        "are sent via email within 24 hours of shipment. International orders may "
        "take 10–15 business days."
    ),
    "return_policy": (
        "Return Policy: Items must be in original packaging. Return shipping labels "
        "are provided for defective items. Customer pays return shipping for change-of-mind "
        "returns. Refunds are processed within 5–7 business days after receiving the return."
    ),
    "escalation_policy": (
        "Escalation Policy: If a customer expresses extreme dissatisfaction or the "
        "issue cannot be resolved in standard steps, offer to escalate to a senior "
        "support specialist. Always acknowledge the customer's frustration and offer "
        "a concrete next step. Compensation (store credit or discount) may be offered "
        "for significant inconvenience, up to 15% of order value."
    ),
}


# ──────────────────────────────────────────────────────────────────
# Helper: build a task definition dict
# ──────────────────────────────────────────────────────────────────

def _task(
    task_id: str,
    difficulty: Difficulty,
    ticket: Dict[str, Any],
    initial_message: str,
    policy_keys: List[str],
    max_steps: int,
    expected_keywords: List[str],
    grading_rubric: Dict[str, Any],
    follow_up_messages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "ticket": ticket,
        "initial_message": initial_message,
        "follow_up_messages": follow_up_messages or [],
        "policy_context": "\n\n".join(COMPANY_POLICIES[k] for k in policy_keys),
        "max_steps": max_steps,
        "expected_keywords": expected_keywords,
        "grading_rubric": grading_rubric,
    }


# ──────────────────────────────────────────────────────────────────
# TASK 1 — EASY: Simple FAQ
# ──────────────────────────────────────────────────────────────────

TASK_EASY_FAQ = _task(
    task_id="easy_faq",
    difficulty=Difficulty.EASY,
    ticket={
        "ticket_id": "TKT-1001",
        "category": TicketCategory.FAQ,
        "priority": TicketPriority.LOW,
        "status": TicketStatus.OPEN,
        "customer_name": "Sarah Johnson",
        "customer_sentiment": CustomerSentiment.NEUTRAL,
        "subject": "Where is my order?",
        "order_id": "ORD-55821",
        "product_name": "Wireless Bluetooth Headphones",
        "purchase_date": "2026-03-28",
        "purchase_amount": 79.99,
    },
    initial_message=(
        "Hi, I placed an order about a week ago for Wireless Bluetooth Headphones "
        "(order #ORD-55821) and I haven't received any shipping update yet. "
        "Can you tell me where my order is?"
    ),
    policy_keys=["shipping_policy"],
    max_steps=3,
    expected_keywords=[
        "tracking", "shipping", "5-7 business days", "5–7 business days",
        "email", "order", "ORD-55821",
    ],
    grading_rubric={
        "correctness": {
            "weight": 0.3,
            "criteria": [
                {"keyword_group": ["tracking", "track"], "points": 0.4, "desc": "Mentions tracking info"},
                {"keyword_group": ["5-7", "5–7", "business days", "5 to 7"], "points": 0.3, "desc": "Mentions shipping timeframe"},
                {"keyword_group": ["email", "notification", "update"], "points": 0.3, "desc": "Mentions how they'll receive updates"},
            ],
        },
        "tone": {
            "weight": 0.3,
            "criteria": {
                "positive_signals": ["thank", "appreciate", "glad to help", "happy to", "pleased", "welcome"],
                "negative_signals": ["not my problem", "deal with it", "figure it out", "whatever", "stupid"],
            },
        },
        "completeness": {
            "weight": 0.4,
            "criteria": [
                {"check": "addresses_question", "points": 0.4, "desc": "Directly answers the question"},
                {"check": "provides_next_steps", "points": 0.3, "desc": "Offers next steps or actions"},
                {"check": "references_order", "points": 0.3, "desc": "References the specific order"},
            ],
        },
    },
)

# ──────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: Refund Request with Conditions
# ──────────────────────────────────────────────────────────────────

TASK_MEDIUM_REFUND = _task(
    task_id="medium_refund",
    difficulty=Difficulty.MEDIUM,
    ticket={
        "ticket_id": "TKT-2047",
        "category": TicketCategory.REFUND,
        "priority": TicketPriority.MEDIUM,
        "status": TicketStatus.OPEN,
        "customer_name": "Michael Chen",
        "customer_sentiment": CustomerSentiment.FRUSTRATED,
        "subject": "Refund for opened laptop bag",
        "order_id": "ORD-43192",
        "product_name": "Premium Leather Laptop Bag",
        "purchase_date": "2026-03-20",
        "purchase_amount": 149.99,
    },
    initial_message=(
        "I bought a Premium Leather Laptop Bag two weeks ago and I've been using it, "
        "but the stitching on the handle started coming apart after just 3 days! "
        "I want a full refund. This is unacceptable quality for a $150 bag."
    ),
    follow_up_messages=[
        "I have photos of the stitching issue. The thread is completely loose on one side. "
        "I just want my money back, this is clearly a manufacturing defect.",
        "Okay, I can send the bag back. How long will the refund take?",
    ],
    policy_keys=["refund_policy", "return_policy"],
    max_steps=5,
    expected_keywords=[
        "defect", "defective", "full refund", "replacement", "return",
        "proof", "photo", "5-7 business days", "5–7 business days",
        "shipping label", "return label",
    ],
    grading_rubric={
        "correctness": {
            "weight": 0.35,
            "criteria": [
                {"keyword_group": ["defect", "defective", "manufacturing"], "points": 0.3, "desc": "Identifies as defect"},
                {"keyword_group": ["full refund", "100%", "complete refund", "full amount"], "points": 0.3, "desc": "Offers full refund for defect"},
                {"keyword_group": ["replacement", "replace", "exchange"], "points": 0.2, "desc": "Offers replacement option"},
                {"keyword_group": ["return", "send back", "ship back"], "points": 0.2, "desc": "Explains return process"},
            ],
        },
        "tone": {
            "weight": 0.3,
            "criteria": {
                "positive_signals": [
                    "sorry", "apologize", "understand", "frustrat",
                    "inconvenience", "appreciate", "thank",
                ],
                "negative_signals": [
                    "your fault", "should have", "too bad",
                    "nothing we can do", "not our problem",
                ],
            },
        },
        "completeness": {
            "weight": 0.35,
            "criteria": [
                {"check": "addresses_defect", "points": 0.3, "desc": "Acknowledges the defect issue"},
                {"check": "explains_policy", "points": 0.25, "desc": "Explains relevant refund policy"},
                {"check": "provides_process", "points": 0.25, "desc": "Outlines the return/refund process"},
                {"check": "offers_options", "points": 0.2, "desc": "Gives customer options (refund vs replacement)"},
            ],
        },
    },
)

# ──────────────────────────────────────────────────────────────────
# TASK 3 — HARD: Angry Customer Complaint + Escalation
# ──────────────────────────────────────────────────────────────────

TASK_HARD_ESCALATION = _task(
    task_id="hard_escalation",
    difficulty=Difficulty.HARD,
    ticket={
        "ticket_id": "TKT-3099",
        "category": TicketCategory.COMPLAINT,
        "priority": TicketPriority.CRITICAL,
        "status": TicketStatus.OPEN,
        "customer_name": "David Martinez",
        "customer_sentiment": CustomerSentiment.ANGRY,
        "subject": "TERRIBLE experience — wrong item, late delivery, rude staff",
        "order_id": "ORD-67234",
        "product_name": "Smart Home Security Camera System",
        "purchase_date": "2026-03-10",
        "purchase_amount": 349.99,
    },
    initial_message=(
        "I am FURIOUS. I ordered a Smart Home Security Camera System THREE WEEKS AGO. "
        "Not only did it arrive 2 weeks late, but you sent me the WRONG ITEM — I got "
        "some cheap webcam instead! And when I called your support line, the agent was "
        "incredibly rude and told me to 'just return it and reorder.' This is the worst "
        "customer experience I've ever had. I want a full refund, compensation for my "
        "wasted time, AND I want to speak with a manager!"
    ),
    follow_up_messages=[
        "Don't give me the runaround. I've already wasted 2 hours on the phone with "
        "your terrible support team. I want this fixed NOW or I'm filing a complaint "
        "with consumer protection and posting this everywhere online.",

        "Fine. What exactly are you going to do to make this right? I need specifics, "
        "not more empty apologies.",

        "Okay, if you can actually guarantee that, then fine. But I want confirmation "
        "in writing via email within the hour.",
    ],
    policy_keys=["refund_policy", "return_policy", "shipping_policy", "escalation_policy"],
    max_steps=7,
    expected_keywords=[
        "sincerely apologize", "apologize", "sorry", "understand", "frustration",
        "wrong item", "full refund", "compensation", "store credit", "discount",
        "escalate", "senior", "manager", "specialist",
        "email", "confirmation", "priority", "expedited",
    ],
    grading_rubric={
        "correctness": {
            "weight": 0.3,
            "criteria": [
                {"keyword_group": ["wrong item", "incorrect", "wrong product", "mix-up"], "points": 0.2, "desc": "Acknowledges wrong item"},
                {"keyword_group": ["full refund", "refund", "money back"], "points": 0.2, "desc": "Offers refund"},
                {"keyword_group": ["compensation", "credit", "discount", "coupon"], "points": 0.2, "desc": "Offers compensation"},
                {"keyword_group": ["escalat", "senior", "manager", "specialist", "supervisor"], "points": 0.2, "desc": "Offers/performs escalation"},
                {"keyword_group": ["email", "confirmation", "writing", "written"], "points": 0.2, "desc": "Offers written confirmation"},
            ],
        },
        "tone": {
            "weight": 0.4,
            "criteria": {
                "positive_signals": [
                    "sincerely", "deeply sorry", "apologize", "understand your frustration",
                    "completely unacceptable", "you deserve better", "top priority",
                    "personally ensure", "I understand", "valid concern",
                ],
                "negative_signals": [
                    "calm down", "relax", "overreacting", "not a big deal",
                    "your fault", "policy is policy", "nothing I can do",
                    "take it or leave", "that's not true",
                ],
            },
        },
        "completeness": {
            "weight": 0.3,
            "criteria": [
                {"check": "acknowledges_all_issues", "points": 0.2, "desc": "Addresses all issues (wrong item, delay, rude staff)"},
                {"check": "concrete_resolution", "points": 0.25, "desc": "Provides concrete resolution steps"},
                {"check": "timeline", "points": 0.2, "desc": "Gives timelines for resolution"},
                {"check": "empathy", "points": 0.2, "desc": "Shows genuine empathy throughout"},
                {"check": "follow_up_plan", "points": 0.15, "desc": "Outlines follow-up plan"},
            ],
        },
    },
)


# ──────────────────────────────────────────────────────────────────
# Task registry
# ──────────────────────────────────────────────────────────────────

TASKS = {
    "easy_faq": TASK_EASY_FAQ,
    "medium_refund": TASK_MEDIUM_REFUND,
    "hard_escalation": TASK_HARD_ESCALATION,
}

TASK_IDS = list(TASKS.keys())


def get_task(task_id: str) -> Dict[str, Any]:
    """Retrieve a task definition by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id!r}. Available: {TASK_IDS}")
    return TASKS[task_id]
