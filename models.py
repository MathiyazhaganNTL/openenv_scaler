"""
Pydantic models for the Customer Support Ticket Resolution Environment.

Defines the Action, Observation, State, and Reward models used for
type-safe communication between the agent and environment.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────

class TicketCategory(str, Enum):
    FAQ = "faq"
    REFUND = "refund"
    COMPLAINT = "complaint"
    TECHNICAL = "technical"
    BILLING = "billing"
    SHIPPING = "shipping"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    AWAITING_CUSTOMER = "awaiting_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


class CustomerSentiment(str, Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ──────────────────────────────────────────────────────────────────
# Action Model
# ──────────────────────────────────────────────────────────────────

class SupportAction(BaseModel):
    """Action taken by the support agent."""
    response_text: str = Field(
        ...,
        description="The agent's response text to the customer",
        min_length=1,
        max_length=2000,
    )
    action_type: str = Field(
        default="respond",
        description="Type of action: 'respond', 'escalate', 'resolve', 'request_info'",
    )
    internal_notes: Optional[str] = Field(
        default=None,
        description="Internal notes for the support team (not visible to customer)",
    )


# ──────────────────────────────────────────────────────────────────
# Observation Model
# ──────────────────────────────────────────────────────────────────

class CustomerMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Either 'customer' or 'agent'")
    content: str = Field(..., description="Message content")
    timestamp: int = Field(..., description="Step number when message was sent")


class TicketInfo(BaseModel):
    """Information about the customer support ticket."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    category: TicketCategory = Field(..., description="Ticket category")
    priority: TicketPriority = Field(..., description="Ticket priority level")
    status: TicketStatus = Field(..., description="Current ticket status")
    customer_name: str = Field(..., description="Customer name")
    customer_sentiment: CustomerSentiment = Field(..., description="Customer emotional state")
    subject: str = Field(..., description="Ticket subject line")
    order_id: Optional[str] = Field(default=None, description="Related order ID if applicable")
    product_name: Optional[str] = Field(default=None, description="Related product if applicable")
    purchase_date: Optional[str] = Field(default=None, description="Purchase date if applicable")
    purchase_amount: Optional[float] = Field(default=None, description="Purchase amount if applicable")


class SupportObservation(BaseModel):
    """Observation returned to the agent after each step."""
    ticket: TicketInfo = Field(..., description="Current ticket information")
    conversation_history: List[CustomerMessage] = Field(
        default_factory=list,
        description="Full conversation history",
    )
    current_message: str = Field(..., description="Latest customer message to respond to")
    available_actions: List[str] = Field(
        default_factory=lambda: ["respond", "escalate", "resolve", "request_info"],
        description="Available action types",
    )
    policy_context: str = Field(
        default="",
        description="Relevant company policy information for the agent",
    )
    task_id: str = Field(..., description="Current task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty level")
    max_steps: int = Field(default=5, description="Maximum steps allowed for this task")
    steps_remaining: int = Field(default=5, description="Steps left before timeout")
    done: bool = Field(default=False, description="Whether the episode is complete")
    reward: float = Field(default=0.0, description="Cumulative reward so far")


# ──────────────────────────────────────────────────────────────────
# Reward Model
# ──────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Detailed breakdown of the reward score."""
    correctness: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Score for factual correctness (0.0-1.0)",
    )
    tone: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Score for professional tone (0.0-1.0)",
    )
    completeness: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Score for response completeness (0.0-1.0)",
    )
    efficiency: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Score for resolution efficiency (0.0-1.0)",
    )
    penalties: float = Field(
        default=0.0,
        ge=-1.0, le=0.0,
        description="Penalty deductions (negative value)",
    )
    total: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Overall weighted score (0.0-1.0)",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the score",
    )


# ──────────────────────────────────────────────────────────────────
# State Model
# ──────────────────────────────────────────────────────────────────

class SupportState(BaseModel):
    """Internal state of the environment."""
    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field(..., description="Current task ID")
    step_count: int = Field(default=0, description="Number of steps taken")
    max_steps: int = Field(default=5, description="Maximum steps allowed")
    done: bool = Field(default=False, description="Whether episode is finished")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated")
    reward_history: List[RewardBreakdown] = Field(
        default_factory=list,
        description="History of reward breakdowns per step",
    )
    ticket_status: TicketStatus = Field(
        default=TicketStatus.OPEN,
        description="Current ticket status",
    )
    resolution_achieved: bool = Field(
        default=False,
        description="Whether the ticket was successfully resolved",
    )


# ──────────────────────────────────────────────────────────────────
# Step Result (matches OpenEnv convention)
# ──────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Result returned from step(), matching OpenEnv convention."""
    observation: SupportObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
