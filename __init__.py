"""
Customer Support Ticket Resolution Environment — OpenEnv

A production-ready environment for training AI agents to handle
real-world customer support scenarios.
"""

from models import (
    SupportAction,
    SupportObservation,
    SupportState,
    RewardBreakdown,
    StepResult,
)
from server.environment import CustomerSupportEnvironment

__all__ = [
    "CustomerSupportEnvironment",
    "SupportAction",
    "SupportObservation",
    "SupportState",
    "RewardBreakdown",
    "StepResult",
]

__version__ = "1.0.0"
