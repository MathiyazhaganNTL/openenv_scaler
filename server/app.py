"""
FastAPI application exposing the Customer Support Environment
via HTTP endpoints compatible with OpenEnv specification.

Endpoints:
    POST /reset        — Reset environment, returns initial observation
    POST /step         — Execute an action, returns (obs, reward, done, info)
    GET  /state        — Get current internal state
    GET  /health       — Health check
    GET  /tasks        — List available tasks
    GET  /             — Environment info
"""

import sys
import os

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import SupportAction, SupportObservation, SupportState
from server.environment import CustomerSupportEnvironment
from tasks import TASK_IDS, TASKS


# ──────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="easy_faq", description="Task ID to load")
    seed: Optional[int] = Field(default=None, description="Random seed (unused)")


class StepRequest(BaseModel):
    action: SupportAction = Field(..., description="Agent action")


class StepResponse(BaseModel):
    observation: SupportObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int


# ──────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Support Environment — OpenEnv",
    description=(
        "AI-Powered Customer Support Ticket Resolution Environment. "
        "Train agents to handle real customer issues using step/reset/state APIs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-agent mode for simplicity)
env = CustomerSupportEnvironment()


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["info"])
def root():
    """Environment info and available endpoints."""
    return {
        "name": "customer_support_env",
        "version": "1.0.0",
        "description": "AI-Powered Customer Support Ticket Resolution Environment",
        "endpoints": {
            "POST /reset": "Reset environment with a task_id",
            "POST /step": "Execute an action",
            "GET /state": "Get current state",
            "GET /health": "Health check",
            "GET /tasks": "List available tasks",
        },
        "available_tasks": TASK_IDS,
    }


@app.get("/health", tags=["health"])
def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "customer_support_env"}


@app.get("/tasks", response_model=list[TaskInfo], tags=["tasks"])
def list_tasks():
    """List all available tasks with metadata."""
    result = []
    for tid, task in TASKS.items():
        result.append(
            TaskInfo(
                task_id=tid,
                name=task["ticket"]["subject"],
                description=f"{task['difficulty'].value.upper()} — {task['ticket']['subject']}",
                difficulty=task["difficulty"].value,
                max_steps=task["max_steps"],
            )
        )
    return result


@app.post("/reset", response_model=SupportObservation, tags=["environment"])
def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment and return the initial observation."""
    try:
        obs = env.reset(task_id=request.task_id, seed=request.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["environment"])
def step(request: StepRequest):
    """Execute an agent action and return the result."""
    try:
        obs, reward, done, info = env.step(action=request.action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=SupportState, tags=["environment"])
def get_state():
    """Get the current internal state of the environment."""
    return env.state()


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    """Run the server directly."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
