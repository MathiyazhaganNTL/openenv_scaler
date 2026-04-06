"""
Baseline Inference Script for the Customer Support Environment.

This script runs an AI agent through all tasks and computes final scores.
It uses the OpenAI-compatible API to generate agent responses.

Environment Variables:
    API_BASE_URL      — Base URL for the LLM API (default: https://api.openai.com/v1)
    MODEL_NAME        — Model to use (default: gpt-3.5-turbo)
    HF_TOKEN          — Hugging Face token (no default)
    LOCAL_IMAGE_NAME  — Optional: local Docker image name when using from_docker_image()
    ENV_BASE_URL      — Base URL for the environment server (default: http://localhost:8000)

Usage:
    python inference.py
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────
# Configuration  (checklist-compliant env var declarations)
# ──────────────────────────────────────────────────────────────────

# Defaults allowed only for API_BASE_URL and MODEL_NAME
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-3.5-turbo")

# No default for HF_TOKEN (required by checklist)
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — only needed when using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# Resolve API key: prefer HF_TOKEN, fall back to empty string
_api_key = HF_TOKEN or ""

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# LLM Client  (uses OpenAI SDK — required by checklist item 4)
# ──────────────────────────────────────────────────────────────────

# Initialise the OpenAI-compatible client once at module level
_llm_client = OpenAI(
    api_key=_api_key,
    base_url=API_BASE_URL,
)


def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    Call the LLM via the OpenAI SDK client.

    Returns:
        The assistant's response text.
    """
    try:
        completion = _llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[ERROR] LLM call failed: {e}")
        return "I apologize for the inconvenience. Let me look into this for you right away."


# ──────────────────────────────────────────────────────────────────
# Environment Client
# ──────────────────────────────────────────────────────────────────

class EnvClient:
    """Simple HTTP client for the Customer Support Environment."""

    def __init__(self, base_url: str = ENV_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy_faq") -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, response_text: str, action_type: str = "respond") -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/step",
            json={
                "action": {
                    "response_text": response_text,
                    "action_type": action_type,
                }
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ──────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional customer support agent for an e-commerce company.

Your responsibilities:
1. Respond to customer inquiries with empathy and professionalism
2. Provide accurate information based on company policies
3. Resolve issues efficiently while maintaining customer satisfaction
4. Escalate complex issues when appropriate

Guidelines:
- Always address the customer by name when possible
- Acknowledge their feelings and concerns
- Provide specific, actionable information
- Reference order numbers and relevant details
- Offer concrete solutions with timelines
- Maintain a warm, professional tone throughout

Company Policy Context (use this to inform your responses):
{policy_context}
"""


# ──────────────────────────────────────────────────────────────────
# Build conversation messages for the LLM
# ──────────────────────────────────────────────────────────────────

def build_messages(
    observation: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build the message list for the LLM from the current observation."""
    # System prompt with policy context
    system_msg = SYSTEM_PROMPT.format(
        policy_context=observation.get("policy_context", "No specific policy context provided."),
    )

    messages = [{"role": "system", "content": system_msg}]

    # Add conversation history
    for msg in observation.get("conversation_history", []):
        role = "user" if msg["role"] == "customer" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    # Add ticket context to the first user message
    ticket = observation.get("ticket", {})
    ticket_context = (
        f"\n\n[Ticket Info — visible only to you]\n"
        f"Ticket ID: {ticket.get('ticket_id', 'N/A')}\n"
        f"Customer: {ticket.get('customer_name', 'N/A')}\n"
        f"Category: {ticket.get('category', 'N/A')}\n"
        f"Priority: {ticket.get('priority', 'N/A')}\n"
        f"Sentiment: {ticket.get('customer_sentiment', 'N/A')}\n"
        f"Subject: {ticket.get('subject', 'N/A')}\n"
        f"Order ID: {ticket.get('order_id', 'N/A')}\n"
        f"Product: {ticket.get('product_name', 'N/A')}\n"
        f"Purchase Date: {ticket.get('purchase_date', 'N/A')}\n"
        f"Purchase Amount: ${ticket.get('purchase_amount', 0):.2f}\n"
    )

    # Inject ticket context into the last user message
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += ticket_context

    return messages


# ──────────────────────────────────────────────────────────────────
# Run single task
# ──────────────────────────────────────────────────────────────────

def run_task(env_client: EnvClient, task_id: str) -> Dict[str, Any]:
    """
    Run a single task to completion and return results.
    """
    logger.info(f"[START] task_id={task_id}")
    start_time = time.time()

    # Reset the environment
    obs = env_client.reset(task_id=task_id)
    logger.info(f"[STEP] task={task_id} step=0 type=reset customer_message=\"{obs['current_message'][:80]}...\"")

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        # Build messages for the LLM
        messages = build_messages(obs)

        # Get LLM response
        agent_response = call_llm(messages)

        # Determine action type
        action_type = "respond"
        steps_remaining = obs.get("steps_remaining", 1)
        if steps_remaining <= 1:
            action_type = "resolve"  # Auto-resolve on last step

        # Step the environment
        result = env_client.step(
            response_text=agent_response,
            action_type=action_type,
        )

        step_count += 1
        step_reward = result.get("reward", 0.0)
        total_reward += step_reward
        done = result.get("done", False)
        obs = result.get("observation", {})
        info = result.get("info", {})

        # Log step
        reward_breakdown = info.get("reward_breakdown", {})
        logger.info(
            f"[STEP] task={task_id} step={step_count} "
            f"reward={step_reward:.4f} "
            f"correctness={reward_breakdown.get('correctness', 0):.2f} "
            f"tone={reward_breakdown.get('tone', 0):.2f} "
            f"completeness={reward_breakdown.get('completeness', 0):.2f} "
            f"done={done}"
        )

    # Compute average reward for this task
    avg_reward = total_reward / max(step_count, 1)
    elapsed = time.time() - start_time

    logger.info(
        f"[END] task_id={task_id} "
        f"steps={step_count} "
        f"total_reward={total_reward:.4f} "
        f"avg_reward={avg_reward:.4f} "
        f"elapsed={elapsed:.1f}s"
    )

    return {
        "task_id": task_id,
        "steps": step_count,
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    """Run the baseline inference across all tasks."""
    logger.info("=" * 60)
    logger.info("Customer Support Environment — Baseline Inference")
    logger.info("=" * 60)
    logger.info(f"API_BASE_URL:  {API_BASE_URL}")
    logger.info(f"MODEL_NAME:    {MODEL_NAME}")
    logger.info(f"ENV_BASE_URL:  {ENV_BASE_URL}")
    logger.info(f"API Key set:   {'Yes' if _api_key else 'No'}")
    logger.info("=" * 60)

    env_client = EnvClient(base_url=ENV_BASE_URL)

    # Wait for environment to be ready
    logger.info("[START] Waiting for environment server...")
    for attempt in range(30):
        if env_client.health():
            logger.info("[START] Environment server is ready!")
            break
        time.sleep(2)
    else:
        logger.error("[ERROR] Environment server not available after 60 seconds.")
        sys.exit(1)

    # Task order: easy → medium → hard
    task_ids = ["easy_faq", "medium_refund", "hard_escalation"]
    results = []

    for task_id in task_ids:
        logger.info("")
        logger.info("-" * 40)
        try:
            result = run_task(env_client, task_id)
            results.append(result)
        except Exception as e:
            logger.error(f"[ERROR] Task {task_id} failed: {e}")
            results.append({
                "task_id": task_id,
                "steps": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "elapsed": 0.0,
                "error": str(e),
            })

    # Compute final score
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

    total_avg = 0.0
    for r in results:
        status = "✓" if r.get("avg_reward", 0) > 0 else "✗"
        logger.info(
            f"  {status} {r['task_id']:20s} | "
            f"avg_reward={r.get('avg_reward', 0):.4f} | "
            f"steps={r.get('steps', 0)} | "
            f"time={r.get('elapsed', 0):.1f}s"
        )
        total_avg += r.get("avg_reward", 0)

    final_score = total_avg / len(results) if results else 0.0
    logger.info("-" * 60)
    logger.info(f"  FINAL SCORE: {final_score:.4f} (0.0 – 1.0)")
    logger.info("=" * 60)

    # Save results to file
    output = {
        "final_score": final_score,
        "task_results": results,
        "config": {
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "env_base_url": ENV_BASE_URL,
        },
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to outputs/inference_results.json")

    return final_score


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score > 0 else 1)
