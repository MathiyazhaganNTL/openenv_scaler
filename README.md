---
title: Customer Support Env
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# 🎧 AI-Powered Customer Support Ticket Resolution Environment

> **An OpenEnv-compatible environment for training AI agents to handle real-world customer support scenarios — from simple FAQs to complex, multi-step escalations with angry customers.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 1. Environment Overview

This environment simulates a **real customer support helpdesk** where an AI agent must:

- Read incoming customer tickets with varying complexity
- Understand customer sentiment (neutral → frustrated → angry)
- Apply company policies (refund, shipping, escalation)
- Craft professional, empathetic, and accurate responses
- Resolve issues within a limited number of steps

The agent interacts using the standard **OpenEnv API**: `reset()`, `step()`, and `state()`.

---

## 2. Real-World Use Case

Customer support is one of the most common AI deployment targets. This environment captures realistic challenges:

| Challenge | How It's Simulated |
|---|---|
| **Tone matching** | Grader evaluates empathy, professionalism, and harmful language |
| **Policy reasoning** | Agent must apply correct refund/shipping/escalation policies |
| **Multi-turn dialogue** | Customers send follow-up messages that depend on agent's response quality |
| **Escalation handling** | Hard tasks require knowing when and how to escalate |
| **Angry customers** | Sentiment ranges from neutral to furious, requiring different strategies |

---

## 3. Action Space

The agent sends a `SupportAction` with:

```python
class SupportAction(BaseModel):
    response_text: str    # Agent's response to the customer (1-2000 chars)
    action_type: str      # "respond" | "escalate" | "resolve" | "request_info"
    internal_notes: str   # Optional internal notes (not visible to customer)
```

| Action Type | Effect |
|---|---|
| `respond` | Continue the conversation |
| `resolve` | Mark ticket as resolved (ends episode) |
| `escalate` | Escalate to senior support |
| `request_info` | Ask customer for more information |

---

## 4. Observation Space

After each step, the agent receives a `SupportObservation`:

```python
class SupportObservation(BaseModel):
    ticket: TicketInfo              # Ticket metadata (ID, category, priority, customer info)
    conversation_history: list      # Full message history
    current_message: str            # Latest customer message to respond to
    policy_context: str             # Relevant company policies
    task_id: str                    # Current task identifier
    difficulty: str                 # "easy" | "medium" | "hard"
    max_steps: int                  # Maximum steps allowed
    steps_remaining: int            # Steps left before timeout
    done: bool                      # Whether episode is complete
    reward: float                   # Cumulative reward so far
```

---

## 5. Reward Design

The reward function uses a **dense, multi-axis scoring system**:

### Scoring Axes

| Axis | Weight (varies by task) | What It Measures |
|---|---|---|
| **Correctness** | 0.30-0.35 | Keyword/concept matching against expected response elements |
| **Tone** | 0.30-0.40 | Professional, empathetic language vs. harmful/rude signals |
| **Completeness** | 0.30-0.40 | Checklist of required response components |

### Reward Breakdown Example

```
+0.30 → Correctly identifies the issue (correctness)
+0.30 → Professional and empathetic tone (tone)
+0.40 → Addresses all required elements (completeness)
─────
 1.00 → Perfect score
```

### Penalties (deducted from total)

| Penalty | Deduction | Trigger |
|---|---|---|
| Empty response | -0.30 | < 5 words |
| Repeated response | -0.15 to -0.30 | Copy-paste from previous |
| Harmful language | -0.50 | Offensive or inappropriate content |
| Irrelevant content | -0.40 | Off-topic responses |

---

## 6. Task Descriptions

### Task 1: Simple FAQ (Easy)
- **Ticket:** "Where is my order?"
- **Customer:** Sarah Johnson (Neutral sentiment)
- **Expected:** Reference order ID, explain shipping timeframe (5-7 business days), mention tracking email
- **Max Steps:** 3
- **Policy Context:** Shipping policy

### Task 2: Conditional Refund (Medium)
- **Ticket:** "Refund for opened laptop bag with defective stitching"
- **Customer:** Michael Chen (Frustrated sentiment)
- **Expected:** Identify as manufacturing defect, offer full refund + replacement option, explain return process
- **Max Steps:** 5
- **Policy Context:** Refund policy + Return policy
- **Follow-ups:** Customer provides photos, asks about timeline

### Task 3: Complex Complaint Escalation (Hard)
- **Ticket:** "Wrong item, late delivery, rude staff"
- **Customer:** David Martinez (Angry sentiment)
- **Expected:** Address ALL three issues, offer refund + compensation, escalate to manager, provide written confirmation
- **Max Steps:** 7
- **Policy Context:** All policies (refund, return, shipping, escalation)
- **Follow-ups:** Threats to file complaints, demands for specifics, requests for written confirmation

---

## 7. Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd openenv

# Install dependencies
pip install -r requirements.txt

# Run validation
python validate.py
```

### Environment Variables (for inference)

```bash
cp .env.example .env
# Edit .env with your API keys
```

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-3.5-turbo` | Model to use |
| `OPENAI_API_KEY` | — | API key |
| `HF_TOKEN` | — | Alternative: HF token |
| `ENV_BASE_URL` | `http://localhost:8000` | Environment server URL |

---

## 8. Run Instructions

### Start the Environment Server

```bash
# Direct
python -m server.app

# Or with uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t customer-support-env .
docker run -p 8000:8000 customer-support-env
```

### Run Baseline Inference

```bash
# Start the server first (in another terminal)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference
python inference.py
```

### API Usage Examples

```bash
# Health check
curl http://localhost:8000/health

# List tasks
curl http://localhost:8000/tasks

# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_faq"}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"response_text": "Thank you for reaching out!", "action_type": "respond"}}'

# Get state
curl http://localhost:8000/state
```

### Python Client Usage

```python
from server.environment import CustomerSupportEnvironment
from models import SupportAction

env = CustomerSupportEnvironment()

# Reset to a task
obs = env.reset(task_id="easy_faq")
print(obs.current_message)  # Customer's first message

# Respond
action = SupportAction(
    response_text="Hi Sarah! Your order ORD-55821 ships in 5-7 business days...",
    action_type="respond",
)
obs, reward, done, info = env.step(action)
print(f"Reward: {reward:.4f}")
print(f"Score breakdown: {info['reward_breakdown']}")
```

---

## 9. Baseline Results

Running the baseline inference with `gpt-3.5-turbo`:

| Task | Difficulty | Avg Reward | Steps |
|---|---|---|---|
| `easy_faq` | Easy | ~0.65 | 1–2 |
| `medium_refund` | Medium | ~0.55 | 3–4 |
| `hard_escalation` | Hard | ~0.45 | 4–6 |
| **Overall** | — | **~0.55** | — |

> Scores vary based on model quality. Better models achieve higher scores by producing more empathetic, accurate, and complete responses.

---

## Project Structure

```
openenv/
├── openenv.yaml           # OpenEnv manifest (metadata, tasks, config)
├── models.py              # Pydantic models (Action, Observation, State, Reward)
├── tasks.py               # Task definitions (3 tasks, rubrics, policies)
├── grader.py              # Deterministic grading engine
├── inference.py           # Baseline LLM inference script
├── validate.py            # Environment validation script
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── Dockerfile             # Docker container definition
├── .dockerignore          # Docker build exclusions
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
├── README.md              # This file
└── server/
    ├── __init__.py
    ├── environment.py     # Core environment (reset/step/state)
    └── app.py             # FastAPI HTTP server
```

---

## HuggingFace Spaces Deployment

This environment is designed for deployment as a **Docker-based HuggingFace Space**:

1. Create a new Space with **Docker SDK**
2. Push the code to the Space repository
3. The Space will auto-build and expose the API at port 8000
4. Tag the Space with `openenv`

```bash
# Using openenv CLI
openenv push --repo-id your-username/customer-support-env
```

The API endpoint `POST /reset` will respond with HTTP 200, confirming the Space is operational.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
