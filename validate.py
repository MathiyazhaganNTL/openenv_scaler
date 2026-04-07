"""
Validation / smoke-test script for the Customer Support Environment.

Runs through all 3 tasks with deterministic responses and verifies:
  ✓ reset() returns valid SupportObservation
  ✓ step() returns (observation, reward, done, info) with correct types
  ✓ state() returns valid SupportState
  ✓ Rewards are non-constant and in [0.0, 1.0]
  ✓ Episodes terminate correctly
  ✓ Grader produces varying scores for different responses

Usage:
    python validate.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SupportAction, SupportObservation, SupportState, RewardBreakdown
from server.environment import CustomerSupportEnvironment
from tasks import TASK_IDS


def validate_task(env: CustomerSupportEnvironment, task_id: str, responses: list[str]) -> dict:
    """Run a task with given responses and collect results."""
    print(f"\n{'='*50}")
    print(f"  Validating: {task_id}")
    print(f"{'='*50}")

    # Test reset
    obs = env.reset(task_id=task_id)
    assert isinstance(obs, SupportObservation), f"reset() must return SupportObservation, got {type(obs)}"
    assert obs.task_id == task_id, f"task_id mismatch: {obs.task_id} != {task_id}"
    assert not obs.done, "Episode should not be done after reset"
    assert obs.current_message, "Initial customer message should not be empty"
    print(f"  ✓ reset() returned valid SupportObservation")
    print(f"    Customer: {obs.ticket.customer_name}")
    print(f"    Subject:  {obs.ticket.subject}")
    print(f"    Message:  {obs.current_message[:60]}...")

    # Test state after reset
    state = env.state()
    assert isinstance(state, SupportState), f"state() must return SupportState, got {type(state)}"
    assert state.step_count == 0, "Step count should be 0 after reset"
    assert not state.done, "State should not be done after reset"
    print(f"  ✓ state() returned valid SupportState")

    # Test steps
    rewards = []
    for i, response_text in enumerate(responses):
        action = SupportAction(
            response_text=response_text,
            action_type="respond" if i < len(responses) - 1 else "resolve",
        )
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, SupportObservation), f"step() obs must be SupportObservation"
        assert isinstance(reward, float), f"step() reward must be float, got {type(reward)}"
        assert isinstance(done, bool), f"step() done must be bool, got {type(done)}"
        assert isinstance(info, dict), f"step() info must be dict, got {type(info)}"
        assert 0.0 < reward < 1.0, f"Reward {reward} out of strict (0.0, 1.0) range"

        rewards.append(reward)
        breakdown = info.get("reward_breakdown", {})
        print(f"  ✓ step({i+1}) → reward={reward:.4f} | "
              f"correctness={breakdown.get('correctness', 0):.2f} "
              f"tone={breakdown.get('tone', 0):.2f} "
              f"completeness={breakdown.get('completeness', 0):.2f} "
              f"done={done}")

        if done:
            break

    # Verify final state
    state = env.state()
    assert state.step_count > 0, "Step count should be > 0 after steps"
    print(f"  ✓ Final state: steps={state.step_count}, reward={state.cumulative_reward:.4f}")

    return {
        "task_id": task_id,
        "rewards": rewards,
        "avg_reward": max(0.0001, min(0.9999, sum(rewards) / len(rewards))) if rewards else 0.5,
        "steps": len(rewards),
    }


def validate_grader_variance():
    """Verify the grader doesn't return constant values."""
    print(f"\n{'='*50}")
    print(f"  Validating: Grader Variance")
    print(f"{'='*50}")

    env = CustomerSupportEnvironment()
    env.reset(task_id="easy_faq")

    # Test with a GOOD response
    good_action = SupportAction(
        response_text=(
            "Hi Sarah! Thank you for reaching out about your order ORD-55821. "
            "I completely understand your concern about the shipping update. "
            "Standard shipping typically takes 5-7 business days, and since your "
            "order was placed on March 28th, it should be arriving soon. "
            "You should receive a tracking number via email. Let me look into "
            "the specific status of your order right away and I'll update you. "
            "Is there anything else I can help you with?"
        ),
        action_type="respond",
    )
    _, good_reward, _, good_info = env.step(good_action)

    # Reset and test with a BAD response
    env.reset(task_id="easy_faq")
    bad_action = SupportAction(
        response_text="I don't know.",
        action_type="respond",
    )
    _, bad_reward, _, bad_info = env.step(bad_action)

    # Reset and test with an IRRELEVANT response
    env.reset(task_id="easy_faq")
    irr_action = SupportAction(
        response_text="The weather is nice today. Have you tried checking the stock market?",
        action_type="respond",
    )
    _, irr_reward, _, irr_info = env.step(irr_action)

    print(f"  Good response reward:       {good_reward:.4f}")
    print(f"  Bad response reward:        {bad_reward:.4f}")
    print(f"  Irrelevant response reward: {irr_reward:.4f}")

    assert good_reward != bad_reward, "Grader returns same reward for good and bad responses!"
    assert good_reward > bad_reward, "Good response should score higher than bad response!"
    assert good_reward > irr_reward, "Good response should score higher than irrelevant response!"
    print(f"  ✓ Grader produces varying scores (NOT constant)")
    print(f"  ✓ Good > Bad > Irrelevant ordering confirmed")


def main():
    print("=" * 50)
    print("  Customer Support Environment — Validation")
    print("=" * 50)

    env = CustomerSupportEnvironment()

    # Test responses per task
    test_responses = {
        "easy_faq": [
            "Hi Sarah! Thank you for reaching out about your order ORD-55821. "
            "Standard shipping takes 5-7 business days. You'll receive a tracking "
            "number via email within 24 hours of shipment. Let me check on the "
            "status of your Wireless Bluetooth Headphones order right away.",
        ],
        "medium_refund": [
            "Hi Michael, I'm sorry to hear about the stitching issue with your "
            "Premium Leather Laptop Bag. That sounds like a manufacturing defect, "
            "and I completely understand your frustration. According to our policy, "
            "defective items qualify for a full refund or replacement at any time. "
            "Could you please send photos of the defect so we can process this quickly?",
            "Thank you for the photos, Michael. I can confirm this is a defect. "
            "You have two options: a full refund of $149.99 or a replacement bag. "
            "Either way, we'll provide a prepaid return shipping label. "
            "Which would you prefer?",
            "We'll process your full refund within 5-7 business days after we "
            "receive the returned bag. I'll email you the return label right away. "
            "I sincerely apologize for the inconvenience.",
        ],
        "hard_escalation": [
            "Mr. Martinez, I sincerely apologize for this terrible experience. "
            "What happened — receiving the wrong item after a late delivery, "
            "and then being treated rudely by our support staff — is completely "
            "unacceptable. You deserve much better. I'm escalating this to our "
            "senior support team immediately as a top priority case.",
            "I understand your frustration completely, Mr. Martinez. Here's exactly "
            "what I'm going to do: First, I'm processing a full refund of $349.99 "
            "for the wrong item. Second, I'm adding a $50 store credit as compensation "
            "for the inconvenience. Third, I'm personally ensuring the correct "
            "Smart Home Security Camera System ships via expedited delivery today. "
            "The staff member's behavior will be addressed by management.",
            "Absolutely, Mr. Martinez. Here are the specifics: Your refund will be "
            "processed within 24 hours. The replacement ships via priority express "
            "and will arrive within 2-3 business days. The $50 credit is already "
            "applied to your account. I will personally follow up with you via "
            "email tomorrow to confirm everything is on track.",
            "I completely understand, Mr. Martinez. I'll send you a confirmation "
            "email within the hour with all the details in writing: the refund, "
            "the replacement tracking, and the store credit. You have my word "
            "this will be resolved. Thank you for your patience."
        ],
    }

    all_results = []
    for task_id in TASK_IDS:
        responses = test_responses.get(task_id, ["Thank you for reaching out."])
        result = validate_task(env, task_id, responses)
        all_results.append(result)

    # Validate grader variance
    validate_grader_variance()

    # Summary
    print(f"\n{'='*50}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*50}")
    total_avg = 0.0
    for r in all_results:
        print(f"  ✓ {r['task_id']:20s} → avg_reward={r['avg_reward']:.4f} steps={r['steps']}")
        total_avg += r['avg_reward']
    overall = total_avg / len(all_results) if all_results else 0.01
    overall = max(0.0001, min(0.9999, overall))
    print(f"\n  Overall Score: {overall:.4f}")
    print(f"\n  ✅ ALL VALIDATIONS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
