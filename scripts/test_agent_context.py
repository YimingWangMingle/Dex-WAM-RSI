"""CPU checks for Agent context compression; the summarizer is a test double."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.context import (
    SUMMARY_PREFIX, AgentContext, ContextBudgetError, ContextPolicy, TaskState,
)


class Summarizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, max_chars):
        self.prompts.append(prompt)
        return f"summary#{len(self.prompts)}: picked cup, gripper closed"


def fill(ctx, turns, size=400):
    for i in range(turns):
        ctx.begin_turn()
        ctx.add("observation", "measured", f"obs {i}", images=(f"img-{i}",))
        ctx.add("tool", "execution", f"turn {i} " + "x" * size)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.state = TaskState("put the cup on the tray")
        self.summarizer = Summarizer()
        self.events = []
        self.policy = ContextPolicy(budget_bytes=6000, summary_trigger_ratio=0.5, keep_turns=3,
                                    image_turns=1, image_reserve_bytes=100)
        self.ctx = AgentContext(self.state, self.policy, summarizer=self.summarizer,
                                on_event=self.events.append)
        self.ctx.pin("experience: stack_bowls r1")

    def test_small_history_is_verbatim(self):
        fill(self.ctx, 2, size=50)
        messages = self.ctx.messages()
        self.assertEqual(self.summarizer.prompts, [])
        self.assertEqual(sum(m["kind"] == "execution" for m in messages), 2)

    def test_old_turns_summarized_recent_kept_whole(self):
        fill(self.ctx, 12)
        self.state.fix_goal("place_xyz", [0.4, 0.1, 0.2])
        messages = self.ctx.messages()
        kinds = [m["kind"] for m in messages]
        self.assertEqual(kinds[:3], ["pinned", "task_state", "summary"])
        self.assertTrue(messages[2]["text"].startswith(SUMMARY_PREFIX))
        kept = [m["text"] for m in messages if m["kind"] == "execution"]
        self.assertTrue(kept and kept[-1].startswith("turn 11"))
        self.assertTrue(all(m["text"].split()[1].isdigit() for m in messages if m["kind"] == "execution"))
        self.assertEqual(len([m for m in messages if m["kind"] == "measured"]),
                         len(kept))   # turns are never split
        self.assertIn("place_xyz", messages[1]["text"])
        with_images = [m for m in messages if isinstance(m.get("images"), list)]
        self.assertEqual(len(with_images), 1)   # only the newest turn keeps images
        self.assertNotIn("img-0", json.dumps(messages))
        self.assertLess(self.ctx.estimate(messages), self.policy.budget_bytes)
        self.assertEqual(self.events[-1]["event"], "context_summary")
        fill(self.ctx, 12)
        self.ctx.messages()
        self.assertIn("summary#1", self.summarizer.prompts[-1])   # summaries are merged

    def test_no_summarizer_fails_loudly(self):
        ctx = AgentContext(self.state, self.policy)
        fill(ctx, 12)
        with self.assertRaises(ContextBudgetError):
            ctx.messages()

    def test_latest_turn_over_budget_is_an_error(self):
        fill(self.ctx, 1, size=8000)
        with self.assertRaises(ContextBudgetError):
            self.ctx.messages()
        self.assertEqual(self.events[-1]["event"], "context_budget_exceeded")

    def test_fixed_goal_is_not_silently_changed(self):
        self.state.fix_goal("lift_dz", 0.05)
        self.state.fix_goal("lift_dz", 0.05)
        with self.assertRaises(ValueError):
            self.state.fix_goal("lift_dz", 0.10)
        self.state.clear_goal("lift_dz")
        self.state.fix_goal("lift_dz", 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
