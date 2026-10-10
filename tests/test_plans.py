"""Plan construction, focused on --pin-cpus: it is honoured by exactly one probe/backend pair, and
the CLI guard is allowed to promise that. A flag that is accepted and silently dropped is the same
class of bug as a fast wrong answer, so the plan and the guard are checked together."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run.plans import PlanOptions, honours_pin_cpus, make_plan  # noqa: E402


def plan_for(probe, backend, **kw):
    opts = PlanOptions(**kw)
    return make_plan(probe, backend, Path(tempfile.mkdtemp()), opts)


class TestPinCpus(unittest.TestCase):
    def test_sync_cuda_actually_pins(self):
        p = plan_for("sync", "cuda", pin_cpus=[0, 10])
        pinned = [s for s in p.steps if "taskset" in s.argv]
        self.assertEqual([s.name for s in pinned], ["schedule=blocking,cpu=0",
                                                   "schedule=blocking,cpu=10"])
        self.assertEqual(pinned[0].argv[:3], ["taskset", "-c", "0"])

    def test_no_other_step_is_pinned(self):
        p = plan_for("sync", "cuda", pin_cpus=[0])
        self.assertEqual(len([s for s in p.steps if "taskset" in s.argv]), 1)
        self.assertEqual(len(p.steps), 5)  # four schedules + one pinned

    def test_guard_agrees_with_what_the_plan_builds(self):
        for probe in ("memory", "file_page", "clock", "pointer_chase", "mma"):
            self.assertFalse(honours_pin_cpus(probe, "cuda"), probe)
            self.assertFalse(any("taskset" in s.argv
                                 for s in plan_for(probe, "cuda", pin_cpus=[0]).steps), probe)
        # Vulkan sync has no SCHEDULE axis at all, so it pins nothing either.
        self.assertFalse(honours_pin_cpus("sync", "vulkan"))
        self.assertFalse(any("taskset" in s.argv
                             for s in plan_for("sync", "vulkan", pin_cpus=[0]).steps))
        self.assertTrue(honours_pin_cpus("sync", "cuda"))


if __name__ == "__main__":
    unittest.main()
