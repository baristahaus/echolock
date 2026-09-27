import json
import sys
import tempfile
import unittest
from pathlib import Path

from report.markdown import render
from run.plans import Plan, Step
from run.runner import Run, RunOptions

FAKE = [sys.executable, str(Path(__file__).with_name("fake_probe.py"))]


def run_plan(root: Path, steps, label="idle"):
    opts = RunOptions(out_root=root, label=label, sample_interval_s=0.2, pre_roll_s=0.3,
                      post_roll_s=0.2, capture_env=False)
    return Run(Plan("fake", "test", steps), opts, run_id=f"run-{label}")


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_step_statuses(self):
        run = run_plan(self.root, [
            Step("ok", FAKE + ["ok"]),
            Step("wrong", FAKE + ["wrong"]),
            Step("crash", FAKE + ["crash"]),
            Step("hang", FAKE + ["hang"], timeout_s=1.0),
        ])
        m = run.execute()
        self.assertEqual([s["status"] for s in m["steps"]], ["ok", "verify_failed", "error", "timeout"])
        self.assertEqual(m["status"], "error")
        self.assertTrue(m["steps"][3]["killed"])
        self.assertEqual(m["steps"][1]["failed_checks"], 1)
        self.assertGreaterEqual(m["samples"], 1)

        records = [json.loads(l) for l in (run.dir / "probe.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["run_id"] == "run-idle" for r in records))
        self.assertTrue(all({"v", "t_ns", "src", "type"} <= r.keys() for r in records))
        self.assertIn("stdout_text", {r["type"] for r in records})
        steps = {r["step"] for r in records if r["src"] == "probe"}
        self.assertEqual(steps, {"ok", "wrong", "crash", "hang"})
        self.assertIn("[ok] diagnostic", (run.dir / "stderr.log").read_text())
        # Samples bracket the steps on the same clock.
        samples = [json.loads(l) for l in (run.dir / "samples.jsonl").read_text().splitlines()]
        self.assertLess(samples[0]["t_ns"], records[0]["t_ns"])

    def test_required_step_aborts(self):
        run = run_plan(self.root, [Step("prep", FAKE + ["crash"], required=True),
                                   Step("after", FAKE + ["ok"])])
        m = run.execute()
        self.assertEqual(len(m["steps"]), 1)
        self.assertEqual(m["status"], "error")

    def test_report_side_by_side(self):
        a = run_plan(self.root, [Step("s", FAKE + ["ok"])], label="idle")
        b = run_plan(self.root, [Step("s", FAKE + ["wrong"])], label="load50")
        a.execute()
        b.execute()
        text = render([a.dir, b.dir])
        self.assertIn("| Step | Metric | Tags | idle | load50 |", text)
        row = next(l for l in text.splitlines() if l.startswith("| s | bw |"))
        self.assertIn("101.0 GB/s [100.0–102.0]", row)
        self.assertTrue(row.rstrip(" |").endswith("✗"), row)  # load50's value failed its oracle
        self.assertIn("silent fallback", text)
        self.assertIn("FAILED `s` bw.xor", text)


if __name__ == "__main__":
    unittest.main()
