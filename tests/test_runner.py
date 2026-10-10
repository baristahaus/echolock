import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from report.markdown import render
from run.plans import Plan, PlanOptions, Step, make_plan
from run.runner import Run, RunOptions

FAKE = [sys.executable, str(Path(__file__).with_name("fake_probe.py"))]
ONE_STEP = [sys.executable, str(Path(__file__).with_name("one_step.py"))]


def child_pids(pid: int) -> list[int]:
    """Direct children of `pid`, zombies excluded."""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            status = (p / "status").read_text()
        except OSError:
            continue
        for line in status.splitlines():
            if line.startswith("PPid:"):
                if int(line.split()[1]) == pid and process_state(int(p.name)) != "Z":
                    out.append(int(p.name))
                break
    return out


def process_state(pid: int) -> str | None:
    """The scheduler state letter, or None when the pid is gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def wait_for(predicate, timeout: float = 20.0, every: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(every)
    return False


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
        a = run_plan(self.root, [Step("s", FAKE + ["skips"])], label="idle")
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
        self.assertIn("## Notes", text)                    # what the probe chose not to measure
        self.assertIn("one section of the probe did not run", text)
        # Declared values sit in one column per run, so a difference between runs is visible.
        declared_row = next(l for l in text.splitlines() if l.startswith("| device.sm_count |"))
        self.assertIn("| fake | 48 | 12 |", declared_row)
        # Each step's own environment window is summarized below the whole-run table.
        self.assertIn("Environment inside each step", text)


class PlansTest(unittest.TestCase):
    """--drop-caches must never run before a step that manages page residency itself: dropping
    behind file_page's resident/evict phases would evict the pages they are about to measure."""

    def test_file_page_eligibility(self):
        plan = make_plan("file_page", "cuda", Path("/tmp/x"), PlanOptions())
        by_name = {s.name: s.drop_caches for s in plan.steps}
        for cold in ("prep", "warm", "cold-serial", "cold-rows"):
            self.assertTrue(by_name[cold], cold)
        for resident in ("res-seq", "res-rows", "evict", "res-rows-again"):
            self.assertFalse(by_name[resident], resident)

    def test_memory_and_sync_eligible(self):
        for probe in ("memory", "sync"):
            plan = make_plan(probe, "cuda", Path("/tmp/x"), PlanOptions())
            self.assertTrue(all(s.drop_caches for s in plan.steps), probe)

    def test_unknown_probe_or_backend(self):
        with self.assertRaises(ValueError):
            make_plan("nope", "cuda", Path("/tmp/x"), PlanOptions())
        with self.assertRaises(ValueError):
            make_plan("memory", "amd", Path("/tmp/x"), PlanOptions())

    def test_new_probes_wire_up(self):
        """clock/mma take one step each; pointer_chase runs un-dropped (it manages residency)
        and cleans its scratch file up like file_page does."""
        clock = make_plan("clock", "cuda", Path("/tmp/x"), PlanOptions())
        self.assertEqual([s.name for s in clock.steps], ["clock"])
        self.assertTrue(clock.steps[0].drop_caches)
        mma = make_plan("mma", "cuda", Path("/tmp/x"), PlanOptions())
        self.assertEqual([s.name for s in mma.steps], ["mma"])
        self.assertTrue(mma.steps[0].drop_caches)
        pc = make_plan("pointer_chase", "cuda", Path("/tmp/x"), PlanOptions())
        self.assertEqual([s.name for s in pc.steps], ["pointer_chase"])
        self.assertFalse(pc.steps[0].drop_caches)
        self.assertEqual([p.name for p in pc.cleanup], ["pointer_chase.scratch"])
        self.assertIn("--file", pc.steps[0].argv)
        # The scratch dir option moves the file.
        pc2 = make_plan("pointer_chase", "cuda", Path("/tmp/x"),
                        PlanOptions(scratch_dir=Path("/scratch")))
        self.assertIn("/scratch/pointer_chase.scratch", pc2.steps[0].argv)


class DropCachesGatingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_steps(self, steps, drop_caches):
        opts = RunOptions(out_root=self.root, label="idle", sample_interval_s=0.2,
                          pre_roll_s=0.1, post_roll_s=0.1, capture_env=False,
                          drop_caches=drop_caches)
        return Run(Plan("fake", "test", steps), opts, run_id="run-dc")

    def test_only_eligible_steps_drop(self):
        run = self.run_steps([Step("drop", FAKE + ["ok"], drop_caches=True),
                              Step("keep", FAKE + ["ok"], drop_caches=False)], True)
        with mock.patch("run.runner.drop_page_cache", return_value="ok") as drop:
            run.execute()
        self.assertEqual(drop.call_count, 1)
        starts = {json.loads(l)["step"]: json.loads(l)["drop_caches"]
                  for l in (run.dir / "probe.jsonl").read_text().splitlines()
                  if json.loads(l)["type"] == "step_start"}
        self.assertEqual(starts, {"drop": "ok", "keep": None})

    def test_flag_off_means_no_calls(self):
        run = self.run_steps([Step("drop", FAKE + ["ok"], drop_caches=True)], False)
        with mock.patch("run.runner.drop_page_cache", return_value="ok") as drop:
            run.execute()
        self.assertEqual(drop.call_count, 0)


class InterruptTest(unittest.TestCase):
    """Ctrl-C or `kill` must leave a readable run directory and must not leave the probe running:
    a step is started in its own session, so nothing else would keep it off the GPU."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def interrupt(self, sig: signal.Signals) -> dict:
        p = subprocess.Popen(ONE_STEP + [str(self.root), "hang"], stdout=subprocess.PIPE,
                             text=True, start_new_session=True)
        try:
            run_dir = Path(p.stdout.readline().strip())
            self.assertTrue(wait_for(lambda: child_pids(p.pid)),
                            "the step's process never started")
            probe = child_pids(p.pid)[0]
            time.sleep(0.3)                       # let it get as far as writing a record
            os.kill(p.pid, sig)
            self.assertEqual(p.wait(timeout=30),
                             130 if sig == signal.SIGINT else 143)   # 128 + signal
            self.assertTrue(wait_for(lambda: (run_dir / "manifest.json").is_file()),
                            f"no manifest in {run_dir}")
            self.assertTrue(wait_for(lambda: process_state(probe) is None),
                            f"probe pid {probe} outlived the run")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            manifest["_dir"] = str(run_dir)
            return manifest
        finally:
            for pid in child_pids(p.pid):
                os.kill(pid, signal.SIGKILL)
            p.kill()
            p.wait(timeout=10)
            p.stdout.close()

    def test_ctrl_c(self):
        m = self.interrupt(signal.SIGINT)
        self.assertEqual(m["status"], "error")
        self.assertEqual([s["name"] for s in m["steps"]], ["hang"])
        self.assertTrue(m["steps"][0]["interrupted"], m["steps"][0])
        self.assertIn("interrupted", m["status_note"])
        self.assertIn("KeyboardInterrupt", m["status_note"])
        self.assertIn("0/1", render([Path(m["_dir"])]))

    def test_sigterm(self):
        m = self.interrupt(signal.SIGTERM)
        self.assertEqual(m["status"], "error")
        self.assertIn("SIGTERM", m["status_note"])


class BrokenRunDirTest(unittest.TestCase):
    """A run killed before it wrote its manifest must still report, not crash the report."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        good = run_plan(self.root, [Step("s", FAKE + ["ok"])], label="idle")
        good.execute()
        self.good = good.dir
        (self.root / "killed").mkdir()                                  # Ctrl-C before the manifest
        (self.root / "corrupt").mkdir()
        (self.root / "corrupt" / "manifest.json").write_text("{'status': 'ok'")
        (self.root / "list").mkdir()
        (self.root / "list" / "manifest.json").write_text("[1, 2]")

    def tearDown(self):
        self.tmp.cleanup()

    def test_render_survives(self):
        text = render([self.good, self.root / "killed", self.root / "corrupt",
                       self.root / "list", self.root / "absent"])
        self.assertIn("unreadable", text)
        self.assertIn("killed", text)
        self.assertIn("## Measured", text)                              # the good run still reports
        self.assertIn("No run directories", render([]))


if __name__ == "__main__":
    unittest.main()
