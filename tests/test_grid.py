"""Grid runner and load generators: a load condition must be a measured occupancy, not an
intention, and a load must never outlive the runner that started it."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run.grid import GridOptions, LoadProcess, load_argv, run_condition  # noqa: E402
from run.plans import Plan, Step  # noqa: E402
from tests.test_runner import FAKE  # noqa: E402

PY = sys.executable


def grid_opts(root: Path, **kw) -> GridOptions:
    kw.setdefault("settle_s", 0.3)
    kw.setdefault("ready_timeout_s", 20.0)
    kw.setdefault("sample_interval_s", 0.2)
    kw.setdefault("pre_roll_s", 0.2)
    kw.setdefault("post_roll_s", 0.2)
    return GridOptions(out_root=root, **kw)


def load_records(run_dir: Path) -> list[dict]:
    p = run_dir / "load.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines()]


class LoadArgvTest(unittest.TestCase):
    def test_conditions(self):
        self.assertIsNone(load_argv("idle"))
        cpu = load_argv("cpu50")
        self.assertTrue(cpu[1].endswith("cpuload.py"))
        self.assertIn("50.0", cpu)
        mem = load_argv("mem80")
        self.assertTrue(mem[1].endswith("memfill.py"))
        self.assertIn("0.8", mem)
        io = load_argv("io500")
        self.assertTrue(io[1].endswith("ioload.py"))
        gpu = load_argv("gpu50")
        self.assertTrue(gpu[0].endswith("gpuload"))
        for bad in ("nope", "cpu", "cpuX"):
            with self.assertRaises(ValueError):
                load_argv(bad)


class GridRunTest(unittest.TestCase):
    """A full grid with fake probe steps: the load lifecycle, the records, and the report."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_idle_and_cpu_condition(self):
        opts = grid_opts(self.root, conditions=["idle", "cpu50"])
        dirs = []
        for cond in ("idle", "cpu50"):
            plan = Plan("fake", "test", [Step("s", FAKE + ["ok"])])
            dirs.append(run_condition(cond, plan, opts, run_id=f"grid-{cond}"))
        for d, cond in zip(dirs, ("idle", "cpu50")):
            m = json.loads((d / "manifest.json").read_text())
            self.assertEqual((m["status"], m["load"]["name"]), ("ok", cond))
        # idle: no load records. cpu50: start, ready, heartbeats, end, and an honest duty.
        self.assertEqual(load_records(dirs[0]), [])
        recs = load_records(dirs[1])
        self.assertEqual(recs[0]["type"], "load_start")
        self.assertIn("load_ready", [r["type"] for r in recs])
        end = [r for r in recs if r["type"] == "load_end"][0]
        self.assertTrue(0.3 < end["achieved"]["duty"] < 0.7,
                        f"measured duty {end['achieved']} vs intended 0.5")

    def test_load_not_ready_still_runs(self):
        """A load that dies before reporting ready must not block the probe; the run records
        what happened instead of silently pretending the condition was met."""
        opts = grid_opts(self.root)
        plan = Plan("fake", "test", [Step("s", FAKE + ["ok"])])
        dying = [PY, "-c", "import sys; print('not json'); sys.exit(0)"]
        with unittest.mock.patch("run.grid.load_argv", return_value=dying):
            d = run_condition("cpu50", plan, opts, run_id="grid-dead-load")
        m = json.loads((d / "manifest.json").read_text())
        self.assertEqual(m["status"], "ok")
        recs = load_records(d)
        self.assertTrue(any(r["type"] == "load_error" for r in recs), recs)

    def test_parent_death_watchdog(self):
        """The load watches the runner's pid; when the runner dies, the load exits by itself."""
        pid = os.fork()
        if pid == 0:
            subprocess.Popen([PY, str(ROOT / "load" / "cpuload.py"), "--duty", "100",
                              "--deadline-min", "5", "--parent-pid", str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os._exit(0)
        os.waitpid(pid, 0)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            alive = []
            for p in Path("/proc").iterdir():
                if not p.name.isdigit():
                    continue
                try:
                    cmd = (p / "cmdline").read_bytes().split(b"\0")
                except OSError:
                    continue
                if len(cmd) > 1 and cmd[1].decode(errors="replace").endswith("cpuload.py"):
                    alive.append(p.name)
            if not alive:
                self.assertTrue(True)  # the load noticed and exited
                return
            time.sleep(0.5)
        self.fail("load outlived its parent")


class LoadGeneratorsTest(unittest.TestCase):
    def test_memfill_measures_resident(self):
        p = subprocess.run([PY, str(ROOT / "load" / "memfill.py"), "--fraction", "0.05",
                            "--deadline-min", "0.05"], capture_output=True, text=True,
                           timeout=60)
        recs = [json.loads(l) for l in p.stdout.splitlines()]
        end = [r for r in recs if r["type"] == "load_end"][0]
        self.assertGreater(end["achieved"]["resident_bytes"], 1e9)  # ~5% of RAM

    def test_ioload_hits_rate(self):
        p = subprocess.run([PY, str(ROOT / "load" / "ioload.py"), "--mbps", "100", "--gib", "1",
                            "--file", str(Path(self.tmpdir()) / "ioload.scratch"),
                            "--deadline-min", "0.05"], capture_output=True, text=True,
                           timeout=60)
        recs = [json.loads(l) for l in p.stdout.splitlines()]
        end = [r for r in recs if r["type"] == "load_end"][0]
        self.assertTrue(70 < end["achieved"]["mbps"] < 130, end["achieved"])

    def test_cpuload_records_protocol(self):
        p = subprocess.run([PY, str(ROOT / "load" / "cpuload.py"), "--duty", "100",
                            "--deadline-min", "0.05"], capture_output=True, text=True,
                           timeout=60)
        recs = [json.loads(l) for l in p.stdout.splitlines()]
        types = [r["type"] for r in recs]
        self.assertEqual(types[0], "load_start")
        self.assertEqual(types[-1], "load_end")
        self.assertIn("load_ready", types)

    @staticmethod
    def tmpdir() -> str:
        import tempfile
        return tempfile.mkdtemp()


if __name__ == "__main__":
    unittest.main()
