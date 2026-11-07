"""load/soak.py: the record protocol — start fields, ready only after the minimum soak
(and only after a temperature plateau on a machine that reports temperatures), achieved
duty in the heartbeats, end on SIGTERM, and exit on its own when the parent dies.

Runs on any Linux box: the CPU heat source is plain Python threads, the GPU section is
skipped with evidence when there is no NVIDIA GPU, and a missing temperature source
disables the plateau gate rather than failing.
"""

import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOAK = ROOT / "load" / "soak.py"
PY = sys.executable


class SoakProcess:
    """soak.py as a subprocess: records drained as they come, so the test can react to
    load_ready instead of racing the whole run."""

    def __init__(self, args: list[str]):
        self.proc = subprocess.Popen([PY, str(SOAK), *args], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self.records: list[dict] = []
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                self.records.append(rec)

    def wait_for(self, type_: str, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if any(r.get("type") == type_ for r in self.records):
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        self._thread.join(timeout=5)


class SoakProtocolTest(unittest.TestCase):
    def test_protocol(self):
        soak_min = 0.1    # 6 s
        p = SoakProcess(["--soak-min", "0.1", "--plateau-window", "1", "--plateau-rise", "1",
                         "--temp-int", "0.3", "--heartbeat-int", "1"])
        try:
            self.assertTrue(p.wait_for("load_ready", 30),
                            (p.proc.communicate() if p.proc.poll() is not None else ""))
            recs = p.records
            start = recs[0]
            self.assertEqual(start["type"], "load_start")
            self.assertEqual(start["src"], "load")
            self.assertEqual(start["v"], 1)
            self.assertEqual(start["load"], "soak")
            self.assertEqual(start["intended"]["cpu_duty"], 1.0)
            self.assertEqual(start["intended"]["soak_min"], soak_min)
            self.assertIn("gpu_duty", start["intended"])
            if start["intended"]["gpu_duty"] == "skipped":
                # A skip must say why, not just that it happened.
                self.assertIsInstance(start["intended"]["gpu_note"], str)
                self.assertTrue(start["intended"]["gpu_note"])
            self.assertIsInstance(start["deadline_min"], float)
            self.assertIsInstance(start["args"], list)

            ready = next(r for r in recs if r["type"] == "load_ready")
            # Ready no earlier than the minimum soak.
            self.assertGreaterEqual(ready["t_ns"] - start["t_ns"], soak_min * 60 * 1e9 - 1e6)
            ach = ready["achieved"]
            self.assertIsInstance(ach["cpu_duty"], float)
            self.assertGreater(ach["cpu_duty"], 0.0)
            # gpu_duty is a measured duty, or the string "skipped" when this machine
            # has no NVIDIA GPU / built gpuload.
            self.assertTrue(ach["gpu_duty"] == "skipped"
                            or isinstance(ach["gpu_duty"], float))
            self.assertIn("temp_c", ach)

            hbs = [r for r in recs if r["type"] == "heartbeat"]
            self.assertGreaterEqual(len(hbs), 1)
            for hb in hbs:
                self.assertIn("cpu_duty", hb["achieved"])
                self.assertIn("gpu_duty", hb["achieved"])
                self.assertIn("temp_c", hb["achieved"])

            # SIGTERM ends the load cleanly with a final record.
            p.proc.terminate()
            self.assertTrue(p.wait_for("load_end", 10))
            self.assertEqual(p.records[-1]["type"], "load_end")
            self.assertIn("achieved", p.records[-1])
        finally:
            p.stop()

    def test_ready_waits_for_the_plateau_when_temperatures_are_reported(self):
        # The minimum soak is short; on a machine that reports temperatures, ready must
        # still wait out the plateau window. (With no temperature source there is nothing
        # to plateau, so the gate does not apply.)
        window = 2.0
        p = SoakProcess(["--soak-min", "0.02", "--plateau-window", str(window),
                         "--plateau-rise", "1", "--temp-int", "0.3", "--heartbeat-int", "1"])
        try:
            self.assertTrue(p.wait_for("load_ready", 45))
            start = p.records[0]
            ready = next(r for r in p.records if r["type"] == "load_ready")
            delta = (ready["t_ns"] - start["t_ns"]) / 1e9
            if ready["achieved"]["temp_c"] is not None:
                self.assertGreaterEqual(delta, window)
        finally:
            p.stop()

    def test_parent_death_exits_on_its_own(self):
        # A parent that dies before the soak starts: the load notices and exits itself,
        # with a final record and no signal from us.
        parent = subprocess.Popen([PY, "-c", "import time; time.sleep(0.2)"])
        parent.wait(timeout=10)
        p = SoakProcess(["--soak-min", "10", "--parent-pid", str(parent.pid)])
        try:
            self.assertTrue(p.wait_for("load_end", 20))
            p.proc.wait(timeout=10)
            self.assertEqual(p.proc.returncode, 0)
            self.assertEqual(p.records[-1]["type"], "load_end")
            self.assertNotIn("load_error", (r["type"] for r in p.records))
        finally:
            p.stop()

    def test_usage_rejects_bad_values(self):
        p = subprocess.run([PY, str(SOAK), "--soak-min", "-1"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
