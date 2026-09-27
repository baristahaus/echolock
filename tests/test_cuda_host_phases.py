"""Runs the GPU-free phases of the built CUDA file_page probe and checks its JSONL contract.

Skipped when build/cuda/file_page does not exist (no nvcc). Needs no GPU.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

EXE = Path(__file__).resolve().parents[1] / "build" / "cuda" / "file_page"


@unittest.skipUnless(EXE.exists(), "build/cuda/file_page not built")
class FilePageHostPhases(unittest.TestCase):
    def run_phase(self, phase, scratch):
        p = subprocess.run([str(EXE), "--phase", phase, "--file", str(scratch), "--bytes", "8M"],
                           capture_output=True, text=True, timeout=120)
        records = [json.loads(l) for l in p.stdout.splitlines()]
        return p.returncode, records

    def test_prep_warm_evict(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            scratch = Path(d) / "scratch.bin"
            for phase in ("prep", "warm", "evict"):
                rc, records = self.run_phase(phase, scratch)
                self.assertEqual(rc, 0, records)
                self.assertEqual(records[0]["type"], "probe_start")
                self.assertEqual(records[-1]["type"], "probe_end")
                self.assertEqual(records[-1]["status"], "ok")
                for r in records:
                    self.assertEqual((r["v"], r["src"], r["probe"]), (1, "probe", "file_page"))
                    self.assertIsInstance(r["t_ns"], int)
                checks = [r for r in records if r["type"] == "check"]
                self.assertTrue(all(c["pass"] for c in checks))
                if phase == "warm":
                    self.assertEqual([c["name"] for c in checks], ["warm.host_xor"])
            self.assertEqual(scratch.stat().st_size, 8 << 20)


if __name__ == "__main__":
    unittest.main()
