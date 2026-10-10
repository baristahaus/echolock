"""Runs the host constants probe and requires its cross-source checks to pass.

Skipped on machines with no GPU sources at all (no NVIDIA driver, no amdgpu).
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "backends" / "host" / "constants.py"


def records() -> list[dict]:
    proc = subprocess.run(["python3", str(PROBE)], capture_output=True, text=True, timeout=120)
    out = []
    for line in proc.stdout.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out, proc.returncode


class TestConstants(unittest.TestCase):
    def test_sources_agree(self):
        recs, rc = records()
        ends = [r for r in recs if r.get("type") == "probe_end"]
        if not ends:
            self.skipTest("no GPU sources on this machine (probe errored before probe_end)")
        self.assertEqual(rc, 0, f"constants probe failed: rc={rc}")
        self.assertEqual(ends[-1].get("status"), "ok")
        failed = [r for r in recs if r.get("type") == "check" and not r.get("pass")]
        self.assertEqual(failed, [],
                         f"sources disagree: {[f.get('name') for f in failed]}")


if __name__ == "__main__":
    unittest.main()
