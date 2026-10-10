"""Runs the built CUDA probes for real and requires the oracles to stay green.

The Python-side tests cannot catch device-code regressions (the memory, mma and packing bugs of
2026-10 were all invisible to them), so this suite executes each probe binary with tiny
parameters and asserts the probe's own element-wise oracle reported ok. Skipped when the
binaries are not built, no CUDA GPU is visible, or the GPU cannot host a context (e.g. its
memory is fully committed to another process).

    python3 -m unittest tests.test_cuda_device -v
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "cuda"

# probe -> arguments small enough to pass on a GPU that shares its memory with other work
CASES = {
    "sync": ["--iterations", "1", "--durations-us", "50"],
    "clock": ["--seconds", "0.5", "--load-gib", "0.2"],
    "pointer_chase": ["--footprints", "0.0625,1", "--kinds", "device", "--reps", "2"],
    "memory": ["--gib", "0.2", "--seconds", "0.2"],
    "mma": ["--seconds", "0.1"],
}


def probe_end_status(exe: Path, args: list[str], timeout: float = 120) -> tuple[str | None, str]:
    """Runs one probe process and returns its final probe_end status (None if it never got there)."""
    proc = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=tempfile.gettempdir())
    status = None
    for line in proc.stdout.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "probe_end":
            status = rec.get("status")
    return status, (proc.stderr + proc.stdout)


class TestCudaDeviceProbes(unittest.TestCase):
    binaries = {name: BUILD / name for name in sorted(CASES)}

    def setUp(self):
        if not all(b.exists() for b in self.binaries.values()):
            self.skipTest(f"probes not built under {BUILD}; run ./echolock build first")
        # A context-creating call decides GPU presence; file_page's prep phase needs no context.
        try:
            status, out = probe_end_status(BUILD / "sync", CASES["sync"], timeout=60)
        except (subprocess.TimeoutExpired, OSError) as e:
            self.skipTest(f"probe execution failed: {e}")
        if status is None:
            if "cudaErrorMemoryAllocation" in out or "out of memory" in out:
                self.skipTest("GPU cannot host a context (memory committed elsewhere)")
            self.skipTest("no usable CUDA GPU (probe produced no probe_end record)")
        if status != "ok":
            self.fail(f"baseline sync run failed (status {status}); fix before running the suite")

    def test_probe_oracles(self):
        for name, args in CASES.items():
            with self.subTest(probe=name):
                status, out = probe_end_status(BUILD / name, args)
                if status is None and "out of memory" in out:
                    self.skipTest(f"{name}: GPU memory committed elsewhere")
                self.assertEqual(status, "ok",
                                 f"{name} failed its own oracle check:\n{out[-2000:]}")

    def test_file_page_phases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:  # not tmpfs: that is the point
            exe = BUILD / "file_page"
            for phase in ("prep", "warm", "res-seq", "res-rows", "cold-serial", "res-rows-again"):
                with self.subTest(phase=phase):
                    status, out = probe_end_status(
                        exe, ["--phase", phase, "--file", f"{scratch}/scratch",
                              "--bytes", "16M"])
                    self.assertEqual(status, "ok", f"file_page {phase}:\n{out[-2000:]}")


if __name__ == "__main__":
    unittest.main()
