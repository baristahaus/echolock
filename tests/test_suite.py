"""The suite runner on a stand-in build tree: cells are ok / not-built as designed, the run
directories are collected, and the combined report renders a probe x backend matrix."""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from run.suite import render_suite_report, suite_matrix  # noqa: E402


def make_fake_build(root: Path, probes=("memory", "mma")) -> Path:
    """A build tree of stand-in probes: shell wrappers around tests/fake_probe.py."""
    build = root / "build"
    for probe in probes:
        d = build / "vulkan" / probe
        d.parent.mkdir(parents=True, exist_ok=True)
        exe = d
        exe.write_text(f"#!/bin/sh\nexec {sys.executable!r} {ROOT / 'tests' / 'fake_probe.py'} ok\n")
        os.chmod(exe, 0o755)
    return build


class SuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.build = make_fake_build(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_matrix_cells_and_report(self):
        runs = self.root / "runs"
        cells = suite_matrix(probes=("memory", "mma", "sync"),
                             backends=("vulkan", "cuda"),
                             conditions=("idle",),
                             out_root=runs,
                             build_root=self.build)
        # built vulkan probes: real run directories, ok status
        for probe in ("memory", "mma"):
            cell = cells[(probe, "vulkan", "idle")]
            self.assertEqual(cell["state"], "ok", cell)
            run_dir = Path(cell["run_dir"])
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertEqual(json.loads((run_dir / "manifest.json").read_text())["status"], "ok")
            # the probe's records landed in the run directory
            self.assertTrue((run_dir / "probe.jsonl").is_file())
        # the sync binary does not exist: a state, not a failure
        self.assertEqual(cells[("sync", "vulkan", "idle")]["state"], "not built")
        # no cuda backend in the build tree at all
        self.assertEqual(cells[("memory", "cuda", "idle")]["state"], "not built")

        text = render_suite_report(cells, ("idle",), ("memory", "mma", "sync"),
                                   ("vulkan", "cuda"))
        self.assertIn("## condition: idle", text)
        self.assertIn("not built", text)
        self.assertIn("## run directories", text)
        self.assertIn(cells[("memory", "vulkan", "idle")]["run_dir"], text)
        self.assertIn("| sync | not built | not built |", text)

    def test_unrunnable_probe_is_a_clean_cell(self):
        """A binary that exists but cannot run here (a CUDA probe without an NVIDIA GPU) must
        be a manifest-status cell, not a crash of the suite."""
        build = self.root / "build2"
        d = build / "vulkan" / "memory"
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text("#!/bin/sh\necho 'no device' >&2\nexit 1\n")
        os.chmod(d, 0o755)
        runs = self.root / "runs2"
        cells = suite_matrix(probes=("memory",), backends=("vulkan",),
                             conditions=("idle",), out_root=runs, build_root=build)
        cell = cells[("memory", "vulkan", "idle")]
        self.assertEqual(cell["state"], "error", cell)
        self.assertTrue(Path(cell["run_dir"]).is_dir())


if __name__ == "__main__":
    unittest.main()
