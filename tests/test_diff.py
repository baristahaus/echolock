"""diff / regress / crosscheck --check on synthetic run directories: verdicts, exit codes,
baseline selection, and the mma ref_xor bit-exact contract (equal -> agrees, different ->
diverges, absent -> missing with the reason, never a pass)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from report import diff as diffmod  # noqa: E402


def make_run(root: Path, name: str, probe: str, backend: str, label: str,
             measures: list[dict], date: str, device: str = "AMD Test GPU",
             checks: list[dict] | None = None, status: str = "ok"):
    """A minimal but real-shaped run directory.

    measures: (name, value, unit, tags); checks: (name, pass, detail)."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "v": 1, "run_id": d.name, "probe": probe, "backend": backend, "label": label,
        "status": status, "started_utc": date,
        "steps": [{"name": probe, "status": status, "returncode": 0, "failed_checks": 0}],
        "kernel_log": {"available": False, "error": "/dev/kmsg: Operation not permitted",
                       "during_run": {}, "lost_records": 0, "overruns": 0},
    }))
    recs = [
        {"v": 1, "t_ns": 1, "src": "probe", "probe": probe, "backend": backend,
         "type": "probe_start", "params": {}},
        {"v": 1, "t_ns": 2, "src": "probe", "probe": probe, "backend": backend,
         "type": "declared", "key": "device.name", "value": device, "source": "fake"},
    ]
    for i, (name_, value, unit, ts) in enumerate(measures):
        recs.append({"v": 1, "t_ns": 3 + i, "src": "probe", "probe": probe,
                     "backend": backend, "type": "measure", "name": name_, "value": value,
                     "unit": unit, "tags": ts, "verified": True})
    for i, (name_, ok, detail) in enumerate(checks or []):
        recs.append({"v": 1, "t_ns": 10 + i, "src": "probe", "probe": probe,
                     "backend": backend, "type": "check", "name": name_, "pass": ok,
                     "detail": detail})
    recs.append({"v": 1, "t_ns": 99, "src": "probe", "probe": probe, "backend": backend,
                 "type": "probe_end", "status": status, "failed_checks": 0})
    (d / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "env.json").write_text(json.dumps({
        "v": 1, "nvidia": {"nvml": {"driver_version": None}},
        "amd": {"cards": [{"vbios_version": "V1", "driver_version": None,
                           "performance_level": "auto",
                           "clocks": {"sclk": {"forced_level": "auto"},
                                      "mclk": {"forced_level": "auto"}}}]},
        "firmware": {"dmi": {"bios_version": "1028"}},
        "toolchain": {"build": {"arch": "sm_121a", "nvcc": "/opt/cuda/bin/nvcc",
                                "flags": "-O3", "built_utc": "2026-01-01T00:00:00Z"}},
    }))
    return d


MEM_A = [("gpu_read", 600.0, "GB/s", {"source": "vulkan_device_local"}),
         ("copy_payload", 50.0, "GB/s", {"source": "vulkan_host_visible", "direction": "h2d"})]


class DiffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _diff(self, a, b, tol=0.1):
        return diffmod.diff_runs(self.root / a, self.root / b, tol)

    def test_within_tolerance(self):
        make_run(self.root, "a", "memory", "vulkan", "idle", MEM_A, "2026-01-01T00:00:00Z")
        make_run(self.root, "b", "memory", "vulkan", "idle",
                 [("gpu_read", 620.0, "GB/s", {"source": "vulkan_device_local"}),
                  ("copy_payload", 52.0, "GB/s",
                   {"source": "vulkan_host_visible", "direction": "h2d"})],
                 "2026-01-02T00:00:00Z")
        d = self._diff("a", "b")
        self.assertEqual(d.exit_code, 0)
        self.assertTrue(all(r.verdict == "within" for r in d.rows))
        self.assertIn("within", diffmod.render_diff(d))

    def test_beyond_tolerance(self):
        make_run(self.root, "a", "memory", "vulkan", "idle", MEM_A, "2026-01-01T00:00:00Z")
        make_run(self.root, "b", "memory", "vulkan", "idle",
                 [("gpu_read", 700.0, "GB/s", {"source": "vulkan_device_local"}),
                  ("copy_payload", 50.0, "GB/s",
                   {"source": "vulkan_host_visible", "direction": "h2d"})],
                 "2026-01-02T00:00:00Z")
        d = self._diff("a", "b")
        self.assertEqual(d.exit_code, 1)
        verdicts = {r.key: r.verdict for r in d.rows}
        self.assertEqual(verdicts["memory.gpu_read.device_local"], "beyond")
        self.assertEqual(verdicts["memory.copy.h2d.host_visible"], "within")

    def test_missing_on_one_side(self):
        make_run(self.root, "a", "memory", "vulkan", "idle", MEM_A, "2026-01-01T00:00:00Z")
        make_run(self.root, "b", "memory", "vulkan", "idle",
                 [("gpu_read", 605.0, "GB/s", {"source": "vulkan_device_local"})],
                 "2026-01-02T00:00:00Z")
        d = self._diff("a", "b")
        self.assertEqual(d.exit_code, 0)  # missing is not beyond
        self.assertEqual({r.key: r.verdict for r in d.rows}
                         ["memory.copy.h2d.host_visible"], "missing")

    def test_env_delta_reports_what_changed(self):
        make_run(self.root, "a", "memory", "vulkan", "idle", MEM_A, "2026-01-01T00:00:00Z")
        b = make_run(self.root, "b", "memory", "vulkan", "idle", MEM_A, "2026-01-02T00:00:00Z")
        env = json.loads((b / "env.json").read_text())
        env["amd"]["cards"][0]["performance_level"] = "manual"
        env["toolchain"]["build"]["built_utc"] = "2026-01-05T00:00:00Z"
        (b / "env.json").write_text(json.dumps(env))
        d = self._diff("a", "b")
        keys = {k for k, _, _ in d.env_rows}
        self.assertIn("amd.cards[0].performance_level", keys)
        self.assertIn("toolchain.build.built_utc", keys)
        self.assertNotIn("amd.cards[0].vbios_version", keys)

    def test_regress_picks_the_newest_matching_baseline(self):
        make_run(self.root, "old", "memory", "vulkan", "idle", MEM_A, "2026-01-01T00:00:00Z")
        make_run(self.root, "newer", "memory", "vulkan", "idle", MEM_A, "2026-01-03T00:00:00Z")
        make_run(self.root, "new", "memory", "vulkan", "idle", MEM_A, "2026-01-04T00:00:00Z")
        # a run with a different label and a different device are not baselines
        make_run(self.root, "otherlabel", "memory", "vulkan", "cpu50", MEM_A,
                 "2026-01-05T00:00:00Z")
        make_run(self.root, "otherdevice", "memory", "vulkan", "idle", MEM_A,
                 "2026-01-06T00:00:00Z", device="NVIDIA RTX 5090")
        self.assertEqual(diffmod.find_baseline(self.root / "new", self.root),
                         self.root / "newer")

    def test_regress_baseline_never_picks_the_new_run_itself(self):
        make_run(self.root, "new", "memory", "vulkan", "idle", MEM_A, "2026-01-02T00:00:00Z")
        self.assertIsNone(diffmod.find_baseline(self.root / "new", self.root))
        self.assertEqual([c.dir.name for c in
                          diffmod.candidates_for(self.root / "new", self.root)], [])

    def test_regress_no_baseline_lists_candidates(self):
        make_run(self.root, "other", "memory", "cuda", "idle", MEM_A, "2026-01-01T00:00:00Z")
        make_run(self.root, "new", "memory", "vulkan", "idle", MEM_A, "2026-01-02T00:00:00Z")
        self.assertIsNone(diffmod.find_baseline(self.root / "new", self.root))
        cands = diffmod.candidates_for(self.root / "new", self.root)
        self.assertEqual([c.dir.name for c in cands], [])  # no same probe+backend run

    def test_cli_diff_exit_codes(self):
        make_run(self.root, "a", "memory", "vulkan", "idle", MEM_A, "2026-01-01T00:00:00Z")
        make_run(self.root, "b", "memory", "vulkan", "idle",
                 [("gpu_read", 900.0, "GB/s", {"source": "vulkan_device_local"})],
                 "2026-01-02T00:00:00Z")
        p = subprocess.run([sys.executable, str(ROOT / "echolock"), "diff",
                            str(self.root / "a"), str(self.root / "b"), "--tol", "0.1"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        p = subprocess.run([sys.executable, str(ROOT / "echolock"), "diff",
                            str(self.root / "nope"), str(self.root / "b")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        p = subprocess.run([sys.executable, str(ROOT / "echolock"), "regress",
                            str(self.root / "b"), "--runs", str(self.root / "empty")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)  # no baseline under an empty run root


class CrosscheckTest(unittest.TestCase):
    """The phase-3 check logic on synthetic two-backend run dirs (this box has no NVIDIA GPU,
    so the first real two-backend verdict happens elsewhere)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _memory_pair(self, a_val, b_val):
        make_run(self.root, "ra", "memory", "vulkan", "idle",
                 [("gpu_read", a_val, "GB/s", {"source": "vulkan_device_local"}),
                  ("copy_payload", 50.0, "GB/s",
                   {"source": "vulkan_host_visible", "direction": "h2d"})],
                 "2026-01-01T00:00:00Z")
        make_run(self.root, "rb", "memory", "cuda", "idle",
                 [("gpu_read", b_val, "GB/s", {"source": "cudaMalloc"}),
                  ("copy_payload", 51.0, "GB/s",
                   {"source": "cudaHostAlloc_mapped", "direction": "h2d"})],
                 "2026-01-01T01:00:00Z")

    def test_memory_within_tolerance(self):
        self._memory_pair(600.0, 630.0)
        recs = diffmod.crosscheck_verdicts(self.root / "ra", self.root / "rb")
        by = {r["name"]: r for r in recs}
        self.assertEqual(len(recs), 2)
        self.assertEqual(by["memory.gpu_read.device_local"]["verdict"], "within-tolerance")
        self.assertEqual(by["memory.gpu_read.device_local"]["tolerance"], 0.10)
        self.assertEqual(by["memory.copy.h2d.host_visible"]["verdict"], "within-tolerance")
        self.assertEqual(recs[0]["src"], "crosscheck")
        self.assertEqual(recs[0]["type"], "crosscheck")
        self.assertEqual(recs[0]["a"]["run_id"], "ra")
        self.assertEqual(recs[0]["b"]["run_id"], "rb")

    def test_memory_diverges(self):
        self._memory_pair(600.0, 800.0)
        recs = diffmod.crosscheck_verdicts(self.root / "ra", self.root / "rb")
        by = {r["name"]: r for r in recs}
        self.assertEqual(by["memory.gpu_read.device_local"]["verdict"], "diverges")

    def test_memory_missing_value(self):
        make_run(self.root, "ra", "memory", "vulkan", "idle",
                 [("gpu_read", 600.0, "GB/s", {"source": "vulkan_device_local"})],
                 "2026-01-01T00:00:00Z")
        make_run(self.root, "rb", "memory", "cuda", "idle",
                 [("gpu_read", 600.0, "GB/s", {"source": "cudaMalloc"})],
                 "2026-01-01T01:00:00Z")
        recs = diffmod.crosscheck_verdicts(self.root / "ra", self.root / "rb")
        by = {r["name"]: r for r in recs}
        self.assertEqual(by["memory.copy.h2d.host_visible"]["verdict"], "missing")

    def _mma_pair(self, a_xor, b_xor):
        checks = [(f"{fmt}.result", True, f"ref_xor={a_xor[fmt]}" if a_xor else {})
                  for fmt in ("fp16", "bf16", "int8")]
        checks_b = [(f"{fmt}.result", True, f"ref_xor={b_xor[fmt]}" if b_xor else {})
                    for fmt in ("fp16", "bf16", "int8")]
        make_run(self.root, "ra", "mma", "vulkan", "idle",
                 [("fp16_tflops", 100.0, "TFLOPS", {}),
                  ("bf16_tflops", 95.0, "TFLOPS", {}),
                  ("int8_tflops", 180.0, "TFLOPS", {})],
                 "2026-01-01T00:00:00Z", checks=checks)
        make_run(self.root, "rb", "mma", "cuda", "idle",
                 [("mma_throughput", 105.0, "TFLOPS", {"format": "fp16"}),
                  ("mma_throughput", 99.0, "TFLOPS", {"format": "bf16"}),
                  ("mma_throughput", 190.0, "TFLOPS", {"format": "int8"})],
                 "2026-01-01T01:00:00Z", checks=checks_b)

    def test_mma_ref_xor_equal(self):
        self._mma_pair({"fp16": "ab", "bf16": "cd", "int8": "ef"},
                       {"fp16": "ab", "bf16": "cd", "int8": "ef"})
        recs = diffmod.crosscheck_verdicts(self.root / "ra", self.root / "rb")
        by = {r["name"]: r for r in recs}
        self.assertEqual(by["mma.ref_xor"]["verdict"], "agrees")
        self.assertIsNone(by["mma.ref_xor"]["tolerance"])
        # the ceilings themselves are within tolerance, aligned across naming schemes
        self.assertEqual(by["mma.tflops.fp16"]["verdict"], "within-tolerance")
        self.assertEqual(by["mma.tflops.int8"]["verdict"], "within-tolerance")

    def test_mma_ref_xor_different(self):
        self._mma_pair({"fp16": "ab", "bf16": "cd", "int8": "ef"},
                       {"fp16": "ab", "bf16": "cd", "int8": "ef00"})
        recs = diffmod.crosscheck_verdicts(self.root / "ra", self.root / "rb")
        by = {r["name"]: r for r in recs}
        self.assertEqual(by["mma.ref_xor"]["verdict"], "diverges")

    def test_mma_ref_xor_absent_is_missing_not_pass(self):
        self._mma_pair({"fp16": "ab", "bf16": "cd", "int8": "ef"}, {})
        recs = diffmod.crosscheck_verdicts(self.root / "ra", self.root / "rb")
        by = {r["name"]: r for r in recs}
        self.assertEqual(by["mma.ref_xor"]["verdict"], "missing")
        self.assertIn("not a pass", by["mma.ref_xor"]["detail"])

    def test_pointer_chase_footprint_pair(self):
        hops_a = [("chase_hop_latency", 250.0, "ns/hop",
                   {"kind": "vulkan_device_local", "footprint_mib": 0.0625, "pattern": "random"}),
                  ("chase_hop_latency", 260.0, "ns/hop",
                   {"kind": "vulkan_device_local", "footprint_mib": 64, "pattern": "random"})]
        hops_b = [("chase_hop_latency", 255.0, "ns/hop",
                   {"kind": "device", "footprint_mib": 0.0625, "pattern": "random"}),
                  ("chase_hop_latency", 400.0, "ns/hop",
                   {"kind": "device", "footprint_mib": 64, "pattern": "random"})]
        make_run(self.root, "ra", "pointer_chase", "vulkan", "idle", hops_a,
                 "2026-01-01T00:00:00Z")
        make_run(self.root, "rb", "pointer_chase", "cuda", "idle", hops_b,
                 "2026-01-01T01:00:00Z")
        recs = diffmod.crosscheck_verdicts(self.root / "ra", self.root / "rb")
        self.assertEqual(len(recs), 2)
        by = {r["name"]: r for r in recs}
        small = [r for r in recs if "smallest" in r["name"]][0]
        large = [r for r in recs if "largest" in r["name"]][0]
        self.assertEqual(small["verdict"], "within-tolerance")
        self.assertEqual(large["verdict"], "diverges")
        text = diffmod.render_crosscheck_section(recs)
        self.assertIn("## Crosscheck", text)
        self.assertIn("diverges", text)


if __name__ == "__main__":
    unittest.main()
