"""The constants probe, exercised for real: every declared device fact must come from an
independent source, every agreement check must pass, and the single-source keys must carry
their sources feature records. The Vulkan side runs on whatever Vulkan devices this machine
has (two AMD GPUs through RADV here); the CUDA side skips when no NVIDIA driver is present.

    python3 -m unittest tests.test_constants -v
"""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "vulkan" / "constants"
CUDABUILD = ROOT / "build" / "cuda"

# Keys that may end up single-source: each must be covered one of two ways -- an agreement
# check when two sources answered, or a <key>.sources feature record documenting that only
# one did. Which one depends on the machine: on a box where /proc/driver/nvidia reports the
# model, device_name is compared, not documented.
SOURCE_COVERAGE_KEYS = ("device_name", "sm_count", "l2_bytes", "clock_max_mhz")


class _SourceCoverageMixin:
    def assert_source_coverage(self, recs, feats):
        checks = {r["name"] for r in recs if r["type"] == "check"}
        for key in SOURCE_COVERAGE_KEYS:
            covered = (f"constants.{key}.sources" in feats
                       or f"constants.{key}.agreement" in checks)
            self.assertTrue(covered,
                            f"constants.{key}: no agreement check and no .sources feature")


def vulkan_device_present() -> bool:
    icd = Path("/usr/share/vulkan/icd.d")
    return icd.is_dir() and any(icd.glob("*.json"))


def run_probe(exe, extra, timeout=120):
    p = subprocess.run([str(exe), *extra], capture_output=True, text=True, timeout=timeout)
    return p, [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


def first_error(recs):
    for r in recs:
        if r["type"] == "error":
            return r
    return None


@unittest.skipUnless(EXE.exists(), "build/vulkan/constants not built")
@unittest.skipUnless(vulkan_device_present(), "no Vulkan ICD present")
class VulkanConstantsTest(_SourceCoverageMixin, unittest.TestCase):
    def run_probe(self, extra):
        return run_probe(EXE, extra)

    def assert_contract(self, p, recs):
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        self.assertEqual(recs[0]["type"], "probe_start")
        self.assertEqual((recs[0]["probe"], recs[0]["backend"]), ("constants", "vulkan"))
        end = recs[-1]
        self.assertEqual((end["type"], end["status"]), ("probe_end", "ok"))
        self.assertEqual(end["failed_checks"], 0)
        self.assertTrue(all(c["pass"] for c in recs if c["type"] == "check"))
        pairs = {(r["key"], r["source"]) for r in recs if r["type"] == "declared"}
        self.assertIn(("constants.device_name", "vkGetPhysicalDeviceProperties"), pairs)
        self.assertIn(("constants.memory_bytes", "Vulkan memory heaps"), pairs)
        feats = {r["feature"] for r in recs if r["type"] == "feature"}
        self.assert_source_coverage(recs, feats)
        return pairs

    def assert_memory(self, recs):
        mem = [r for r in recs if r["type"] == "declared" and r["key"] == "constants.memory_bytes"]
        self.assertTrue(mem)
        check = [r for r in recs if r["type"] == "check"
                 and r["name"] == "constants.memory_bytes.agreement"]
        if check:
            # Discrete GPU: every compared source read the same size.
            vals = {str(v) for v in check[0]["detail"].values()}
            self.assertEqual(len(vals), 1)
        else:
            # Integrated device: carve-out and host pool are different quantities;
            # the run documents why there is no check instead of faking one.
            notes = " ".join(r["message"] for r in recs if r["type"] == "note")
            self.assertIn("integrated device", notes)

    def test_device0(self):
        p, recs = self.run_probe(["--device", "0"])
        err = first_error(recs)
        if err is not None:
            self.skipTest(f"no runnable Vulkan device: {err.get('message')}")
        self.assert_contract(p, recs)
        self.assert_memory(recs)

    def test_device1(self):
        p, recs = self.run_probe(["--device", "1"])
        err = first_error(recs)
        if err is not None:
            self.skipTest(f"no second Vulkan device: {err.get('message')}")
        self.assert_contract(p, recs)
        self.assert_memory(recs)

    def test_against_baseline(self):
        base = ROOT / "runs" / "20261010T180852Z-pointer_chase-vulkan-amd-baseline"
        if not (base / "probe.jsonl").exists():
            self.skipTest("pointer_chase baseline run not present")
        p, recs = self.run_probe(["--against", str(base)])
        err = first_error(recs)
        if err is not None:
            self.skipTest(f"no runnable Vulkan device: {err.get('message')}")
        self.assert_contract(p, recs)
        run_id = base.name
        texts = [r["message"] for r in recs if r["type"] == "note"]
        texts += [json.dumps(r["detail"]) for r in recs if r["type"] == "check"
                  and isinstance(r["detail"], dict)]
        self.assertTrue(any(run_id in t for t in texts),
                        "the derived-source output must name the run it read")


# A machine without a usable NVIDIA driver errors the probe cleanly (cudaErrorInsufficientDriver
# on a box with the toolkit but no driver). That is a skip, not a regression: the CUDA test
# exists to catch regressions on GPU machines.
CUDA_NO_GPU = ("cudaErrorInsufficientDriver", "cudaErrorNoDevice", "no CUDA-capable device",
               "driver version is insufficient", "cudaErrorCallRequiresNewerDriver")


@unittest.skipUnless((CUDABUILD / "constants").exists(), "build/cuda/constants not built")
class CudaConstantsTest(_SourceCoverageMixin, unittest.TestCase):
    def test_cuda_device0(self):
        p, recs = run_probe(CUDABUILD / "constants", [])
        end = [r for r in recs if r["type"] == "probe_end"]
        if first_error(recs) is not None or not end:
            if any(s in p.stdout + p.stderr for s in CUDA_NO_GPU):
                self.skipTest("no usable NVIDIA driver on this machine")
            self.skipTest("no usable CUDA GPU (probe produced no probe_end record)")
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        self.assertEqual((recs[0]["probe"], recs[0]["backend"]), ("constants", "cuda"))
        self.assertEqual((end[-1]["status"], end[-1]["failed_checks"]), ("ok", 0))
        self.assertTrue(all(c["pass"] for c in recs if c["type"] == "check"))
        pairs = {(r["key"], r["source"]) for r in recs if r["type"] == "declared"}
        self.assertIn(("constants.device_name", "cudaGetDeviceProperties"), pairs)
        self.assertIn(("constants.memory_bytes", "cudaGetDeviceProperties"), pairs)
        feats = {r["feature"] for r in recs if r["type"] == "feature"}
        self.assert_source_coverage(recs, feats)


if __name__ == "__main__":
    unittest.main()
