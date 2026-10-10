"""The Vulkan backend, exercised for real: the memory probe runs on whatever Vulkan device this
machine has (an AMD GPU through RADV here) and every check must pass. Skipped when the binary
is not built or no Vulkan device is present."""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "vulkan" / "memory"
SHADER = ROOT / "build" / "vulkan" / "read_fold.spv"


def vulkan_device_present() -> bool:
    icd = Path("/usr/share/vulkan/icd.d")
    return icd.is_dir() and any(icd.glob("*.json"))


@unittest.skipUnless(EXE.exists() and SHADER.exists(), "build/vulkan/memory not built")
@unittest.skipUnless(vulkan_device_present(), "no Vulkan ICD present")
class VulkanMemoryTest(unittest.TestCase):
    def run_probe(self, extra):
        p = subprocess.run([str(EXE), "--mib", "16", "--seconds", "0.05",
                            "--cpu-threads", "2", *extra],
                           capture_output=True, text=True, timeout=120)
        return p, [json.loads(l) for l in p.stdout.splitlines()]

    def test_memory_probe_contract(self):
        p, recs = self.run_probe(["--device", "0"])
        if any(r["type"] == "error" for r in recs):
            self.skipTest(f"no runnable Vulkan device: {recs[0].get('message')}")
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        self.assertEqual(recs[0]["type"], "probe_start")
        self.assertEqual(recs[0]["probe"], "memory")
        self.assertEqual(recs[0]["backend"], "vulkan")
        end = recs[-1]
        self.assertEqual((end["type"], end["status"]), ("probe_end", "ok"))
        self.assertEqual(end["failed_checks"], 0)
        checks = [r for r in recs if r["type"] == "check"]
        self.assertTrue(checks and all(c["pass"] for c in checks))
        # Declared values and verified measures exist, same contract as the CUDA probes.
        declared = {r["key"] for r in recs if r["type"] == "declared"}
        self.assertIn("vulkan.api_version", declared)
        self.assertIn("device.name", declared)
        measures = [r for r in recs if r["type"] == "measure"]
        summaries = [r for r in measures if "per_rep" in r.get("tags", {})]
        self.assertTrue(summaries)
        self.assertTrue(all(r.get("verified") for r in summaries))
        names = {r["name"] for r in measures}
        self.assertTrue({"gpu_read", "copy_payload", "cpu_read"} <= names)


CHASE = ROOT / "build" / "vulkan" / "pointer_chase"
CHASE_SHADER = ROOT / "build" / "vulkan" / "chase.spv"


@unittest.skipUnless(CHASE.exists() and CHASE_SHADER.exists(),
                     "build/vulkan/pointer_chase not built")
@unittest.skipUnless(vulkan_device_present(), "no Vulkan ICD present")
class VulkanPointerChaseTest(unittest.TestCase):
    def test_latency_map_contract(self):
        p = subprocess.run([str(CHASE), "--footprints", "0.0625,1", "--hops", "256",
                            "--reps", "2", "--kinds", "device", "--patterns", "seq",
                            "--device", "0"],
                           capture_output=True, text=True, timeout=120)
        recs = [json.loads(l) for l in p.stdout.splitlines()]
        if any(r["type"] == "error" for r in recs):
            self.skipTest(f"no runnable Vulkan device: {recs[0].get('message')}")
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        end = recs[-1]
        self.assertEqual((end["type"], end["status"]), ("probe_end", "ok"))
        checks = [r for r in recs if r["type"] == "check"]
        self.assertTrue(checks and all(c["pass"] for c in checks))
        summaries = [r for r in recs if r["type"] == "measure" and "per_rep" in r.get("tags", {})]
        self.assertTrue(summaries)
        self.assertTrue(all(r.get("verified") for r in summaries))
        for r in summaries:
            self.assertGreater(r["value"], 0.0)
            self.assertLess(r["value"], 100000.0)  # a hop latency, sanity band


if __name__ == "__main__":
    unittest.main()
