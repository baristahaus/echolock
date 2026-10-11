"""The constants probe is host-only, so its parsing is testable without a GPU -- and it needs to
be, because on GB10 it was silently reading one source and reporting `ok`. Fixture XML is trimmed
from real `nvidia-smi -q -x` output on an Asus GX10 (driver 580.173.02): clocks are `sm_clock`
with units, memory totals are `N/A`, and the device identity lives in a <uuid> child element while
the id attribute is a bus address in a format no other source uses.

Regression guards, in order of how much damage each hides:
  1. nvidia_smi() must return the device under `gpus` and carry its UUID -- the probe read
     smi["gpu"] (singular), so the source was empty on every platform and `status ok` lied.
  2. devices must bucket under one key across sources, or nothing is ever compared.
  3. `N/A` must become None, never 0: a zero is a wrong fact, a None is a missing one.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.host import constants  # noqa: E402
from env import capture  # noqa: E402

UUID = "GPU-5a82916b-5815-3cd5-7be3-b04633a2141f"

# Trimmed real output. Structure kept exact: the bug this file guards was a schema mismatch.
SMI_XML = f"""<?xml version="1.0"?>
<nvidia_smi_log>
  <driver_version>580.173.02</driver_version>
  <cuda_version>13.2</cuda_version>
  <attached_gpus>1</attached_gpus>
  <gpu id="0000000F:01:00.0">
    <product_name>NVIDIA GB10</product_name>
    <uuid>{UUID}</uuid>
    <vbios_version>9A.0B.25.00.00</vbios_version>
    <gsp_firmware_version>580.173.02</gsp_firmware_version>
    <fb_memory_usage><total>N/A</total><used>N/A</used></fb_memory_usage>
    <max_clocks>
      <graphics_clock>3003 MHz</graphics_clock>
      <sm_clock>3003 MHz</sm_clock>
      <mem_clock>N/A</mem_clock>
    </max_clocks>
    <power><instant_power_draw>35.69 W</instant_power_draw>
      <current_power_limit>N/A</current_power_limit></power>
  </gpu>
</nvidia_smi_log>
"""

NVML_FACTS = {  # what NVML reports for the same device on the same box
    "name": "NVIDIA GB10", "uuid": UUID, "vbios": "9A.0B.25.00.00",
    "max_sm_clock_mhz": 3003, "max_mem_clock_mhz": None,
    "power_limit_w": None, "memory_total": None, "pci_bus_id": "0001:01:00.0",
}


class _FakeNvml:
    available, handles = True, ["h0"]

    def describe(self, _):
        return dict(NVML_FACTS)


class _NoProcDir:
    """Stand in for Path so /proc/driver/nvidia/gpus is absent: this test is about NVIDIA sources,
    and on GB10 that file reports `Model: Unknown` with no Bus Location anyway."""

    def __init__(self, *_a):
        pass

    def glob(self, _pat):
        return []

    def read_text(self):
        return ""


class TestSmiParsing(unittest.TestCase):
    def setUp(self):
        self._run_cmd = capture.run_cmd
        self._nvml, self._path = constants.Nvml, constants.Path
        capture.run_cmd = lambda cmd, **kw: SMI_XML if "-x" in cmd else "GPU 0: NVIDIA GB10 (UUID: %s)" % UUID
        constants.Nvml, constants.Path = _FakeNvml, _NoProcDir

    def tearDown(self):
        capture.run_cmd = self._run_cmd
        constants.Nvml, constants.Path = self._nvml, self._path

    def test_device_list_is_returned_under_gpus_with_its_uuid(self):
        d = capture.nvidia_smi()
        self.assertIn("gpus", d, "constants.py once read smi['gpu']; that key never existed")
        self.assertEqual(len(d["gpus"]), 1)
        # The identity must be carried, or cross-source comparison has nothing to join on:
        # nvidia-smi's id attribute (`0000000F:01:00.0`) matches no other source's bus id.
        self.assertEqual(d["gpus"][0]["uuid"], UUID)

    def test_constants_sees_two_sources_for_one_device(self):
        srcs = constants.nvidia_sources()
        self.assertEqual(list(srcs), [UUID], "one GPU must be one bucket, keyed by identity")
        self.assertEqual(sorted(srcs[UUID]), ["nvidia_smi", "nvml"])

    def test_clocks_compare_after_unit_strip_and_na_is_missing_not_zero(self):
        facts = constants.nvidia_sources()[UUID]["nvidia_smi"]
        self.assertEqual(facts["max_sm_clock_mhz"], 3003)   # from <sm_clock>3003 MHz</sm_clock>
        self.assertIsNone(facts["max_mem_clock_mhz"])       # <mem_clock>N/A</mem_clock>
        self.assertIsNone(facts["memory_bytes"])            # unified memory: no frame buffer
        self.assertIsNone(facts["power_limit_w"])           # no power limit on this part
        self.assertEqual(facts["vbios"], NVML_FACTS["vbios"])

    def test_compare_emits_no_failure_when_the_two_sources_agree(self):
        self.assertEqual(constants.compare(constants.nvidia_sources(), []), 0)

    def test_a_disagreement_is_a_failure_not_a_note(self):
        bad = constants.nvidia_sources()
        bad[UUID]["nvidia_smi"]["max_sm_clock_mhz"] = 1500  # outside the 2% tolerance
        emitted = []
        real_check = constants.check
        constants.check = lambda name, ok, detail: emitted.append((name, ok))  # emit nothing for real
        try:
            self.assertEqual(constants.compare(bad, []), 1)
            self.assertTrue(any(name.endswith("max_sm_clock_mhz") and not ok for name, ok in emitted))
        finally:
            constants.check = real_check


if __name__ == "__main__":
    unittest.main()
