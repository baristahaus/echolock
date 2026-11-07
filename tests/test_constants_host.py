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


class _FakeKfdPath:
    """Stand-in for Path: the KFD topology properties file, with a real node's field order."""

    def __init__(self, *parts):
        self.parts = parts

    def __truediv__(self, part):
        return _FakeKfdPath(*(self.parts + (part,)))

    def read_text(self):
        return "cpu_cores_count 0\ngfx_target_version 120001\nvendor_id 4098\n" \
               "device_id 30032\nlocation_id 768\ndomain 0\n"


class _FakeAmCard:
    def __init__(self, name):
        self.name = name

    def __truediv__(self, _part):
        return self


class _FakeAmdgpu:
    """The env.amdgpu surface amd_sources() uses, shaped like the S3-rewritten API:
    dpm_table returns {"states": {level: MHz}, ...}, describe carries `pci` as a BDF string
    and `device_id` as hex text, _kfd_nodes yields node id + mem_banks (no device_id)."""

    def __init__(self):
        self.cards_list = [_FakeAmCard("card0"), _FakeAmCard("card1")]
        self.desc = {
            "card0": {"pci": "0000:03:00.0", "device_id": "0x7550",
                      "vram_total_bytes": 17095983104, "vbios_version": "D05"},
            "card1": {"pci": "0000:0e:00.0", "device_id": "0x13c0",
                      "vram_total_bytes": 2147483648, "vbios_version": None},
        }
        self._dpm = {
            "pp_dpm_sclk": {"states": {"0": 1400.0, "s": 2970.0}, "active": "s",
                            "forced_level": "auto"},
            "pp_dpm_mclk": {"states": {"0": 2551.0}, "active": "0", "forced_level": "auto"},
        }
        self.kfd_nodes = [
            {"node": "1", "mem_banks": [{"size_bytes": 17095983104}]},
            {"node": "2", "mem_banks": [{"size_bytes": 32467476480}]},
        ]
        self.node_device_ids = {"1": 0x7550, "2": 0x13c0}

    def cards(self):
        return self.cards_list

    def describe(self, card):
        return self.desc[card.name]

    def dpm_table(self, _dev, attr):
        return self._dpm.get(attr, {})

    def hwmon(self, _dev):
        return {}

    def _kfd_nodes(self):
        return self.kfd_nodes


class TestAmdSources(unittest.TestCase):
    """The AMD path is the one that broke: the probe predated the env/amdgpu rewrite (dpm_table
    shape, _kfd_nodes fields) and silently collected nothing, then compared a PCI device ID
    against marketing names and reported failure. Guards: the node-properties parse, the
    one-GPU-one-bucket rule, and the dpm_table states dict."""

    def setUp(self):
        self._amdgpu, self._path = constants.amdgpu, constants.Path
        self._node_devid = constants._kfd_node_device_id
        self.fake = _FakeAmdgpu()
        constants.amdgpu = self.fake
        constants.Path = _FakeKfdPath
        constants._kfd_node_device_id = self.fake.node_device_ids.get

    def tearDown(self):
        constants.amdgpu, constants.Path = self._amdgpu, self._path
        constants._kfd_node_device_id = self._node_devid

    def test_kfd_node_device_id_reads_the_topology_properties(self):
        # The real parser (self._node_devid) against the fake properties file: 30032 = 0x7550.
        self.assertEqual(self._node_devid("1"), 30032)

    def test_each_gpu_is_one_bucket_with_both_kernel_views(self):
        srcs = constants.amd_sources()
        self.assertEqual(sorted(srcs), ["0000:03:00.0", "0000:0e:00.0"],
                         "the KFD node must join its card, not form a kfdN bucket")
        for key in srcs:
            self.assertEqual(sorted(srcs[key]), ["amdgpu_sysfs", "kfd"])
        d = srcs["0000:03:00.0"]
        self.assertEqual(d["amdgpu_sysfs"]["max_sm_clock_mhz"], 2970.0)  # max of the states dict
        self.assertEqual(d["amdgpu_sysfs"]["memory_bytes"], 17095983104)
        self.assertEqual(d["kfd"]["memory_bytes"], 17095983104)

    def test_unmatched_kfd_node_keeps_its_own_bucket(self):
        self.fake.kfd_nodes.append({"node": "7", "mem_banks": [{"size_bytes": 8 * 2**30}]})
        self.fake.node_device_ids["7"] = 0x9999
        srcs = constants.amd_sources()
        self.assertIn("kfd7", srcs)

    def test_sysfs_exposes_device_id_not_a_name(self):
        srcs = constants.amd_sources()
        self.assertEqual(srcs["0000:03:00.0"]["amdgpu_sysfs"]["device_id"], "0x7550")
        self.assertNotIn("name", srcs["0000:03:00.0"]["amdgpu_sysfs"],
                         "the PCI device ID is not a name; a name check on it can only fail")


def _amd_devices():
    return {
        "0000:03:00.0": {
            "amdgpu_sysfs": {"device_id": "0x7550", "memory_bytes": 17095983104,
                             "max_sm_clock_mhz": 2970.0},
            "kfd": {"memory_bytes": 17095983104},
        },
        "0000:0e:00.0": {  # UMA: the KFD bank sizes the host pool, the drm value the carve-out
            "amdgpu_sysfs": {"device_id": "0x13c0", "memory_bytes": 2 * 2**30,
                             "max_sm_clock_mhz": 2900.0},
            "kfd": {"memory_bytes": 30 * 2**30},
        },
    }


class TestAmdCompare(unittest.TestCase):
    """compare() on AMD-shaped buckets: the UMA spread must be a note, never a failure, and the
    vulkan voice must match on device ID and stay silent on facts this vulkaninfo does not print."""

    def setUp(self):
        self.emitted, self.notes = [], []
        self._check, self._note = constants.check, constants.note
        constants.check = lambda name, ok, detail: self.emitted.append((name, ok))
        constants.note = lambda msg: self.notes.append(msg)

    def tearDown(self):
        constants.check, constants.note = self._check, self._note

    def test_uma_carveout_is_a_note_not_a_failure(self):
        self.assertEqual(constants.compare(_amd_devices(), []), 0)
        self.assertFalse(any(n == "0000:0e:00.0.memory_bytes" for n, _ in self.emitted),
                         "a >4x spread means different quantities, not a mismatch")
        self.assertTrue(any("UMA" in n for n in self.notes))

    def test_agreeing_sources_emit_no_check(self):
        self.assertEqual(constants.compare(_amd_devices(), []), 0)
        self.assertFalse(any(n == "0000:03:00.0.memory_bytes" for n, _ in self.emitted),
                         "agreement is the expected case; the declared records show it")

    def test_device_id_matches_the_vulkan_stack(self):
        vk = [{"name": "AMD Radeon RX 9070 XT (RADV GFX1201)", "device_id": "0x7550",
               "memory_bytes": None},
              {"name": "AMD Ryzen 7 9800X3D 8-Core Processor (RADV RAPHAEL_MENDOCINO)",
               "device_id": "0x13c0", "memory_bytes": None}]
        self.assertEqual(constants.compare(_amd_devices(), vk), 0)
        names = [n for n, ok in self.emitted]
        self.assertIn("0000:03:00.0.device_id.vs_vulkan", names)
        self.assertIn("0000:0e:00.0.device_id.vs_vulkan", names)
        self.assertFalse(any(n.endswith("memory_bytes.vs_vulkan") for n in names),
                         "vulkaninfo --summary prints no deviceSize: nothing to compare")

    def test_device_id_mismatch_is_a_failure(self):
        vk = [{"name": "X", "device_id": "0xdead", "memory_bytes": None}]
        self.assertEqual(constants.compare(_amd_devices(), vk), 2,
                         "a wrong ID in both cards is a real disagreement")


if __name__ == "__main__":
    unittest.main()
