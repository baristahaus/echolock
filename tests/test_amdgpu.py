"""env/amdgpu.py against a fake sysfs tree: normalization, and the layout cross-check.

Skips nothing: the fake tree stands in for /sys so the parsing rules are tested on any machine.
"""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import amdgpu                                     # noqa: E402
from env.sampler import Sampler                            # noqa: E402


def metrics_blob(version=1, revision=3, gfxclk=2100, edge=57, power=123, energy=987654,
                 throttle=0, size=120) -> bytes:
    """A gpu_metrics_v1_3 buffer, laid out the way kgd_pp_interface.h orders the first 80 bytes."""
    blob = bytearray(size)
    struct.pack_into("<HBB", blob, 0, size, version, revision)
    struct.pack_into("<H", blob, 4, edge)            # temperature_edge
    struct.pack_into("<H", blob, 6, edge + 5)        # temperature_hotspot
    struct.pack_into("<H", blob, 8, 80)              # temperature_mem
    struct.pack_into("<H", blob, 16, 42)             # average_gfx_activity
    struct.pack_into("<H", blob, 18, 11)             # average_umc_activity
    struct.pack_into("<H", blob, 22, power)          # average_socket_power
    struct.pack_into("<Q", blob, 24, energy)         # energy_accumulator
    struct.pack_into("<H", blob, 40, gfxclk)         # average_gfxclk_frequency
    struct.pack_into("<H", blob, 44, 1258)           # average_uclk_frequency
    struct.pack_into("<H", blob, 54, gfxclk)         # current_gfxclk
    struct.pack_into("<H", blob, 58, 1258)           # current_uclk
    struct.pack_into("<I", blob, 68, throttle)       # throttle_status
    struct.pack_into("<H", blob, 74, 16)             # pcie_link_width
    struct.pack_into("<H", blob, 76, 25)             # pcie_link_speed, 0.1 GT/s
    return bytes(blob)


class FakeSysfs(unittest.TestCase):
    """One amdgpu card under a temporary directory, with the attributes a real driver writes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        dev = root / "card1/device"
        dev.mkdir(parents=True)
        (root / "amdgpu").mkdir()                       # stands in for the driver's sysfs dir
        (dev / "driver").symlink_to(root / "amdgpu")
        files = {
            "vendor": "0x1002\n", "device": "0x7550\n", "revision": "0xc0\n",
            "gpu_busy_percent": "37\n", "mem_busy_percent": "9\n", "power_state": "D0\n",
            "mem_info_vram_total": "17095983104\n", "mem_info_vram_used": "1234567\n",
            "mem_info_vram_vendor": "hynix\n",
            "vbios_version": "113-R907XTGOL-F2\n",
            "power_dpm_force_performance_level": "auto\n",
            "pp_dpm_sclk": "0: 500Mhz \n1: 2100Mhz *\n2: 2520Mhz \n",
            "pp_dpm_mclk": "0: 96Mhz \n5: 1258Mhz *\n",
            "pp_features": ("No. Feature            Bit : State\n"
                            "00. FW_DATA_READ        ( 0) : enabled\n"
                            "18. GFXOFF              (18) : enabled\n"
                            "24. SMARTSHIFT          (24) : disabled\n"),
        }
        for name, text in files.items():
            (dev / name).write_text(text)
        self.dev = dev
        self.card = dev.parent
        self._cards = amdgpu.cards
        amdgpu.cards = lambda: [self.card]

    def tearDown(self):
        amdgpu.cards = self._cards
        amdgpu._HWMON_DIR.clear()
        amdgpu._LAYOUT_CHECKED.clear()
        self.tmp.cleanup()

    def test_sample_is_normalized_like_nvml(self):
        rec = amdgpu.sample(self.card, 0)
        self.assertEqual(rec["sm_clock_mhz"], 2100)
        self.assertEqual(rec["mem_clock_mhz"], 1258)
        self.assertEqual(rec["util_gpu_pct"], 37)
        self.assertEqual(rec["pstate"], "D0")
        self.assertEqual(rec["card"], "card1")
        # The keys report/markdown.py reads out of an NVML sample.
        for key in ("sm_clock_mhz", "temp_c", "power_w", "util_gpu_pct"):
            self.assertIn(key, rec)

    def test_dpm_table_and_features(self):
        table = amdgpu.dpm_table(self.dev, "pp_dpm_sclk")
        self.assertEqual(table["states"], {"0": 500.0, "1": 2100.0, "2": 2520.0})
        self.assertEqual(table["active"], "1")
        self.assertEqual(table["forced_level"], "auto")
        self.assertEqual(amdgpu._features(self.dev / "pp_features"),
                         {"FW_DATA_READ": True, "GFXOFF": True, "SMARTSHIFT": False})

    def test_gpu_metrics_decodes_when_it_cross_checks(self):
        (self.dev / "gpu_metrics").write_bytes(metrics_blob())
        m = amdgpu.gpu_metrics(self.dev)
        self.assertNotIn("error", m, m)
        self.assertEqual(m["average_gfxclk_frequency"], 2100)
        self.assertEqual(m["energy_accumulator"], 987654)
        self.assertEqual(m["throttle_status"], 0)
        self.assertTrue(m["cross_check"]["current_gfxclk_in_dpm_table"])
        rec = amdgpu.sample(self.card, 0)
        self.assertEqual(rec["energy_mj"], 987654)
        self.assertEqual(rec["metrics_average_gfx_activity"], 42)

    def test_gpu_metrics_refused_when_it_does_not_cross_check(self):
        # A clock no DPM level can produce: the layout is wrong, not the GPU.
        (self.dev / "gpu_metrics").write_bytes(metrics_blob(gfxclk=41000))
        m = amdgpu.gpu_metrics(self.dev)
        self.assertIn("error", m)
        self.assertFalse(any(m["cross_check"].values()))
        # ... and the sample then carries no decoded extras, rather than wrong ones.
        rec = amdgpu.sample(self.card, 0)
        self.assertNotIn("energy_mj", rec)

    def test_unsupported_fields_are_dropped_not_reported(self):
        blob = bytearray(metrics_blob())
        struct.pack_into("<Q", blob, 24, 0xFFFFFFFFFFFFFFFF)     # no energy counter
        struct.pack_into("<H", blob, 8, 0xFFFF)                  # no memory temperature
        (self.dev / "gpu_metrics").write_bytes(bytes(blob))
        m = amdgpu.gpu_metrics(self.dev)
        self.assertNotIn("energy_accumulator", m)
        self.assertNotIn("temperature_mem", m)
        self.assertIn("temperature_edge", m)

    def test_missing_everything_is_not_an_error(self):
        for f in self.dev.iterdir():
            if f.is_file():
                f.unlink()
        rec = amdgpu.sample(self.card, 0)
        self.assertEqual(rec["sm_clock_mhz"], None)
        self.assertEqual(rec["util_gpu_pct"], None)
        self.assertEqual(amdgpu.gpu_metrics(self.dev), {"error": "No such file or directory"})

    def test_sampler_names_the_source(self):
        import io
        buf = io.StringIO()
        s = Sampler(buf, 1.0)
        s.nvml.available = False
        gpu, source = s.gpu_state()
        self.assertEqual(source, "amdgpu_sysfs")
        self.assertEqual(gpu[0]["sm_clock_mhz"], 2100)

    def test_describe_records_what_would_move_a_number(self):
        (self.dev / "gpu_metrics").write_bytes(metrics_blob())
        d = amdgpu.describe(self.card)
        self.assertEqual(d["device_id"], "0x7550")
        self.assertEqual(d["performance_level"], "auto")
        self.assertEqual(d["overclock_applied"], False)
        self.assertTrue(d["features_enabled"]["GFXOFF"])
        self.assertFalse(d["overclock_applied"])          # the table alone is not an overclock
        self.assertEqual(json.dumps(d["clocks"]["sclk"]["active"]), '"1"')


if __name__ == "__main__":
    unittest.main()
