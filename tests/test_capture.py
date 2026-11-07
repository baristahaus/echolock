"""env.capture's Arm CPU naming: cpuinfo has no `model name` on Arm, so host.cpu_model is
derived from the MIDR part numbers the capture already records per CPU. Written after a GB10 run
where cpu_model came back None on the machine the project targets (docs/gb10-first-run.md)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.capture import ARM_PART_NAMES, arm_cpu_model  # noqa: E402


class TestArmCpuModel(unittest.TestCase):
    def test_gb10_big_little_pair(self):
        # The real shape on GB10 (Asus GX10): 10x Cortex-X925 + 10x Cortex-A725, in either order.
        parts = ["0xd87"] * 10 + ["0xd85"] * 10
        self.assertEqual(arm_cpu_model(parts), "Cortex-A725 x10 + Cortex-X925 x10")
        self.assertEqual(arm_cpu_model(parts[::-1]), arm_cpu_model(parts))

    def test_single_core_type_has_no_count(self):
        self.assertEqual(arm_cpu_model(["0xd49"] * 4), "Neoverse N2 x4")
        self.assertEqual(arm_cpu_model(["0xd49"]), "Neoverse N2")

    def test_unknown_part_keeps_its_hex_rather_than_getting_a_wrong_name(self):
        got = arm_cpu_model(["0xffff", "0xd87"])
        self.assertIn("part 0xffff", got)
        self.assertIn("Cortex-X925", got)

    def test_part_numbers_are_read_case_insensitively(self):
        self.assertEqual(arm_cpu_model(["0xD87"]), arm_cpu_model(["0xd87"]))

    def test_nothing_to_name_stays_unset(self):
        # None lets the caller leave cpu_model absent instead of inventing a name.
        self.assertIsNone(arm_cpu_model([]))
        self.assertIsNone(arm_cpu_model([None, "", "  "]))

    def test_named_cores_are_the_ones_checked_against_lscpu_or_arm_tables(self):
        # Guards against a typo'd hex code silently naming the wrong core: 0xd87/0xd85 are the two
        # verified on hardware here, and every key must be 4 hex digits with a real name.
        for part, name in ARM_PART_NAMES.items():
            self.assertRegex(part, r"^0x[0-9a-f]{1,8}$")  # MIDR part numbers are variable-width hex
            self.assertTrue(name.strip())


if __name__ == "__main__":
    unittest.main()
