import unittest

from env.kmsg import parse_kmsg_record


class ParseKmsg(unittest.TestCase):
    def test_xid(self):
        e = parse_kmsg_record("3,1234,5678901,-;NVRM: Xid (PCI:000f:01:00): 31, pid=4242, "
                              "name=memory, Ch 00000008, intr 00000000. MMU Fault\n SUBSYSTEM=pci")
        self.assertEqual(e.category, "xid")
        self.assertEqual(e.xid, 31)
        self.assertEqual(e.device, "PCI:000f:01:00")
        self.assertEqual(e.ts_ns, 5678901000)
        self.assertEqual(e.seq, 1234)
        self.assertNotIn("SUBSYSTEM", e.message)
        rec = e.record()
        self.assertEqual((rec["src"], rec["type"], rec["xid"]), ("kernel", "kernel_event", 31))

    def test_oom_and_hang(self):
        self.assertEqual(parse_kmsg_record(
            "6,1,2,-;python3 invoked oom-killer: gfp_mask=0x140cca").category, "oom")
        self.assertEqual(parse_kmsg_record(
            "3,2,3,-;INFO: task probe:99 blocked for more than 120 seconds.").category, "hang")

    def test_ignored_and_malformed(self):
        self.assertIsNone(parse_kmsg_record("6,1,2,-;usb 1-1: new high-speed USB device"))
        self.assertIsNone(parse_kmsg_record("garbage without separator"))
        self.assertIsNone(parse_kmsg_record("x,y,z,-;NVRM: Xid (PCI:0): 13, detail"))


if __name__ == "__main__":
    unittest.main()
