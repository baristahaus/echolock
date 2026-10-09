import errno
import io
import json
import unittest
from unittest import mock

from env.kmsg import KernelLogWatcher, parse_kmsg_record, parse_seq


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

    def test_amdgpu_and_reset(self):
        # The two AMD forms that matter: an error/hang dump, and work the driver threw away.
        self.assertEqual(parse_kmsg_record(
            "3,7,99,-;amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, ring ready to run").category,
            "gpu")
        self.assertEqual(parse_kmsg_record(
            "3,8,100,-;amdgpu 0000:03:00.0: amdgpu: MODE: (256) 0x1 0x0 0x0").category, "gpu")
        self.assertEqual(parse_kmsg_record(
            "3,9,101,-;amdgpu 0000:03:00.0: gpu reset begin, mask:0x3").category, "reset")
        self.assertEqual(parse_kmsg_record(
            "3,10,102,-;NVRM: gpu at 0000:01:00.0, pci:0000:01:00.0: GPU has fallen off the bus.").category,
            "nvrm")

    def test_ignored_and_malformed(self):
        self.assertIsNone(parse_kmsg_record("6,1,2,-;usb 1-1: new high-speed USB device"))
        self.assertIsNone(parse_kmsg_record("garbage without separator"))
        self.assertIsNone(parse_kmsg_record("x,y,z,-;NVRM: Xid (PCI:0): 13, detail"))


class SeqTracking(unittest.TestCase):
    """A jump in the kernel's sequence numbers means the ring lost messages before the watcher
    reached them. A report missing events must say so, not look complete."""

    def test_parse_seq(self):
        self.assertEqual(parse_seq("3,1234,5678901,-;NVRM: Xid (PCI:0): 13, detail"), 1234)
        self.assertEqual(parse_seq("6,7,8,-;unmatched message"), 7)
        self.assertIsNone(parse_seq("garbage"))
        self.assertIsNone(parse_seq("3,notanumber,5,-;x"))

    def test_gaps_counted_and_recorded(self):
        out = io.StringIO()
        w = KernelLogWatcher(out=out)
        w._track_seq("3,10,20,-;x", history=True)
        self.assertEqual(w.lost_records, 0)
        w._track_seq("3,12,21,-;x", history=True)   # one message the ring lost before us
        self.assertEqual(w.lost_records, 1)
        self.assertEqual(out.getvalue(), "")           # boot history is not this run's loss
        w._lost_before, w._overruns_before = w.lost_records, w.overruns
        w._track_seq("3,15,30,-;x", history=False)    # two lost while the run was watched
        self.assertEqual(w.lost_records - w._lost_before, 2)
        gap = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual((gap["type"], gap["src"], gap["lost"], gap["between_seq"]),
                         ("kernel_event_gap", "kernel", 2, [12, 15]))
        self.assertEqual((w.summary()["lost_records"], w.summary()["overruns"]), (2, 0))

    def test_epipe_counts_overrun(self):
        w = KernelLogWatcher(out=io.StringIO())
        w._fd = 0
        with mock.patch("env.kmsg.os.read",
                        side_effect=[OSError(errno.EPIPE, "ring overran"), BlockingIOError()]):
            w._drain(history=False)
        self.assertEqual(w.overruns, 1)
        self.assertIsNone(w.error)


if __name__ == "__main__":
    unittest.main()
