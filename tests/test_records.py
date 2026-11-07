"""record-format v1 validator: a synthetic valid corpus passes, corrupted records fail, and
every real run directory under runs/ validates."""

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

from report import records  # noqa: E402


def measure(name="gpu_read", tags=None, **over):
    r = {"v": 1, "t_ns": 1000, "src": "probe", "probe": "memory", "backend": "vulkan",
         "type": "measure", "name": name, "value": 1.0, "unit": "GB/s",
         "tags": tags if tags is not None else {"source": "vulkan_device_local"},
         "verified": True}
    r.update(over)
    return r


def probe_stream(status="ok", failed=False):
    recs = [
        {"v": 1, "t_ns": 1, "src": "runner", "type": "step_start", "step": "memory",
         "argv": ["x"], "drop_caches": None},
        {"v": 1, "t_ns": 2, "src": "probe", "probe": "memory", "backend": "vulkan",
         "type": "probe_start", "params": {"bytes": 1}},
        {"v": 1, "t_ns": 3, "src": "probe", "probe": "memory", "backend": "vulkan",
         "type": "declared", "key": "device.name", "value": "AMD Test", "source": "fake"},
        measure(rep=0), measure(rep=1),
        measure(tags={"source": "vulkan_device_local",
                      "per_rep": {"min": 1.0, "p50": 2.0, "max": 3.0}}),
        {"v": 1, "t_ns": 7, "src": "probe", "probe": "memory", "backend": "vulkan",
         "type": "check", "name": "gpu_read.xor", "pass": not failed, "detail": {}},
        {"v": 1, "t_ns": 8, "src": "probe", "probe": "memory", "backend": "vulkan",
         "type": "note", "message": "hello"},
        {"v": 1, "t_ns": 9, "src": "probe", "probe": "memory", "backend": "vulkan",
         "type": "feature", "feature": "x", "requested": True, "granted": True,
         "effect_observed": True, "evidence": {}},
        {"v": 1, "t_ns": 10, "src": "runner", "type": "step_end", "step": "memory",
         "status": "ok", "returncode": 0, "duration_s": 1.0, "timed_out": False,
         "unkillable": False, "interrupted": False},
        {"v": 1, "t_ns": 11, "src": "probe", "probe": "memory", "backend": "vulkan",
         "type": "probe_end", "status": "verify_failed" if failed else status,
         "failed_checks": 1 if failed else 0},
    ]
    return recs


def make_run_dir(root: Path, stream=None, manifest_status=None, name="r1"):
    d = root / f"20260101T000000Z-memory-vulkan-{name}"
    d.mkdir(parents=True, exist_ok=True)
    status = manifest_status or "ok"
    (d / "manifest.json").write_text(json.dumps({
        "v": 1, "run_id": d.name, "probe": "memory", "backend": "vulkan", "label": name,
        "status": status, "started_utc": "2026-01-01T00:00:00Z",
        "steps": [{"name": "memory", "status": "ok", "returncode": 0, "failed_checks": 0}],
    }))
    (d / "probe.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (stream if stream is not None else probe_stream())))
    (d / "env.json").write_text(json.dumps({"v": 1, "t_ns": 1, "host": {"hostname": "x"}}))
    (d / "samples.jsonl").write_text(json.dumps(
        {"v": 1, "t_ns": 1, "src": "sampler", "type": "sample", "gpu": {}}) + "\n")
    (d / "kernel.jsonl").write_text(json.dumps(
        {"v": 1, "t_ns": 1, "src": "kernel", "type": "kernel_event", "category": "gpu",
         "priority": 3, "seq": 1, "message": "ok"}) + "\n")
    (d / "load.jsonl").write_text(json.dumps(
        {"v": 1, "t_ns": 1, "src": "load", "type": "load_start", "load": "cpu"}) + "\n")
    return d


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_run_dir_passes(self):
        d = make_run_dir(self.root)
        self.assertEqual(records.validate_run_dir(d), [])

    def test_missing_t_ns_fails(self):
        stream = probe_stream()
        del stream[1]["t_ns"]
        d = make_run_dir(self.root, stream=stream)
        self.assertTrue(any("t_ns" in e for e in records.validate_run_dir(d)))

    def test_v2_fails(self):
        stream = probe_stream()
        stream[1]["v"] = 2
        d = make_run_dir(self.root, stream=stream)
        self.assertTrue(any("v is 2" in e for e in records.validate_run_dir(d)))

    def test_unknown_type_fails(self):
        stream = probe_stream()
        stream[3]["type"] = "vibes"
        d = make_run_dir(self.root, stream=stream)
        self.assertTrue(any("unknown" in e for e in records.validate_run_dir(d)))

    def test_probe_end_before_probe_start_fails(self):
        stream = probe_stream()
        stream[1], stream[-1] = stream[-1], stream[1]
        d = make_run_dir(self.root, stream=stream)
        errs = records.validate_run_dir(d)
        self.assertTrue(any("expected probe_start" in e for e in errs))

    def test_records_after_probe_end_fails(self):
        stream = probe_stream()
        stream.append(measure())
        d = make_run_dir(self.root, stream=stream)
        self.assertTrue(any("expected probe_end" in e for e in records.validate_run_dir(d)))

    def test_status_inconsistent_with_checks_fails(self):
        stream = probe_stream(failed=True)
        stream[-1]["status"] = "ok"  # lies about the failed check
        d = make_run_dir(self.root, stream=stream, manifest_status="verify_failed")
        errs = records.validate_run_dir(d)
        self.assertTrue(any("probe_end status" in e for e in errs))

    def test_failed_checks_count_mismatch_fails(self):
        stream = probe_stream()
        stream[-1]["failed_checks"] = 3
        d = make_run_dir(self.root, stream=stream)
        self.assertTrue(any("failed_checks" in e for e in records.validate_run_dir(d)))

    def test_manifest_status_worse_than_steps_fails(self):
        d = make_run_dir(self.root, manifest_status="error")
        self.assertTrue(any("worst" in e for e in records.validate_run_dir(d)))

    def test_crosscheck_record_type_accepted(self):
        rec = {"v": 1, "t_ns": 1, "src": "crosscheck", "type": "crosscheck",
               "probe": "memory", "name": "memory.gpu_read.device_local",
               "a": {"run_id": "a", "value": 1.0}, "b": {"run_id": "b", "value": 1.1},
               "verdict": "within-tolerance", "tolerance": 0.1, "detail": "delta 0.1"}
        self.assertEqual(records.validate_record(rec), [])
        rec["verdict"] = "wobbly"
        self.assertTrue(records.validate_record(rec))

    @unittest.skipUnless(RUNS.is_dir(), "no runs/ corpus on this machine")
    def test_every_real_run_dir_validates(self):
        results = records.validate_all(RUNS)
        self.assertTrue(results, "no run directories found under runs/")
        for name, errs in results.items():
            self.assertEqual(errs, [], f"{name}: {errs}")


if __name__ == "__main__":
    unittest.main()
