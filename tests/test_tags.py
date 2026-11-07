"""canonical tag mapping: the explicit table covers every metric the backends emit (both the
synthetic fixtures built from the probe sources and every probe.jsonl under runs/), and the
rows actually align the two backends."""

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

from report import tags  # noqa: E402


# Every metric name each backend emits, from the probe sources:
#   backends/cuda/*.cu: gpu_read/copy_payload/cpu_read/gpu_read_contention_ratio/shared_sum/
#     compression_gain, chase_hop_latency/cold_fault_hop/cold_fault_cost/l2_vs_dram_delta,
#     mma_throughput/format_ratio, sync_return_lag/sync_wait_cpu_fraction/graph_launch_*,
#     derived_sm_clock_mhz/clock_drop_ratio/declared_over_derived_ratio/globaltimer_vs_host_ms,
#     cold_fault_throughput/evict_duration/evictor_read/gpu_seq_read/gpu_seq_read_wall/
#     host_warm_read/page_cache_resident/prep_write
#   backends/vulkan/*.cpp: the memory/pointer_chase names plus vulkan_device_local /
#     vulkan_host_visible kinds, <format>_tflops, sync_<dur>us.lag, idle_clock/loaded_clock/
#     idle_submit_overhead/loaded_submit_overhead/shader_clock_over_sclk, cold_rows/res_rows/
#     res_rows_again/res_seq
FIXTURES = [
    # memory
    ("memory", "cuda", "gpu_read", {"source": "cudaMalloc"}),
    ("memory", "cuda", "gpu_read", {"source": "cudaHostAlloc_mapped", "contention": "cpu_read"}),
    ("memory", "cuda", "gpu_read", {"source": "malloc_pageable", "fill": "random"}),
    ("memory", "cuda", "gpu_read", {"source": "vmm_compression_requested"}),
    ("memory", "cuda", "gpu_read", {"source": "vmm_plain"}),
    ("memory", "cuda", "copy_payload", {"source": "cudaHostAlloc_mapped", "direction": "h2d"}),
    ("memory", "cuda", "copy_payload", {"source": "malloc_pageable", "direction": "h2d"}),
    ("memory", "cuda", "cpu_read", {"source": "malloc_pageable", "threads": 8}),
    ("memory", "cuda", "cpu_read", {"source": "malloc_pageable", "contention": "gpu_read"}),
    ("memory", "cuda", "gpu_read_contention_ratio", {}),
    ("memory", "cuda", "shared_sum", {"note": "GPU read + CPU read while contending"}),
    ("memory", "cuda", "compression_gain", {}),
    ("memory", "vulkan", "gpu_read", {"source": "vulkan_device_local",
                                      "heap_device_local": True, "heap_host_visible": True}),
    ("memory", "vulkan", "gpu_read", {"source": "vulkan_host_visible", "contention": "cpu_read"}),
    ("memory", "vulkan", "copy_payload", {"source": "vulkan_host_visible", "direction": "h2d"}),
    ("memory", "vulkan", "cpu_read", {"source": "vulkan_host_visible", "threads": 8}),
    ("memory", "vulkan", "gpu_read_contention_ratio", {}),
    ("memory", "vulkan", "shared_sum", {}),
    # pointer_chase
    ("pointer_chase", "cuda", "chase_hop_latency",
     {"kind": "device", "footprint_mib": 64, "pattern": "random", "hops": 4096}),
    ("pointer_chase", "cuda", "chase_hop_latency",
     {"kind": "pinned", "footprint_mib": 1, "pattern": "seq"}),
    ("pointer_chase", "cuda", "chase_hop_latency",
     {"kind": "pageable", "footprint_mib": 1, "pattern": "seq"}),
    ("pointer_chase", "cuda", "chase_hop_latency",
     {"kind": "file", "footprint_mib": 0.0625, "pattern": "random"}),
    ("pointer_chase", "cuda", "cold_fault_hop", {"hops": 4096}),
    ("pointer_chase", "cuda", "cold_fault_cost", {"pages": 100}),
    ("pointer_chase", "cuda", "l2_vs_dram_delta", {}),
    ("pointer_chase", "vulkan", "chase_hop_latency",
     {"kind": "vulkan_device_local", "footprint_mib": 64, "pattern": "random"}),
    ("pointer_chase", "vulkan", "chase_hop_latency",
     {"kind": "vulkan_host_visible", "footprint_mib": 0.0625, "pattern": "seq"}),
    # mma
    ("mma", "cuda", "mma_throughput", {"format": "fp16", "shape": "m16n16k16"}),
    ("mma", "cuda", "mma_throughput", {"format": "bf16", "shape": "m16n16k16"}),
    ("mma", "cuda", "mma_throughput", {"format": "tf32", "shape": "m16n8k16"}),
    ("mma", "cuda", "mma_throughput", {"format": "int8", "shape": "m16n16k8"}),
    ("mma", "cuda", "mma_throughput", {"format": "fp8", "shape": "m16n8k32"}),
    ("mma", "cuda", "format_ratio", {"format": "fp8"}),
    ("mma", "vulkan", "fp16_tflops", {"iters": 1}),
    ("mma", "vulkan", "bf16_tflops", {"iters": 1}),
    ("mma", "vulkan", "int8_tflops", {"iters": 1}),
    # sync
    ("sync", "cuda", "sync_return_lag", {"schedule": "blocking", "kernel_us": 1000}),
    ("sync", "cuda", "sync_return_lag", {"schedule": "spin", "kernel_us": 50}),
    ("sync", "cuda", "sync_wait_cpu_fraction", {"schedule": "yield"}),
    ("sync", "cuda", "graph_launch_call", {"schedule": "auto", "nodes": 4}),
    ("sync", "cuda", "graph_launch_to_first_kernel", {"schedule": "auto", "nodes": 4}),
    ("sync", "vulkan", "sync_50us.lag", {}),
    ("sync", "vulkan", "sync_1000us.lag", {}),
    ("sync", "vulkan", "sync_10000us.lag", {}),
    ("sync", "vulkan", "sync_60000us.lag", {}),
    # clock
    ("clock", "cuda", "derived_sm_clock_mhz", {}),
    ("clock", "cuda", "clock_drop_ratio", {}),
    ("clock", "cuda", "declared_over_derived_ratio", {}),
    ("clock", "cuda", "globaltimer_vs_host_ms", {}),
    ("clock", "vulkan", "idle_clock", {}),
    ("clock", "vulkan", "loaded_clock", {}),
    ("clock", "vulkan", "idle_submit_overhead", {}),
    ("clock", "vulkan", "loaded_submit_overhead", {}),
    ("clock", "vulkan", "shader_clock_over_sclk", {}),
    # file_page
    ("file_page", "cuda", "cold_fault_throughput", {}),
    ("file_page", "cuda", "evict_duration", {}),
    ("file_page", "cuda", "evictor_read", {}),
    ("file_page", "cuda", "gpu_seq_read", {}),
    ("file_page", "cuda", "gpu_seq_read_wall", {}),
    ("file_page", "cuda", "host_warm_read", {}),
    ("file_page", "cuda", "page_cache_resident", {}),
    ("file_page", "cuda", "prep_write", {}),
    ("file_page", "vulkan", "cold_rows", {}),
    ("file_page", "vulkan", "res_rows", {}),
    ("file_page", "vulkan", "res_rows_again", {}),
    ("file_page", "vulkan", "res_seq", {}),
    # constants (new probe: declared facts + checks, one measure per derived fact)
    ("constants", "cuda", "l2_bytes", {"source": "properties"}),
    ("constants", "vulkan", "l2_bytes", {"source": "limits"}),
]


class TagTest(unittest.TestCase):
    def test_every_fixture_maps_to_exactly_one_key(self):
        for probe, backend, name, ts in FIXTURES:
            with self.subTest(probe=probe, backend=backend, name=name):
                key = tags.canonical_key(probe, backend, name, ts)
                self.assertIsNotNone(key, f"no table row for {probe}/{backend}/{name} {ts}")
                self.assertEqual(key, tags.canonical_key(probe, backend, name, dict(ts)),
                                 "mapping is not a pure function of (probe, backend, name, tags)")

    def test_rows_align_the_backends(self):
        # memory
        self.assertEqual(
            tags.canonical_key("memory", "cuda", "gpu_read", {"source": "cudaMalloc"}),
            tags.canonical_key("memory", "vulkan", "gpu_read", {"source": "vulkan_device_local"}))
        self.assertEqual(
            tags.canonical_key("memory", "cuda", "copy_payload",
                               {"source": "cudaHostAlloc_mapped", "direction": "h2d"}),
            tags.canonical_key("memory", "vulkan", "copy_payload",
                               {"source": "vulkan_host_visible", "direction": "h2d"}))
        # pointer_chase
        self.assertEqual(
            tags.canonical_key("pointer_chase", "cuda", "chase_hop_latency",
                               {"kind": "device", "footprint_mib": 64, "pattern": "random"}),
            tags.canonical_key("pointer_chase", "vulkan", "chase_hop_latency",
                               {"kind": "vulkan_device_local", "footprint_mib": 64,
                                "pattern": "random"}))
        # mma
        self.assertEqual(
            tags.canonical_key("mma", "cuda", "mma_throughput",
                               {"format": "fp16", "shape": "m16n16k16"}),
            tags.canonical_key("mma", "vulkan", "fp16_tflops", {}))
        self.assertEqual(tags.canonical_key("mma", "cuda", "mma_throughput",
                                            {"format": "fp16", "shape": "m16n16k16"}),
                         "mma.tflops.fp16")

    def test_keys_are_stable_strings(self):
        self.assertEqual(tags.canonical_key("memory", "cuda", "gpu_read",
                                            {"source": "cudaMalloc"}),
                         "memory.gpu_read.device_local")
        self.assertEqual(tags.canonical_key("memory", "cuda", "gpu_read",
                                            {"source": "cudaHostAlloc_mapped"}),
                         "memory.gpu_read.host_visible")
        self.assertEqual(tags.canonical_key("sync", "vulkan", "sync_1000us.lag", {}),
                         "sync.lag.queue.1000us")
        self.assertEqual(tags.canonical_key("sync", "cuda", "sync_return_lag",
                                            {"schedule": "auto", "kernel_us": 1000}),
                         "sync.lag.auto.1000us")

    @unittest.skipUnless(RUNS.is_dir(), "no runs/ corpus on this machine")
    def test_every_real_measure_maps_to_exactly_one_key(self):
        checked = 0
        for f in sorted(RUNS.glob("*/probe.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("type") != "measure":
                    continue
                checked += 1
                key = tags.canonical_key(r.get("probe", ""), r.get("backend", ""),
                                         r.get("name", ""), r.get("tags"))
                self.assertIsNotNone(key, f"{f.parent.name}: unmapped {r.get('name')} "
                                          f"{r.get('tags')}")
        self.assertGreater(checked, 0, "no measure records found under runs/")

    def test_run_metrics_prefers_the_summary_record(self):
        class Run:
            def __init__(self, recs):
                self.recs = recs

            def of_type(self, t):
                return [r for r in self.recs if r.get("type") == t]

        recs = [
            {"type": "measure", "name": "gpu_read", "value": 1.0, "unit": "GB/s",
             "tags": {"source": "vulkan_device_local", "gpu_us": 400}, "verified": True,
             "rep": 0, "probe": "memory", "backend": "vulkan"},
            {"type": "measure", "name": "gpu_read", "value": 2.0, "unit": "GB/s",
             "tags": {"source": "vulkan_device_local", "gpu_us": 401}, "verified": True,
             "rep": 1, "probe": "memory", "backend": "vulkan"},
            {"type": "measure", "name": "gpu_read", "value": 634.3, "unit": "GB/s",
             "tags": {"source": "vulkan_device_local", "per_rep": {"min": 1.0, "max": 2.0}},
             "verified": True, "probe": "memory", "backend": "vulkan"},
            {"type": "measure", "name": "l2_vs_dram_delta", "value": 9.0, "unit": "ns",
             "tags": {"x": 1}, "verified": True, "rep": 0, "probe": "pointer_chase",
             "backend": "vulkan"},
            {"type": "measure", "name": "l2_vs_dram_delta", "value": 11.0, "unit": "ns",
             "tags": {"x": 1}, "verified": True, "rep": 1, "probe": "pointer_chase",
             "backend": "vulkan"},
        ]
        m = tags.run_metrics(Run(recs))
        self.assertEqual(m["memory.gpu_read.device_local"]["value"], 634.3)
        self.assertEqual(m["pointer_chase.l2_vs_dram_delta"]["value"], 10.0)  # median
        self.assertEqual(tags.unmapped_metrics(Run(recs)), [])

    def test_unmapped_metrics_reports_the_gap(self):
        class Run:
            def of_type(self, t):
                if t != "measure":
                    return []
                return [{"type": "measure", "name": "gpu_read", "value": 1.0, "unit": "GB/s",
                         "tags": {"source": "mystery_heap"}, "verified": True,
                         "probe": "memory", "backend": "vulkan"}]

        self.assertEqual(tags.unmapped_metrics(Run()), ["gpu_read (source=mystery_heap)"])


if __name__ == "__main__":
    unittest.main()
