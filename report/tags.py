"""Canonical metric mapping: (probe, backend, name, tags) -> one canonical key.

This is the row alignment docs/crosscheck.md left open after phase 1: the same quantity measured
by two backends comes out under different names and tag spellings, so it gets one key here.

The table is explicit, per probe x backend, built from the metric names in the probe sources
(backends/cuda/*.cu, backends/vulkan/*.cpp) and the records under runs/:

- memory
    cuda:    gpu_read / copy_payload with tags.source in {cudaMalloc, cudaHostAlloc_mapped,
              malloc_pageable, vmm_*}, copy direction h2d
    vulkan:  the same names with tags.source in {vulkan_device_local, vulkan_host_visible}
    aligned on the semantic source kind: device_local / host_visible / pageable / vmm.
- pointer_chase
    cuda:    chase_hop_latency with tags.kind in {device, pinned, pageable, file}
    vulkan:  chase_hop_latency with tags.kind in {vulkan_device_local, vulkan_host_visible}
    aligned on kind, with footprint (MiB) and pattern in the key.
- mma
    cuda:    mma_throughput with tags.format (+ tags.shape)
    vulkan:  <format>_tflops (one measure name per format)
    aligned as mma.tflops.<format>; both backends run one shape per format.
- sync
    cuda:    sync_return_lag with tags {schedule, kernel_us}
    vulkan:  sync_<dur>us.lag (one measure name per duration; no schedule axis on a queue
              submission, so the key's schedule slot is "queue")
- clock, file_page, constants
    the backends measure different things here (shader clock vs SM clock; file pages vs
    anonymous rows), so there is no alignment: the key is <probe>.<name> and each backend's
    rows stand alone.

Anything not in the table maps to None: the diff and the crosscheck report it as unmapped
rather than guessing an alignment.
"""

from __future__ import annotations

# memory source kind, per backend spelling, to the semantic kind
_MEMORY_SOURCE = {
    "cudaMalloc": "device_local",
    "vulkan_device_local": "device_local",
    "cudaHostAlloc_mapped": "host_visible",
    "vulkan_host_visible": "host_visible",
    "malloc_pageable": "pageable",
    "vmm_compression_requested": "vmm",
    "vmm_plain": "vmm",
}

# pointer_chase memory kind, per backend spelling, to the semantic kind
_CHASE_KIND = {
    "device": "device_local",
    "vulkan_device_local": "device_local",
    "pinned": "pinned",
    "vulkan_host_visible": "host_visible",
    "pageable": "pageable",
    "file": "file",
}

# Probes whose metrics are probe-specific (no cross-backend alignment): the key keeps the
# raw name, so a metric is identified by probe + name alone.
_PASS_THROUGH_PROBES = ("clock", "file_page", "constants")

_PASS_THROUGH = {
    "memory": ("cpu_read", "gpu_read_contention_ratio", "shared_sum", "compression_gain"),
    "pointer_chase": ("cold_fault_hop", "cold_fault_cost", "l2_vs_dram_delta"),
    "sync": ("graph_launch_call", "graph_launch_to_first_kernel"),
}


def _mib(v: object) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "?"


def canonical_key(probe: str, backend: str, name: str, tags: dict | None = None) -> str | None:
    """One canonical key for a measure series, or None when the table has no row for it."""
    tags = tags or {}
    if probe in _PASS_THROUGH_PROBES:
        return f"{probe}.{name}"
    if probe in _PASS_THROUGH and name in _PASS_THROUGH[probe]:
        if name == "cpu_read":
            cont = str(tags.get("contention", "none"))
            return f"memory.cpu_read" + ("" if cont == "none" else f".{cont}")
        return f"memory.{name}" if probe == "memory" else f"{probe}.{name}"

    if probe == "memory":
        if name == "gpu_read":
            src = _MEMORY_SOURCE.get(str(tags.get("source")))
            if src is None:
                return None
            cont = str(tags.get("contention", "none"))
            return f"memory.gpu_read.{src}" + ("" if cont == "none" else f".{cont}")
        if name == "copy_payload":
            src = _MEMORY_SOURCE.get(str(tags.get("source")))
            if src is None:
                return None
            return f"memory.copy.{tags.get('direction', 'h2d')}.{src}"
        return None

    if probe == "pointer_chase":
        if name == "chase_hop_latency":
            kind = _CHASE_KIND.get(str(tags.get("kind")))
            if kind is None:
                return None
            return (f"pointer_chase.hop.{kind}.{_mib(tags.get('footprint_mib'))}"
                    f".{tags.get('pattern', '?')}")
        return None

    if probe == "mma":
        if name == "mma_throughput":  # cuda: one series per (format, shape)
            fmt = tags.get("format")
            return f"mma.tflops.{fmt}" if fmt else None
        if name.endswith("_tflops"):  # vulkan: one measure name per format
            return f"mma.tflops.{name[:-len('_tflops')]}"
        if name == "format_ratio":
            fmt = tags.get("format")
            return f"mma.format_ratio.{fmt}" if fmt else None
        return None

    if probe == "sync":
        if name == "sync_return_lag":  # cuda: schedule x wait duration
            sched, us = tags.get("schedule"), tags.get("kernel_us")
            return f"sync.lag.{sched}.{us}us" if sched is not None and us is not None else None
        if name == "sync_wait_cpu_fraction":
            return f"sync.wait_cpu_fraction.{tags.get('schedule')}"
        if backend == "vulkan" and name.startswith("sync_") and name.endswith(".lag"):
            return f"sync.lag.queue.{name[len('sync_'):-len('.lag')]}"
        return None

    return None


def run_metrics(run: "object") -> dict[str, dict]:
    """canonical key -> summary entry {value, unit, verified, name, tags} for one run.

    `run` needs .of_type("measure"). Summary records (the one without "rep") win; a series
    with only per-rep records falls back to their median, like report/markdown.py does.
    """
    import statistics

    per: dict[str, list[dict]] = {}
    out: dict[str, dict] = {}
    for r in run.of_type("measure"):
        key = canonical_key(r.get("probe", ""), r.get("backend", ""), r.get("name", ""),
                            r.get("tags"))
        if key is None:
            continue
        tags = {k: v for k, v in (r.get("tags") or {}).items() if k not in ("per_rep", "reps")}
        if "rep" in r:
            per.setdefault(key, []).append(r)
        elif key not in out:
            out[key] = {"value": r.get("value"), "unit": r.get("unit", ""),
                        "verified": r.get("verified"), "name": r.get("name"), "tags": tags}
    for key, reps in per.items():
        if key in out:
            continue
        vals = [x["value"] for x in reps if isinstance(x.get("value"), (int, float))
                and not isinstance(x.get("value"), bool)]
        out[key] = {"value": statistics.median(vals) if vals else None,
                    "unit": reps[0].get("unit", ""),
                    "verified": all(x.get("verified") for x in reps),
                    "name": reps[0].get("name"), "tags": dict(reps[0].get("tags") or {})}
    return out


def unmapped_metrics(run: "object") -> list[str]:
    """Measure series in a run that the table has no row for, as 'name (tags)' strings."""
    seen: set[tuple] = set()
    out = []
    for r in run.of_type("measure"):
        tags = {k: v for k, v in (r.get("tags") or {}).items() if k not in ("per_rep", "reps")}
        ident = (r.get("name"), tuple(sorted(tags.items())))
        if ident in seen or canonical_key(r.get("probe", ""), r.get("backend", ""),
                                          r.get("name", ""), r.get("tags")) is not None:
            continue
        seen.add(ident)
        extra = ", ".join(f"{k}={v}" for k, v in sorted(tags.items()))
        out.append(f"{r.get('name')}" + (f" ({extra})" if extra else ""))
    return sorted(out)
