"""Markdown report for one or more run directories.

Runs of the same probe taken under different loads sit side by side: one column per run label
(idle, 50% load, 95% fill ...). Declared values come first, then measurements, then everything
that qualifies them: failed checks, silent fallbacks, what the clocks, temperature and power did
during the run, and kernel events.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def load_json(path: Path) -> tuple[dict, str | None]:
    """Reads a JSON object, or says why it could not. A run whose records survived but whose
    manifest did not is worth reading, so a broken one is reported, not fatal."""
    try:
        text = path.read_text()
    except OSError as e:
        return {}, f"{path.name}: {e.strerror}"
    try:
        obj = json.loads(text)
    except ValueError as e:
        return {}, f"{path.name}: {e}"
    if not isinstance(obj, dict):
        return {}, f"{path.name}: holds a {type(obj).__name__}, not an object"
    return obj, None


class RunData:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.manifest, problem = load_json(run_dir / "manifest.json")
        env_path = run_dir / "env.json"
        self.env = (load_json(env_path)[0] if env_path.is_file() else {})
        self.records = load_jsonl(run_dir / "probe.jsonl")
        self.samples = load_jsonl(run_dir / "samples.jsonl")
        self.kernel = load_jsonl(run_dir / "kernel.jsonl")
        if not run_dir.is_dir():
            problem = "not a directory"
        elif not self.records and problem is None:
            problem = "no probe records: the run was interrupted before any probe wrote one"
        self.problem = problem

    @property
    def label(self) -> str:
        return self.manifest.get("label", self.dir.name)

    @property
    def column(self) -> str:
        """Column key in a side-by-side report. Runs from different backends of the same probe
        get one column each, so a CUDA run and a Vulkan run sit side by side even when they share
        a load-condition label (crosscheck mode)."""
        backend = self.manifest.get("backend")
        return f"{self.label} [{backend}]" if backend else self.label

    def of_type(self, t: str) -> list[dict]:
        return [r for r in self.records if r.get("type") == t]


def fmt(v: object, unit: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        a = abs(v)
        s = f"{v:.0f}" if a >= 1000 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 0.1 else f"{v:.3g}"
    else:
        s = str(v)
    return f"{s} {unit}".strip()


def tags_text(tags: dict) -> str:
    skip = {"per_rep", "reps"}
    return ", ".join(f"{k}={v}" for k, v in tags.items() if k not in skip) or "—"


def measurement_rows(run: RunData) -> dict[tuple[str, str, str], dict]:
    """(step, name, tags) -> summary value, unit, verified, and a per-rep spread.

    Series with a summary record (no "rep") use it; series with only per-rep records get a
    median. Per-rep records without a summary are summarized here.
    """
    summaries: dict[tuple[str, str, str], dict] = {}
    reps: dict[tuple[str, str, str], list[dict]] = {}
    for r in run.of_type("measure"):
        tags = dict(r.get("tags") or {})
        per_rep = tags.pop("per_rep", None)
        tags.pop("reps", None)
        tags.pop("calibration_rtt_us", None)
        key = (r.get("step", ""), r["name"], tags_text(tags))
        if "rep" in r:
            reps.setdefault(key, []).append(r)
        else:
            summaries[key] = {"value": r["value"], "unit": r.get("unit", ""),
                              "verified": r.get("verified"), "per_rep": per_rep}
    for key, rs in reps.items():
        values = [x["value"] for x in rs if isinstance(x.get("value"), (int, float))]
        entry = summaries.setdefault(key, {
            "value": statistics.median(values) if values else None,
            "unit": rs[0].get("unit", ""), "verified": all(x.get("verified") for x in rs),
            "per_rep": None})
        entry["n"] = len(rs)
        if values and entry.get("per_rep") is None:
            entry["per_rep"] = {"min": min(values), "max": max(values)}
        entry["verified"] = entry["verified"] and all(x.get("verified") for x in rs)
    return summaries


def fmt_bytes(v: object) -> str:
    return f"{v / 2 ** 30:.0f} GiB" if isinstance(v, (int, float)) and v else "—"


def env_lines(run: RunData) -> list[str]:
    env = run.env
    host = env.get("host", {})
    nv = env.get("nvidia", {})
    smi = (nv.get("smi") or {})
    gpus = smi.get("gpus") or []
    amd = env.get("amd", {})
    fw = env.get("firmware", {}).get("dmi", {})
    driver = (smi.get("driver_version") or (nv.get("nvml") or {}).get("driver_version")
              or amd.get("driver_version") or "—")
    built = (env.get("toolchain", {}).get("build") or {}).get("arch")
    lines = [
        f"- Host: {host.get('os', {}).get('pretty_name', '?')}, kernel {host.get('kernel', '?')}, "
        f"{host.get('machine', '?')}, {len(host.get('cpus', []))} CPUs",
        f"- Driver: {driver}; CUDA (driver) {smi.get('cuda_version') or '—'}; "
        f"nvcc: {env.get('toolchain', {}).get('nvcc') or '—'}; "
        f"probes built for {built or '—'}",
    ]
    for g in gpus:
        lines.append(f"- GPU: {g.get('product_name')} ({g.get('pci_device_id')}), "
                     f"VBIOS {g.get('vbios_version')}, GSP firmware {g.get('gsp_firmware_version')}")
    for c in amd.get("cards") or []:
        sclk = (c.get("clocks") or {}).get("sclk") or {}
        vram = f"{fmt_bytes(c.get('vram_total_bytes'))} {c.get('vram_vendor') or ''}".strip()
        lines.append(f"- AMD GPU: {c.get('vendor_id')}:{c.get('device_id')} rev "
                     f"{c.get('revision')} at {c.get('pci')}, {vram}, VBIOS "
                     f"{c.get('vbios_version')}, driver {c.get('driver')}")
        # What could move a measurement without anything asking it to.
        lines.append(f"  - sclk {sclk.get('states')} MHz, active level {sclk.get('active')}, "
                     f"performance level `{sclk.get('forced_level') or c.get('performance_level')}`, "
                     f"overclock offsets: "
                     f"{'applied' if c.get('overclock_applied') else 'none'}")
    kfd = amd.get("kfd_nodes") or []
    if kfd:
        lines.append("- KFD nodes: " + ", ".join(
            f"{n['node']}: {n.get('simd_count', 0) // 4} CU, "
            + ", ".join(fmt_bytes(b.get("size_bytes")) for b in n.get("mem_banks") or [])
            for n in kfd))
    if any(fw.values()):
        lines.append(f"- Platform firmware: {fw.get('sys_vendor')} {fw.get('product_name')}, "
                     f"BIOS {fw.get('bios_version')} ({fw.get('bios_date')})")
    rev = env.get("echolock", {})
    if rev.get("commit"):
        lines.append(f"- echolock {rev['commit'][:12]}{' (dirty)' if rev.get('dirty') else ''}")
    return lines


# Clock-event reasons that mean the measurement is polluted: the GPU was throttling, not running
# at the clocks the numbers assume. gpu_idle is normal and excluded.
POLLUTION_REASONS = {"sw_power_cap", "hw_slowdown", "sw_thermal_slowdown",
                     "hw_thermal_slowdown", "hw_power_brake_slowdown", "sync_boost"}


def pollution_warnings(run: RunData) -> list[str]:
    """One line per GPU whose samples show throttling during the run, so a polluted number is
    flagged in the report instead of read as a result."""
    out: list[str] = []
    per_gpu: dict[int, set[str]] = {}
    for s in run.samples:
        for g in s.get("gpu") or []:
            names = {n for n in (g.get("clock_event_names") or []) if n in POLLUTION_REASONS}
            if names:
                per_gpu.setdefault(g.get("index", 0), set()).update(names)
    for i, names in sorted(per_gpu.items()):
        out.append(f"\u26a0 GPU{i} throttled during the run: {', '.join(sorted(names))} -- "
                   "clock-dependent numbers are lower than an unthrottled run would give.")
    return out


def sample_stats(run: RunData) -> list[str]:
    """Clock, temperature and power over the run, per GPU, plus host memory headroom."""
    lines: list[str] = []
    per_gpu: dict[int, dict[str, list]] = {}
    mem_avail = []
    reasons: dict[int, set[str]] = {}
    energy: dict[int, list[tuple[int, int]]] = {}
    for s in run.samples:
        for g in s.get("gpu") or []:
            i = g.get("index", 0)
            d = per_gpu.setdefault(i, {})
            for k in ("sm_clock_mhz", "temp_c", "power_w", "util_gpu_pct"):
                if isinstance(g.get(k), (int, float)):
                    d.setdefault(k, []).append(g[k])
            for name in g.get("clock_event_names") or []:
                reasons.setdefault(i, set()).add(name)
            if isinstance(g.get("energy_mj"), int):
                energy.setdefault(i, []).append((s["t_ns"], g["energy_mj"]))
        ma = ((s.get("host") or {}).get("mem") or {}).get("MemAvailable_kib")
        if isinstance(ma, int):
            mem_avail.append(ma)
    if not per_gpu and not mem_avail:
        return ["No environment samples."]
    lines.append("| Series | min | mean | max |\n|---|---:|---:|---:|")
    names = {"sm_clock_mhz": "MHz", "temp_c": "°C", "power_w": "W", "util_gpu_pct": "%"}
    for i, d in sorted(per_gpu.items()):
        for k, unit in names.items():
            v = d.get(k)
            if v:
                lines.append(f"| GPU{i} {k} | {fmt(float(min(v)), unit)} | "
                             f"{fmt(statistics.fmean(v), unit)} | {fmt(float(max(v)), unit)} |")
        e = energy.get(i)
        if e and len(e) > 1 and e[-1][0] > e[0][0]:
            watts = (e[-1][1] - e[0][1]) / 1e3 / ((e[-1][0] - e[0][0]) / 1e9)
            lines.append(f"| GPU{i} mean power from energy counter | | {fmt(watts, 'W')} | |")
    if mem_avail:
        lines.append(f"| host MemAvailable | {fmt(min(mem_avail) / 2**20, 'GiB')} | "
                     f"{fmt(statistics.fmean(mem_avail) / 2**20, 'GiB')} | "
                     f"{fmt(max(mem_avail) / 2**20, 'GiB')} |")
    for i, rs in sorted(reasons.items()):
        seen = sorted(rs - {"gpu_idle"})
        if seen:
            lines.append(f"\nGPU{i} clock-event reasons seen: {', '.join(seen)}")
    lines.append(f"\n{len(run.samples)} samples at {run.manifest.get('sample_interval_s', 1)} s.")
    return lines


def load_lines(run: RunData) -> list[str]:
    """What the load generator intended and achieved, from load.jsonl -- the grid axis made
    checkable: a condition is only as good as the load actually reached."""
    recs = load_jsonl(run.dir / "load.jsonl")
    if not recs:
        return []
    start = next((r for r in recs if r.get("type") == "load_start"), None)
    end = next((r for r in reversed(recs) if r.get("type") == "load_end"), None)
    errs = [r for r in recs if r.get("type") == "load_error"]
    if start is None:
        return [f"- Load: records exist but no load_start (`{json.dumps(recs[:1])}`)"]
    lines = [f"- Load `{start.get('load')}` intended `{json.dumps(start.get('intended'))}`"]
    if end is not None:
        lines.append(f"  - achieved `{json.dumps(end.get('achieved'))}` at stop")
    heartbeats = [r for r in recs if r.get("type") == "heartbeat"]
    if heartbeats:
        lines.append(f"  - {len(heartbeats)} heartbeats, last "
                      f"`{json.dumps(heartbeats[-1].get('achieved'))}`")
    for e in errs:
        lines.append(f"  - ERROR `{e.get('message')}`")
    return lines


def step_env_rows(run: RunData) -> list[str]:
    """Environment inside each step's own window, so a step's numbers can be read against what the
    GPU was doing while it ran. A step shorter than the sample interval can fall between two
    samples; the whole-run table above still brackets it, and such a step shows 0 samples."""
    steps = [s for s in run.manifest.get("steps", [])
             if isinstance(s.get("start_ns"), int) and isinstance(s.get("end_ns"), int)]
    if not steps or not run.samples:
        return []
    out = ["Environment inside each step (samples between the step's start and end):", "",
           "| Step | Samples | Clock MHz | Temp °C | Power W | Util % | MemAvail GiB |",
           "|---|---:|---|---|---|---|---|"]
    for s in steps:
        win = [sm for sm in run.samples if s["start_ns"] <= sm.get("t_ns", 0) <= s["end_ns"]]
        if not win:
            out.append(f"| {s['name']} | 0 | — | — | — | — | — |")
            continue
        per_gpu: dict[int, dict[str, list[float]]] = {}
        mem: list[int] = []
        for sm in win:
            for g in sm.get("gpu") or []:
                d = per_gpu.setdefault(g.get("index", 0), {})
                for k in ("sm_clock_mhz", "temp_c", "power_w", "util_gpu_pct"):
                    if isinstance(g.get(k), (int, float)):
                        d.setdefault(k, []).append(g[k])
            ma = ((sm.get("host") or {}).get("mem") or {}).get("MemAvailable_kib")
            if isinstance(ma, int):
                mem.append(ma)

        def cell(key: str) -> str:
            parts = []
            for _, d in sorted(per_gpu.items()):
                v = d.get(key)
                parts.append(f"{fmt(float(min(v)))}-{fmt(float(max(v)))}" if v else "—")
            return ", ".join(parts) or "—"

        ma_cell = fmt(min(mem) / 2 ** 20, "GiB") if mem else "—"
        out.append(f"| {s['name']} | {len(win)} | {cell('sm_clock_mhz')} | {cell('temp_c')} | "
                   f"{cell('power_w')} | {cell('util_gpu_pct')} | {ma_cell} |")
    return out


# Tag values that mean the same resource under different backend vocabularies, and backend-only
# tags dropped when aligning rows (they record per-backend mechanics, not the resource).
TAG_ALIASES = {
    "kind": {"vulkan_device_local": "device", "vulkan_host_visible": "pinned"},
    "source": {"vulkan_device_local": "cudaMalloc", "vulkan_host_visible": "cudaHostAlloc_mapped"},
}

# Metric names that mean the same measurement under different backend vocabularies: normalized
# name plus tags the backend did not record (the format the name already carries).
NAME_ALIASES = {
    "fp16_tflops": ("mma_throughput", {"format": "fp16"}),
    "bf16_tflops": ("mma_throughput", {"format": "bf16"}),
    "int8_tflops": ("mma_throughput", {"format": "int8"}),
}
DROPPED_TAGS = {"heap_device_local", "heap_host_visible", "gpu_us", "copy_us", "verify_read_us",
                "footprint_bytes", "shape", "iters", "chains", "workgroups", "device",
                "exec_span_us", "host_span_us"}

# Relative tolerance per metric for the cross-stack verdict: both implementations must find the
# same device behaviour, but they time through different APIs.
METRIC_TOLERANCE = {
    "gpu_read": 0.15, "copy_payload": 0.20, "cpu_read": 0.20,
    "chase_hop_latency": 0.30, "mma_throughput": 0.25,
}
DEFAULT_TOLERANCE = 0.15


def _align_tags(tags: dict) -> dict:
    out = {}
    for k, v in (tags or {}).items():
        if k in DROPPED_TAGS:
            continue
        out[k] = TAG_ALIASES.get(k, {}).get(v, v)
    return out


def _normalize(step: str, name: str, tags: dict) -> tuple[str, str, dict]:
    """Backend vocabularies -> one canonical (step, name, tags). Regex aliases cover names that
    encode a parameter (Vulkan's sync_<us>us.lag carries the duration in the name)."""
    m = re.fullmatch(r"sync_(\d+)us\.lag", name)
    if m:
        # One step name: the CUDA probe steps over schedules, the Vulkan probe has no schedule
        # axis, and the aligned comparison is the lag at each duration. The schedule tag stays
        # behind: a blocking-schedule CUDA run and the Vulkan submission are different mechanisms,
        # and the aligned question is the end-to-end lag.
        name, step = "sync_return_lag", "sync"
        tags = {"kernel_us": int(m.group(1))}
    if name == "sync_return_lag" and "kernel_us" in tags:
        # Canonical form on both backends: one step, one tag, the duration.
        step, tags = "sync", {"kernel_us": int(tags["kernel_us"])}
    return step, name, tags


def cross_stack_section(runs: list[RunData], rows: list[dict]) -> list[str]:
    """Aligns equivalent (step, metric, tags) rows across backends and judges each pair against
    the metric's tolerance. Only pairs both backends measured appear; a one-sided row stays in
    the Measured table above."""
    per_backend: dict[str, dict] = {}
    for r, rs in zip(runs, rows):
        backend = r.manifest.get("backend") or "?"
        for (step, name, tags_text_), e in rs.items():
            if e.get("verified") is False:
                continue  # a failed oracle is not a result to compare
            tags = _align_tags(_parse_tags(tags_text_))
            extra = NAME_ALIASES.get(name)
            if extra:
                name, injected = extra
                tags.update(injected)
            step, name, tags = _normalize(step, name, tags)
            per_backend.setdefault(backend, {})[(step, name, tags_text(tags))] = e
    backends = sorted(per_backend)
    if len(backends) < 2:
        return []
    a, b = backends[0], backends[1]
    out = [f"## Cross-stack: {a} vs {b}", "",
           "Equivalent metrics aligned through the tag-alias map (docs/crosscheck.md). The ratio "
           "is " + a + " / " + b + "; ✓ within tolerance, ⚠ diverges -- a divergence is a "
           "finding, not a failure.", ""]
    out.append("| Step | Metric | Tags | " + f"{a} | {b} | ratio | tol | verdict |")
    out.append("|---|---|---|---:|---:|---:|---:|---|")
    n_agree = n_diverge = 0
    for key in sorted(set(per_backend[a]) & set(per_backend[b])):
        ea, eb = per_backend[a][key], per_backend[b][key]
        va, vb = ea["value"], eb["value"]
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)) or vb == 0:
            continue
        tol = METRIC_TOLERANCE.get(key[1], DEFAULT_TOLERANCE)
        ratio = va / vb
        ok = abs(ratio - 1.0) <= tol
        n_agree += ok
        n_diverge += not ok
        mark = "✓" if ok else "⚠"
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {fmt(va, ea['unit'])} | "
                   f"{fmt(vb, eb['unit'])} | {ratio:.3f} | {int(tol * 100)}% | {mark} |")
    if n_agree + n_diverge == 0:
        return []  # nothing aligned: say nothing rather than an empty table
    out += ["", f"{n_agree} agree, {n_diverge} diverge within the aligned set.", ""]
    return out


def _parse_tags(text: str) -> dict:
    """Inverse of tags_text for the alias pass; tags_text output is 'k=v, k=v' ordered."""
    out = {}
    for part in (text or "").split(", "):
        k, _, v = part.partition("=")
        if v:
            out[k] = v
    return out


def render(run_dirs: list[Path]) -> str:
    runs = [RunData(d) for d in run_dirs]
    if not runs:
        return "# echolock\n\nNo run directories given.\n"
    first = runs[0]
    probe = next((r.manifest.get("probe") for r in runs if r.manifest.get("probe")), "unknown probe")
    backend = next((r.manifest.get("backend") for r in runs if r.manifest.get("backend")), "?")
    out = [f"# echolock: {probe} ({backend})", ""]
    out.append("| Run | Label | Load | Status | Started (UTC) | Steps |\n|---|---|---|---|---|---|")
    for r in runs:
        m = r.manifest
        steps = m.get("steps", [])
        ok = sum(1 for s in steps if s.get("status") == "ok")
        status = m.get("status") or "unreadable"
        if m.get("status_note"):
            status += f" ({m['status_note']})"
        if r.problem:
            status += f" \u26a0 {r.problem}"
        lc = (m.get("load_check") or {})
        if lc.get("pass") is False:
            status += f" \u26a0 load below target ({json.dumps(lc)})"
        out.append(f"| {m.get('run_id') or r.dir} | {r.label} | {json.dumps(m.get('load'))} | "
                   f"{status} | {m.get('started_utc')} | "
                   f"{ok}/{len(m.get('planned_steps', steps))} ok |")
    out += ["", "## Environment", "", *env_lines(first), ""]
    warnings: list[str] = []
    for r in runs:
        for line in pollution_warnings(r):
            warnings.append(f"- ({r.column}) {line}")
    if warnings:
        out += ["## Warnings", "", *warnings, ""]

    # Declared, one column per run, so a value that differs between runs (a clock under load,
    # granted flags per schedule) reads side by side like the measurements do.
    declared_rows: dict[str, dict] = {}
    for r in runs:
        per_key: dict[str, list[dict]] = {}
        for d in r.of_type("declared"):
            per_key.setdefault(d.get("key", "?"), []).append(d)
        for key, ds in per_key.items():
            # One row when the run declares the key once; one row per step when it declares the
            # same key in several steps (sync's device_flags, once per schedule).
            for d in ds:
                row_id = key if len(ds) == 1 else f"{key} [{d.get('step', '')}]"
                row = declared_rows.setdefault(row_id, {"source": d.get("source", ""), "values": {}})
                row["values"][r.column] = json.dumps(d.get("value"))
    if declared_rows:
        out += ["## Declared", "",
                "| Key | Source | " + " | ".join(r.column for r in runs) + " |",
                "|---|---|" + "---|" * len(runs)]
        for rid, row in declared_rows.items():
            cells = " | ".join(row["values"].get(r.column, "—") for r in runs)
            out.append(f"| {rid} | {row['source']} | {cells} |")
        out.append("")

    # Measured, one column per run.
    rows = [measurement_rows(r) for r in runs]
    keys: list[tuple[str, str, str]] = []
    for rs in rows:
        for k in rs:
            if k not in keys:
                keys.append(k)
    if keys:
        out += ["## Measured", "",
                "Values are the series summary (per-rep spread in brackets). ✗ marks a value whose "
                "oracle check failed; it is not a result.", ""]
        out.append("| Step | Metric | Tags | " + " | ".join(r.column for r in runs) + " |")
        out.append("|---|---|---|" + "---:|" * len(runs))
        for k in keys:
            cells = []
            for rs in rows:
                e = rs.get(k)
                if e is None:
                    cells.append("—")
                    continue
                cell = fmt(e["value"], e["unit"])
                pr = e.get("per_rep")
                if isinstance(pr, dict) and "min" in pr:
                    cell += f" [{fmt(pr['min'])}–{fmt(pr['max'])}]"
                if e.get("verified") is False:
                    cell += " ✗"
                cells.append(cell)
            out.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")
        out.append("")

    # Cross-stack: when the runs come from more than one backend, align equivalent metrics through
    # a tag-alias map and give a tolerance verdict per pair. Two independent implementations of
    # the same probe semantics must find the same device behaviour; a divergence is a finding.
    if len({r.manifest.get("backend") for r in runs}) > 1:
        out += cross_stack_section(runs, rows)

    # Checks and fallbacks.
    out += ["## Verification", ""]
    for r in runs:
        checks = r.of_type("check")
        failed = [c for c in checks if not c.get("pass")]
        out.append(f"- {r.label}: {len(checks) - len(failed)}/{len(checks)} checks passed")
        for c in failed:
            out.append(f"  - FAILED `{c.get('step')}` {c.get('name')}: "
                       f"`{json.dumps(c.get('detail'))}`")
        for e in r.of_type("error"):
            out.append(f"  - ERROR `{e.get('step')}`: {e.get('message')}")
    out.append("")
    # What the probe chose not to measure, which is as much a result as what it did.
    notes = [(r, n) for r in runs for n in r.of_type("note")]
    if notes:
        out += ["## Notes", "",
                "Things a probe gave up on or skipped. A section of the probe that produced no "
                "measurements and appears nowhere else appears only here.", ""]
        for r, n in notes:
            out.append(f"- {r.label} `{n.get('step')}`: {n.get('message')}")
        out.append("")

    features = [(r, f) for r in runs for f in r.of_type("feature")]
    if features:
        out += ["## Requested features", "",
                "| Run | Feature | Requested | Granted (API) | Effect observed | Evidence |",
                "|---|---|---|---|---|---|"]
        for r, f in features:
            silent = f.get("requested") and f.get("granted") is not False and not f.get("effect_observed")
            out.append(f"| {r.label} | {f.get('feature')} | {fmt(f.get('requested'))} | "
                       f"{fmt(f.get('granted'))} | {fmt(f.get('effect_observed'))}"
                       f"{' — **silent fallback**' if silent else ''} | "
                       f"`{json.dumps(f.get('evidence'))}` |")
        out.append("")

    # Environment during the run, and kernel events.
    for r in runs:
        out += [f"## During the run: {r.label}", "", *sample_stats(r), ""]
        rows = step_env_rows(r)
        if rows:
            out += ["", *rows, ""]
        load = load_lines(r)
        if load:
            out += ["### Load", "", *load, ""]
        if r.problem:
            out += [f"Incomplete run directory: {r.problem}", ""]
        kl = r.manifest.get("kernel_log", {})
        lost, overruns = kl.get("lost_records"), kl.get("overruns")
        if isinstance(lost, int) and lost > 0:
            out += [f"Kernel log lost {lost} message(s) while watching"
                     f" ({overruns if isinstance(overruns, int) else 0} ring overrun(s)): "
                     f"events around them may be missing.", ""]
        if not kl.get("available", False):
            out += [f"Kernel log not watched: {kl.get('error')}", ""]
        elif r.kernel:
            out += ["Kernel events during the run:", ""]
            t0 = r.manifest.get("t_start_ns") or 0
            for k in r.kernel:
                when = (f"+{(k['t_ns'] - t0) / 1e9:.1f} s"
                        if t0 and isinstance(k.get("t_ns"), int) else "t unknown")
                out.append(f"- {when} [{k.get('category')}] {k.get('message')}")
            out.append("")
        else:
            prior = kl.get("before_run") or {}
            out += ["No kernel events during the run"
                    + (f" (before it, since boot: {json.dumps(prior)})" if prior else "") + ".", ""]
    return "\n".join(out)
