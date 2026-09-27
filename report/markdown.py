"""Markdown report for one or more run directories.

Runs of the same probe taken under different loads sit side by side: one column per run label
(idle, 50% load, 95% fill ...). Declared values come first, then measurements, then everything
that qualifies them: failed checks, silent fallbacks, what the clocks, temperature and power did
during the run, and kernel events.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


class RunData:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.manifest = json.loads((run_dir / "manifest.json").read_text())
        env_path = run_dir / "env.json"
        self.env = json.loads(env_path.read_text()) if env_path.exists() else {}
        self.records = load_jsonl(run_dir / "probe.jsonl")
        self.samples = load_jsonl(run_dir / "samples.jsonl")
        self.kernel = load_jsonl(run_dir / "kernel.jsonl")

    @property
    def label(self) -> str:
        return self.manifest.get("label", self.dir.name)

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


def env_lines(run: RunData) -> list[str]:
    env = run.env
    host = env.get("host", {})
    nv = env.get("nvidia", {})
    smi = (nv.get("smi") or {})
    gpus = smi.get("gpus") or []
    fw = env.get("firmware", {}).get("dmi", {})
    lines = [
        f"- Host: {host.get('os', {}).get('pretty_name', '?')}, kernel {host.get('kernel', '?')}, "
        f"{host.get('machine', '?')}, {len(host.get('cpus', []))} CPUs",
        f"- Driver: {smi.get('driver_version') or (nv.get('nvml') or {}).get('driver_version') or '—'}; "
        f"CUDA (driver) {smi.get('cuda_version') or '—'}; "
        f"nvcc: {env.get('toolchain', {}).get('nvcc') or '—'}",
    ]
    for g in gpus:
        lines.append(f"- GPU: {g.get('product_name')} ({g.get('pci_device_id')}), "
                     f"VBIOS {g.get('vbios_version')}, GSP firmware {g.get('gsp_firmware_version')}")
    if any(fw.values()):
        lines.append(f"- Platform firmware: {fw.get('sys_vendor')} {fw.get('product_name')}, "
                     f"BIOS {fw.get('bios_version')} ({fw.get('bios_date')})")
    rev = env.get("echolock", {})
    if rev.get("commit"):
        lines.append(f"- echolock {rev['commit'][:12]}{' (dirty)' if rev.get('dirty') else ''}")
    return lines


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


def render(run_dirs: list[Path]) -> str:
    runs = [RunData(d) for d in run_dirs]
    first = runs[0]
    probe = first.manifest.get("probe")
    out = [f"# echolock: {probe} ({first.manifest.get('backend')})", ""]
    out.append("| Run | Label | Load | Status | Started (UTC) | Steps |\n|---|---|---|---|---|---|")
    for r in runs:
        m = r.manifest
        steps = m.get("steps", [])
        ok = sum(1 for s in steps if s.get("status") == "ok")
        status = m.get("status") + (f" ({m['status_note']})" if m.get("status_note") else "")
        out.append(f"| {m.get('run_id')} | {r.label} | {json.dumps(m.get('load'))} | {status} | "
                   f"{m.get('started_utc')} | {ok}/{len(m.get('planned_steps', steps))} ok |")
    out += ["", "## Environment", "", *env_lines(first), ""]

    # Declared.
    declared: dict[str, tuple[str, str]] = {}
    for r in runs:
        for d in r.of_type("declared"):
            declared.setdefault(d["key"], (json.dumps(d.get("value")), d.get("source", "")))
    if declared:
        out += ["## Declared", "", "| Key | Value | Source |", "|---|---|---|"]
        out += [f"| {k} | {v} | {src} |" for k, (v, src) in declared.items()]
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
        out.append("| Step | Metric | Tags | " + " | ".join(r.label for r in runs) + " |")
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
        kl = r.manifest.get("kernel_log", {})
        if not kl.get("available", False):
            out += [f"Kernel log not watched: {kl.get('error')}", ""]
        elif r.kernel:
            out += ["Kernel events during the run:", ""]
            t0 = r.manifest.get("t_start_ns", 0)
            for k in r.kernel:
                out.append(f"- +{(k['t_ns'] - t0) / 1e9:.1f} s [{k.get('category')}] {k.get('message')}")
            out.append("")
        else:
            prior = kl.get("before_run") or {}
            out += ["No kernel events during the run"
                    + (f" (before it, since boot: {json.dumps(prior)})" if prior else "") + ".", ""]
    return "\n".join(out)
