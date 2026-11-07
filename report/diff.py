"""Run-to-run diff (report/diff.py), regress baseline selection, and the phase-3 crosscheck
invariants (docs/crosscheck.md) that ./echolock crosscheck --check applies.

The diff compares two run directories on the canonical tag mapping (report/tags.py): every
canonical metric present in either run gets a row with both values, the relative delta, and a
verdict (within / beyond / missing). Exit contract: 0 when every present metric is within the
tolerance, 1 when any is beyond, 2 on usage error.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from report.markdown import RunData
from report.tags import run_metrics, unmapped_metrics

# Phase-3 invariants (docs/crosscheck.md): named relative tolerances, |b - a| / |a| <= tol.
# Both stacks should find the same ceiling on the same hardware; the tolerances are loose
# enough for clock and scheduling drift and tight enough to catch a broken measurement path.
CROSSCHECK_TOL = {
    "memory": 0.10,        # DRAM read bandwidth and h2d copy payload
    "pointer_chase": 0.15, # dependent-load hop latency
    "mma": 0.20,           # tensor-core ceilings; the two stacks use different instructions
}

VERDICT_A = "within-tolerance"
VERDICT_B = "diverges"
VERDICT_AGREES = "agrees"
VERDICT_MISSING = "missing"


def _num(v: object) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def rel_delta(a: float | None, b: float | None) -> float | None:
    """(b - a) / |a|; None when the values are not both usable numbers."""
    if a is None or b is None:
        return None
    if a == 0:
        return 0.0 if b == 0 else None
    return (b - a) / abs(a)


@dataclass
class DiffRow:
    key: str
    a: float | None
    b: float | None
    unit: str
    delta: float | None
    verdict: str  # within / beyond / missing


@dataclass
class Diff:
    a: Path
    b: Path
    tol: float
    rows: list[DiffRow] = field(default_factory=list)
    env_rows: list[tuple[str, object, object]] = field(default_factory=list)
    a_kernel: dict = field(default_factory=dict)
    b_kernel: dict = field(default_factory=dict)
    a_load: dict = field(default_factory=dict)
    b_load: dict = field(default_factory=dict)
    a_unmapped: list[str] = field(default_factory=list)
    b_unmapped: list[str] = field(default_factory=list)

    @property
    def beyond(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "beyond")

    @property
    def exit_code(self) -> int:
        return 1 if self.beyond else 0


def diff_runs(a_dir: Path, b_dir: Path, tol: float) -> Diff:
    a, b = RunData(a_dir), RunData(b_dir)
    ma, mb = run_metrics(a), run_metrics(b)
    out = Diff(a_dir, b_dir, tol)
    for key in sorted(set(ma) | set(mb)):
        va, vb = _num(ma.get(key, {}).get("value")) if key in ma else None, \
            _num(mb.get(key, {}).get("value")) if key in mb else None
        delta = rel_delta(va, vb)
        if key not in ma or key not in mb or va is None or vb is None:
            verdict = "missing"
        elif delta is None:
            verdict = "beyond"  # present on both sides but not comparable numbers
        elif abs(delta) <= tol:
            verdict = "within"
        else:
            verdict = "beyond"
        unit = (ma.get(key, {}) or mb.get(key, {})).get("unit", "")
        out.rows.append(DiffRow(key, va, vb, unit, delta, verdict))
    out.env_rows = env_delta(a, b)
    out.a_kernel, out.b_kernel = kernel_summary(a), kernel_summary(b)
    out.a_load, out.b_load = load_summary(a), load_summary(b)
    out.a_unmapped, out.b_unmapped = unmapped_metrics(a), unmapped_metrics(b)
    return out


def env_delta(a: RunData, b: RunData) -> list[tuple[str, object, object]]:
    """Environment facts that differ between two runs: driver, firmware, toolchain.build, the
    AMD forced performance level, and the declared keys whose values differ."""
    out: list[tuple[str, object, object]] = []
    ea, eb = a.env, b.env
    pairs: list[tuple[str, object]] = [
        ("nvidia.nvml.driver_version", (ea.get("nvidia") or {}).get("nvml", {}).get("driver_version")),
        ("firmware.dmi.bios_version", ((ea.get("firmware") or {}).get("dmi") or {}).get("bios_version")),
    ]
    for i, card in enumerate(ea.get("amd", {}).get("cards") or []):
        pairs.append((f"amd.cards[{i}].vbios_version", card.get("vbios_version")))
        pairs.append((f"amd.cards[{i}].driver_version", card.get("driver_version")))
        pairs.append((f"amd.cards[{i}].performance_level", card.get("performance_level")))
        for clk in ("sclk", "mclk"):
            states = ((card.get("clocks") or {}).get(clk) or {})
            pairs.append((f"amd.cards[{i}].clocks.{clk}.forced_level", states.get("forced_level")))
    for i, card in enumerate(eb.get("amd", {}).get("cards") or []):
        for key, _ in (
            (f"amd.cards[{i}].vbios_version", card.get("vbios_version")),
            (f"amd.cards[{i}].driver_version", card.get("driver_version")),
            (f"amd.cards[{i}].performance_level", card.get("performance_level")),
        ):
            if all(k != key for k, _ in pairs):
                pairs.append((key, None))
        for clk in ("sclk", "mclk"):
            key = f"amd.cards[{i}].clocks.{clk}.forced_level"
            if all(k != key for k, _ in pairs):
                pairs.append((key, None))
    for key, keyname in ((k, k) for k in ("arch", "nvcc", "flags", "built_utc")):
        pairs.append((f"toolchain.build.{keyname}",
                      ((ea.get("toolchain") or {}).get("build") or {}).get(key)))
    # b-side values for every compared key
    bvals: dict[str, object] = {}
    for i, card in enumerate(eb.get("amd", {}).get("cards") or []):
        bvals[f"amd.cards[{i}].vbios_version"] = card.get("vbios_version")
        bvals[f"amd.cards[{i}].driver_version"] = card.get("driver_version")
        bvals[f"amd.cards[{i}].performance_level"] = card.get("performance_level")
        for clk in ("sclk", "mclk"):
            bvals[f"amd.cards[{i}].clocks.{clk}.forced_level"] = \
                ((card.get("clocks") or {}).get(clk) or {}).get("forced_level")
    bvals["nvidia.nvml.driver_version"] = (eb.get("nvidia") or {}).get("nvml", {}).get("driver_version")
    bvals["firmware.dmi.bios_version"] = ((eb.get("firmware") or {}).get("dmi") or {}).get("bios_version")
    bbuild = (eb.get("toolchain") or {}).get("build") or {}
    for key in ("arch", "nvcc", "flags", "built_utc"):
        bvals[f"toolchain.build.{key}"] = bbuild.get(key)

    for key, va in pairs:
        vb = bvals.get(key)
        if va != vb:
            out.append((key, va, vb))
    # declared keys: what the two probes reported about the device
    da = {d.get("key"): d.get("value") for d in a.of_type("declared")}
    db = {d.get("key"): d.get("value") for d in b.of_type("declared")}
    for key in sorted(set(da) | set(db)):
        if da.get(key) != db.get(key):
            out.append((f"declared {key}", da.get(key), db.get(key)))
    return out


def kernel_summary(run: RunData) -> dict:
    kl = run.manifest.get("kernel_log") or {}
    during = kl.get("during_run") or {}
    return {
        "available": kl.get("available"),
        "error": kl.get("error"),
        "during_run_events": sum(during.values()) if isinstance(during, dict) else None,
        "during_run": during if during else None,
        "lost_records": kl.get("lost_records"),
        "overruns": kl.get("overruns"),
    }


def load_summary(run: RunData) -> dict:
    """Intended vs achieved from the run's load.jsonl (empty for runs without a load)."""
    recs = [json.loads(l) for l in (run.dir / "load.jsonl").read_text().splitlines()
            if l.strip()] if (run.dir / "load.jsonl").is_file() else []
    if not recs:
        return {}
    start = next((r for r in recs if r.get("type") == "load_start"), {})
    end = next((r for r in reversed(recs) if r.get("type") in ("load_end", "heartbeat")), {})
    return {
        "load": start.get("load"),
        "intended": start.get("intended"),
        "ready": any(r.get("type") == "load_ready" for r in recs),
        "achieved": end.get("achieved"),
    }


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_diff(d: Diff) -> str:
    ma, mb = RunData(d.a).manifest, RunData(d.b).manifest
    lines = ["# Run diff", ""]
    lines.append(f"- **A**: `{d.a.name}` — {ma.get('probe')} / {ma.get('backend')} / "
                 f"{ma.get('label')} — {ma.get('started_utc')} (status {ma.get('status')})")
    lines.append(f"- **B**: `{d.b.name}` — {mb.get('probe')} / {mb.get('backend')} / "
                 f"{mb.get('label')} — {mb.get('started_utc')} (status {mb.get('status')})")
    lines.append(f"- tolerance: ±{d.tol:.0%} relative; verdicts: within / beyond / missing")
    n_within = sum(1 for r in d.rows if r.verdict == "within")
    n_missing = sum(1 for r in d.rows if r.verdict == "missing")
    lines += ["", f"**{n_within} within, {d.beyond} beyond, {n_missing} missing** of "
              f"{len(d.rows)} canonical metric(s).", ""]
    lines += ["| canonical metric | A | B | rel. delta | verdict |", "|---|---|---|---|---|"]
    for r in d.rows:
        delta = "—" if r.delta is None else f"{r.delta:+.2%}"
        unit = f" {r.unit}" if r.unit else ""
        lines.append(f"| {r.key} | {_fmt(r.a)} | {_fmt(r.b)} | {delta} | {r.verdict} |"
                     if not unit else
                     f"| {r.key} | {_fmt(r.a)} | {_fmt(r.b)} {unit} | {delta} | {r.verdict} |")
    if d.a_unmapped or d.b_unmapped:
        lines += ["", "Unmapped measure series (no canonical row in report/tags.py):"]
        for side, run_dir, lst in (("A", d.a, d.a_unmapped), ("B", d.b, d.b_unmapped)):
            if lst:
                lines.append(f"- {side} ({run_dir.name}): " + "; ".join(lst))
    if d.env_rows:
        lines += ["", "## Environment delta", "", "| key | A | B |", "|---|---|---|"]
        for key, va, vb in d.env_rows:
            lines.append(f"| {key} | {_fmt(va)} | {_fmt(vb)} |")
    else:
        lines += ["", "## Environment delta", "", "none: driver, firmware, toolchain.build, "
                  "forced performance level and all declared keys agree."]
    lines += ["", "## Kernel events during run", "",
              f"- A: {_kernel_text(d.a_kernel)}", f"- B: {_kernel_text(d.b_kernel)}"]
    lines += ["", "## Load", "",
              f"- A: {_load_text(d.a_load)}", f"- B: {_load_text(d.b_load)}"]
    lines.append("")
    return "\n".join(lines)


def _kernel_text(k: dict) -> str:
    if not k.get("available"):
        return "unavailable" + (f" ({k.get('error')})" if k.get("error") else "")
    events = k.get("during_run")
    text = "none during run" if not events else f"{k.get('during_run_events')} during run: " \
        + ", ".join(f"{c}×{n}" for c, n in sorted(events.items()))
    if k.get("lost_records") or k.get("overruns"):
        text += f"; lost {k.get('lost_records')} records, {k.get('overruns')} overruns"
    return text


def _load_text(l: dict) -> str:
    if not l:
        return "no load condition"
    ready = "ready" if l.get("ready") else "never reported ready"
    return (f"{l.get('load')} intended {json.dumps(l.get('intended'))} — {ready}, "
            f"achieved {json.dumps(l.get('achieved'))}")


# ---------------------------------------------------------------- regress baselines

def _declared_device(run: RunData) -> str | None:
    return next((d.get("value") for d in run.of_type("declared")
                 if d.get("key") == "device.name"), None)


def run_dirs(run_root: Path) -> list[RunData]:
    if not run_root.is_dir():
        return []
    return [RunData(p) for p in sorted(run_root.iterdir())
            if p.is_dir() and (p / "manifest.json").is_file()]


def find_baseline(new_dir: Path, run_root: Path) -> Path | None:
    """The most recent run (manifest started_utc) with the same probe, backend, label and
    declared device.name; None when no such run exists."""
    new = RunData(new_dir)
    new_resolved = new_dir.resolve()
    probe, backend, label = (new.manifest.get(k) for k in ("probe", "backend", "label"))
    device = _declared_device(new)
    best: tuple[str, Path] | None = None
    for rd in run_dirs(run_root):
        if rd.dir.resolve() == new_resolved or not rd.manifest:
            continue
        if (rd.manifest.get("probe"), rd.manifest.get("backend"), rd.manifest.get("label")) != \
                (probe, backend, label):
            continue
        if device is not None and _declared_device(rd) != device:
            continue
        stamp = str(rd.manifest.get("started_utc", ""))
        if best is None or stamp > best[0]:
            best = (stamp, rd.dir)
    return best[1] if best else None


def candidates_for(new_dir: Path, run_root: Path) -> list[RunData]:
    """Runs with the same probe and backend (any label/device), newest first."""
    new = RunData(new_dir)
    new_resolved = new_dir.resolve()
    probe, backend = new.manifest.get("probe"), new.manifest.get("backend")
    out = [rd for rd in run_dirs(run_root) if rd.dir.resolve() != new_resolved
           and rd.manifest.get("probe") == probe and rd.manifest.get("backend") == backend]
    out.sort(key=lambda rd: str(rd.manifest.get("started_utc", "")), reverse=True)
    return out


# ---------------------------------------------------------------- crosscheck --check

def _crosscheck_record(probe: str, name: str, a: RunData, b: RunData,
                       va: object, vb: object, verdict: str, tolerance: float | None,
                       detail: str) -> dict:
    return {
        "v": 1, "t_ns": time.monotonic_ns(), "src": "crosscheck", "type": "crosscheck",
        "probe": probe, "name": name,
        "a": {"run_id": a.manifest.get("run_id"), "value": va},
        "b": {"run_id": b.manifest.get("run_id"), "value": vb},
        "verdict": verdict, "tolerance": tolerance, "detail": detail,
    }


def crosscheck_verdicts(a_dir: Path, b_dir: Path) -> list[dict]:
    """The documented phase-3 invariants for the probe of two ok runs, as crosscheck records.

    Numeric invariants compare canonical keys with the named tolerances in CROSSCHECK_TOL.
    The mma bit-exact check compares the backends' ref_xor checksum detail strings exactly:
    when either run has no such record (mma not run, or a binary that predates the checksum
    change) the verdict is "missing" with the reason, not a pass.
    """
    a, b = RunData(a_dir), RunData(b_dir)
    probe = a.manifest.get("probe")
    if a.manifest.get("backend") == b.manifest.get("backend"):
        return []
    ma, mb = run_metrics(a), run_metrics(b)
    recs: list[dict] = []

    def add(name: str, key: str, tol: float) -> None:
        va = _num(ma.get(key, {}).get("value")) if key in ma else None
        vb = _num(mb.get(key, {}).get("value")) if key in mb else None
        delta = rel_delta(va, vb)
        if delta is None:
            recs.append(_crosscheck_record(probe, name, a, b, va, vb, VERDICT_MISSING, tol,
                                           "value absent from one or both runs"))
        elif delta == 0.0:
            recs.append(_crosscheck_record(probe, name, a, b, va, vb, VERDICT_AGREES, tol,
                                           "identical values"))
        elif abs(delta) <= tol:
            recs.append(_crosscheck_record(probe, name, a, b, va, vb, VERDICT_A, tol,
                                           f"relative delta {delta:+.2%}"))
        else:
            recs.append(_crosscheck_record(probe, name, a, b, va, vb, VERDICT_B, tol,
                                           f"relative delta {delta:+.2%}"))

    if probe == "memory":
        for key in ("memory.gpu_read.device_local", "memory.copy.h2d.host_visible"):
            add(key, key, CROSSCHECK_TOL["memory"])
    elif probe == "pointer_chase":
        def _footprint(k: str) -> float:
            try:
                return float(k.split(".")[3])
            except (ValueError, IndexError):
                return float("inf")

        hop_keys = sorted((k for k in set(ma) | set(mb)
                           if k.startswith("pointer_chase.hop.device_local.")),
                          key=lambda k: (_footprint(k), k))
        for key in ([hop_keys[0], hop_keys[-1]] if len(hop_keys) >= 2 else hop_keys):
            add(f"{key} ({'smallest' if key == hop_keys[0] else 'largest'} footprint)",
                key, CROSSCHECK_TOL["pointer_chase"])
        if not hop_keys:
            recs.append(_crosscheck_record(probe, "pointer_chase.hop.device_local", a, b,
                                           None, None, VERDICT_MISSING,
                                           CROSSCHECK_TOL["pointer_chase"],
                                           "no device-local hop metrics in either run"))
    elif probe == "mma":
        for fmt in ("fp16", "bf16", "int8"):
            add(f"mma.tflops.{fmt}", f"mma.tflops.{fmt}", CROSSCHECK_TOL["mma"])
        ra = ref_xor_details(a)
        rb = ref_xor_details(b)
        if ra and rb:
            if ra == rb:
                recs.append(_crosscheck_record(probe, "mma.ref_xor", a, b,
                                               len(ra), len(rb), VERDICT_AGREES, None,
                                               f"{len(ra)} ref_xor checksum(s) equal"))
            else:
                recs.append(_crosscheck_record(probe, "mma.ref_xor", a, b, ra, rb,
                                               VERDICT_B, None,
                                               "ref_xor checksums differ: "
                                               f"A={ra} B={rb}"))
        else:
            why = ("no ref_xor check records in "
                   + ", ".join(f"{r.manifest.get('run_id')}"
                               for r, lst in ((a, ra), (b, rb)) if not lst))
            recs.append(_crosscheck_record(probe, "mma.ref_xor", a, b,
                                           ra or None, rb or None, VERDICT_MISSING, None,
                                           f"not a pass: {why} (mma not run, or binary "
                                           "predates the checksum change)"))
    else:
        recs.append(_crosscheck_record(probe, f"{probe}.no-invariants", a, b, None, None,
                                       VERDICT_MISSING, None,
                                       "no phase-3 invariants are defined for this probe"))
    return recs


def ref_xor_details(run: RunData) -> list[str]:
    """Check-record detail strings that carry an mma ref_xor checksum (the mma backends fold
    every oracle-checked D-fragment word into one XOR per format and put it in the detail)."""
    out = []
    for c in run.of_type("check"):
        d = c.get("detail")
        if isinstance(d, str) and "ref_xor" in d:
            out.append(d)
    return sorted(out)


def render_crosscheck_section(recs: list[dict]) -> str:
    lines = ["", "## Crosscheck", ""]
    if not recs:
        lines.append("No crosscheck verdicts were produced (see the run log).")
        return "\n".join(lines) + "\n"
    lines += ["Verdicts are `crosscheck` records, also written to `crosscheck.jsonl` in both "
              "run directories. Tolerances are the named ones in report/diff.py.", "",
              "| check | A | B | verdict | tolerance | detail |",
              "|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(f"| {r['name']} | {r['a']['run_id']}: {_fmt(r['a']['value'])} "
                     f"| {r['b']['run_id']}: {_fmt(r['b']['value'])} | {r['verdict']} "
                     f"| {r['tolerance'] if r['tolerance'] is not None else '—'} "
                     f"| {r['detail']} |")
    lines.append("")
    return "\n".join(lines)
