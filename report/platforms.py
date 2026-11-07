"""docs/platforms.md generator: one table per probe, one row per run.

Every cell comes from the run's own data: device and vendor from the probe's declared records
(falling back to env.json), driver from the declared version records, the headline metric from
the canonical tag mapping (report/tags.py), the oracle verdict from the manifest. Rows are
sorted deterministically so two runs of the generator over the same corpus give the same file.
"""

from __future__ import annotations

from pathlib import Path

from report.markdown import RunData
from report.tags import run_metrics

ROOT = Path(__file__).resolve().parents[1]

# The headline metric per probe: the number the probe exists to report, by canonical key
# (first present wins when a run used non-default parameters).
#
# memory        device-local gpu_read with no contention -- the DRAM read bandwidth the whole
#               probe is about (GB/s).
# pointer_chase device-local hop latency at the largest default footprint (64 MiB), random
#               pattern -- the DRAM-bound hop (ns/hop); the smallest footprint is the cache
#               floor and says less about the hardware.
# mma           fp16 ceiling (TFLOPS) -- the format every backend runs and the one the others
#               are expressed against (format_ratio).
# sync          lag at the 1 ms duration -- the middle of the duration axis (us).
# clock         the comparison the probe exists to report: derived clock vs the declared one.
# file_page     the GPU's steady-state read rate over resident rows.
# constants     no measured quantity -- device facts and their agreement across sources.
HEADLINE: dict[str, list[str] | dict[str, list[str]]] = {
    "memory": ["memory.gpu_read.device_local"],
    "pointer_chase": ["pointer_chase.hop.device_local.64.random",
                      "pointer_chase.hop.device_local.64.seq"],
    "mma": ["mma.tflops.fp16"],
    "sync": {"vulkan": ["sync.lag.queue.1000us"],
             "cuda": ["sync.lag.auto.1000us", "sync.lag.blocking.1000us"]},
    "clock": {"vulkan": ["clock.shader_clock_over_sclk"], "cuda": ["clock.derived_sm_clock_mhz"]},
    "file_page": {"vulkan": ["file_page.res_rows"], "cuda": ["file_page.gpu_seq_read"]},
    "constants": [],
}

# Human column heading for the headline metric of each probe.
HEADLINE_COLUMN = {
    "memory": "gpu_read device-local (GB/s)",
    "pointer_chase": "hop latency 64 MiB random (ns/hop)",
    "mma": "fp16 ceiling (TFLOPS)",
    "sync": "sync lag 1 ms (us)",
    "clock": "derived-vs-declared clock",
    "file_page": "GPU read, resident rows",
    "constants": "—",
}


def vendor_of(device: str | None, env: dict) -> str:
    """Vendor from the declared device name, then from env.json's driver sections."""
    d = (device or "").upper()
    if "NVIDIA" in d:
        return "NVIDIA"
    if "AMD" in d or "RADEON" in d or "RYZEN" in d:
        return "AMD"
    if env.get("nvidia", {}).get("nvml", {}).get("devices"):
        return "NVIDIA"
    if env.get("amd", {}).get("cards"):
        return "AMD"
    return "?"


def driver_of(run: RunData) -> str:
    """Driver version from the probe's declared records, then from env.json."""
    declared = {d.get("key"): d.get("value") for d in run.of_type("declared")}
    for key in ("vulkan.driver_version", "cuda.driver_version"):
        if declared.get(key):
            return str(declared[key])
    nv = run.env.get("nvidia", {}).get("nvml", {}).get("driver_version")
    if nv:
        return str(nv)
    cards = run.env.get("amd", {}).get("cards") or []
    if cards and cards[0].get("driver_version"):
        return str(cards[0]["driver_version"])
    return "—"


def run_row(run: RunData) -> dict | None:
    """One platforms.md row for a run directory, or None when the manifest is unusable."""
    m = run.manifest
    if not m.get("run_id"):
        return None
    declared = {d.get("key"): d.get("value") for d in run.of_type("declared")}
    device = str(declared.get("device.name", "—"))
    backend = str(m.get("backend", "?"))
    probe = str(m.get("probe", "?"))
    headline_keys = HEADLINE.get(probe)
    if isinstance(headline_keys, dict):
        headline_keys = headline_keys.get(backend, [])
    metrics = run_metrics(run)
    headline = None
    for key in headline_keys or []:
        if key in metrics and metrics[key].get("value") is not None:
            headline = f"{metrics[key]['value']:.1f} {metrics[key].get('unit') or ''}".strip()
            break
    status = m.get("status", "?")
    return {
        "probe": probe,
        "backend": backend,
        "device": device,
        "vendor": vendor_of(declared.get("device.name"), run.env),
        "driver": driver_of(run),
        "label": str(m.get("label", "?")),
        "headline": headline or "—",
        "oracle": {"ok": "pass", "verify_failed": "verify failed"}.get(status, status),
        "run_id": m.get("run_id"),
        "date": str(m.get("started_utc", "?"))[:10],
    }


def run_rows(run_root: Path) -> list[dict]:
    rows = []
    for child in sorted(run_root.iterdir()):
        if not (child.is_dir() and (child / "manifest.json").is_file()):
            continue
        row = run_row(RunData(child))
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (r["vendor"], r["backend"], r["device"], r["label"],
                             r["date"], r["run_id"]))
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = ["# Platforms", "",
             "Generated by `./echolock platforms`; do not edit by hand.", "",
             "One row per run directory under `runs/`: the device the probe declared, the stack "
             "that measured it, the run's load condition, one headline metric per probe (the "
             "choice is documented in report/platforms.py), and the probe's own oracle verdict "
             "from the manifest."]
    probes = sorted({r["probe"] for r in rows})
    for probe in probes:
        probe_rows = [r for r in rows if r["probe"] == probe]
        lines += ["", f"## {probe}", "",
                  "| device | vendor | backend | driver | label | "
                  f"{HEADLINE_COLUMN.get(probe, 'headline')} | oracle | run_id | date |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in probe_rows:
            lines.append(f"| {r['device']} | {r['vendor']} | {r['backend']} | {r['driver']} "
                         f"| {r['label']} | {r['headline']} | {r['oracle']} | {r['run_id']} "
                         f"| {r['date']} |")
    lines.append("")
    return "\n".join(lines)


def generate(run_root: Path, out_path: Path) -> int:
    rows = run_rows(run_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(rows))
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: `platforms [RUN_DIR...] [--out PATH]`; no args = every run under runs/."""
    import argparse

    p = argparse.ArgumentParser(prog="platforms",
                                description="docs/platforms.md: one row per run directory")
    p.add_argument("runs", nargs="*", help="run directories; default: every run under runs/")
    p.add_argument("--out", default=None, help="output file (default docs/platforms.md)")
    a = p.parse_args(argv)
    out = Path(a.out) if a.out else ROOT / "docs" / "platforms.md"
    rows: list[dict] = []
    if a.runs:
        for d in a.runs:
            row = run_row(RunData(Path(d)))
            if row:
                rows.append(row)
    else:
        rows = run_rows(ROOT / "runs")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(rows))
    print(f"{out} ({len(rows)} runs)")
    return 0
