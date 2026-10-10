#!/usr/bin/env python3
"""platforms: one table per probe, one row per platform, from finished run directories.

The publication layer of docs/crosscheck.md: the same metric, measured the same way, across the
machines and backends echolock has run on. Rows appear as runs happen; nothing is hand-written.

  ./echolock platforms                 # scan runs/
  ./echolock platforms RUN_DIR...      # only these
  ./echolock platforms --out docs/platforms.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report.markdown import RunData, fmt                      # noqa: E402


def measures(run: RunData) -> list[dict]:
    """The summary records in the order the probe wrote them."""
    return [r for r in run.of_type("measure") if r.get("rep") is None]


def platform_name(run: RunData) -> str:
    env = run.env or {}
    nvidia = env.get("nvidia") or {}
    devices = (nvidia.get("nvml") or {}).get("devices") or []
    name = devices[0].get("name") if devices else (env.get("host") or {}).get("hostname", "?")
    backend = run.manifest.get("backend", "?")
    return f"{name} [{backend}]"


def headline(run: RunData) -> dict[str, str]:
    """One to four headline values per probe: what you would quote about this platform."""
    probe = run.manifest.get("probe")
    out: dict[str, str] = {}
    if run.manifest.get("status") != "ok":
        return {"status": run.manifest.get("status") or "?"}
    m = measures(run)
    def val(step: str, name: str, tags: dict | None = None) -> str:
        for r in m:
            have = r.get("tags") or {}
            if r.get("step") == step and r.get("name") == name and \
                    (tags is None or all(have.get(k) == v for k, v in tags.items())):
                return fmt(r["value"], r.get("unit", ""))
        return "—"
    if probe == "memory":
        out["device read"] = val("memory", "gpu_read", {"contention": "none"})
        out["copy h2d"] = val("memory", "copy_payload", {"direction": "h2d"})
    elif probe == "pointer_chase":
        out["L1 hop seq"] = val("pointer_chase", "chase_hop_latency",
                                {"pattern": "seq", "kind": "device"})
        if out["L1 hop seq"] == "—":
            out["L1 hop seq"] = val("pointer_chase", "chase_hop_latency",
                                    {"pattern": "seq", "kind": "vulkan_device_local"})
        out["random worst"] = val("pointer_chase", "chase_hop_latency",
                                  {"pattern": "random", "kind": "device"})
    elif probe == "mma":
        for r in m:
            if r.get("name") == "mma_throughput" and (r.get("tags") or {}).get("per_rep"):
                fmt_tag = r["tags"].get("format", "?")
                if fmt_tag not in out:
                    out[fmt_tag] = fmt(r["value"], r.get("unit", ""))
    elif probe == "sync":
        out["return lag"] = val("sync", "sync_return_lag")  # first schedule wins
    elif probe == "clock":
        out["clock drop"] = val("clock", "clock_drop_ratio")
        out["declared/derived"] = val("clock", "declared_over_derived_ratio")
    elif probe == "constants":
        checks = run.of_type("check")
        ok = sum(1 for c in checks if c.get("pass"))
        out["source agreement"] = f"{ok}/{len(checks)} checks"
    if not out:
        out["status"] = "ok"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="platforms", description=__doc__)
    ap.add_argument("runs", nargs="*", help="run directories; default: every run under runs/")
    ap.add_argument("--out", default=str(ROOT / "docs" / "platforms.md"))
    args = ap.parse_args(argv)

    if args.runs:
        dirs = [Path(d) for d in args.runs]
    else:
        runs_root = ROOT / "runs"
        dirs = sorted(p.parent for p in runs_root.glob("*/manifest.json")) if runs_root.is_dir() else []
    if not dirs:
        print("no run directories found", file=sys.stderr)
        return 1

    # platform x probe -> {headline key: value}; the newest run wins per cell.
    table: dict[tuple[str, str], dict[str, str]] = {}
    for d in dirs:
        run = RunData(d)
        if run.problem and not run.records:
            continue
        platform = platform_name(run)
        probe = run.manifest.get("probe", "?")
        table[(platform, probe)] = headline(run)

    probes = sorted({p for _, p in table})
    columns: list[str] = []
    for p in probes:
        keys = sorted({k for (plat, pr), cell in table.items() if pr == p for k in cell})
        columns += [(p, k) for k in keys]

    lines = ["# echolock platforms", "",
             "Generated from run directories (`./echolock platforms`); the newest run of each",
             "(platform, probe) pair wins each cell. Regenerated, never hand-written.", ""]
    lines.append("| Platform | " + " | ".join(f"{p} {k}" for p, k in columns) + " |")
    lines.append("|---|" + "---|" * len(columns))
    for platform in sorted({plat for plat, _ in table}):
        cells = [table.get((platform, p), {}).get(k, "—") for p, k in columns]
        lines.append(f"| {platform} | " + " | ".join(cells) + " |")
    lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
