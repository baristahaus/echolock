"""The suite runner: every probe x backend whose binary exists, under each load condition,
one combined report.

Reuses run/grid.py's condition machinery (run_condition: the load generator starts, reports
ready, the probe runs, the load stops, its records land in load.jsonl) -- this module only
decides which (probe, backend, condition) cells to run and collects their run directories.

Cell states:
  ok / verify_failed / error / timeout / incomplete  -- the run's manifest status; the cell
      shows the failed-check count and, when the probe produced numbers, its headline metric
  not built                                           -- the binary does not exist: a state,
      not a failure (the suite's job is to report coverage, not to punish an unbuilt backend)
  error: <why>                                        -- the run could not even start (e.g. the
      runner itself failed); a clean cell, never a crash
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from run.plans import BACKENDS, BUILD, PROBES, PlanOptions, make_plan
from run.grid import GridOptions, run_condition
from run.runner import new_run_id
from report.markdown import RunData
from report.platforms import HEADLINE
from report.tags import run_metrics


def _headline(run_dir: Path, probe: str, backend: str) -> str:
    keys = HEADLINE.get(probe, [])
    if isinstance(keys, dict):
        keys = keys.get(backend, [])
    try:
        metrics = run_metrics(RunData(run_dir))
    except Exception:
        return ""
    for key in keys:
        v = metrics.get(key, {}).get("value")
        if v is not None:
            return f"{v:.1f} {metrics[key].get('unit', '')}".strip()
    return ""


def suite_matrix(probes: tuple[str, ...] = PROBES,
                 backends: tuple[str, ...] = BACKENDS,
                 conditions: tuple[str, ...] = ("idle",),
                 out_root: Path = Path("runs"),
                 build_root: Path | None = None,
                 probe_args: list[str] | None = None,
                 settle_s: float = 2.0,
                 ready_timeout_s: float = 300.0,
                 sample_interval_s: float = 1.0) -> dict[tuple[str, str, str], dict]:
    """Run every (probe, backend) with a binary under each condition; return the cells.

    build_root defaults to the real build tree; tests point it at a tree of stand-in probes.
    make_plan supplies the step structure (timeouts, drop-caches rules, file_page scratch);
    the suite swaps in the executable so an alternate build tree is honoured.
    """
    build_root = build_root or BUILD
    probe_args = list(probe_args or [])
    cells: dict[tuple[str, str, str], dict] = {}
    for probe in probes:
        for backend in backends:
            exe = build_root / backend / probe
            if not (exe.exists() and exe.stat().st_mode & 0o111):
                for cond in conditions:
                    cells[(probe, backend, cond)] = {"state": "not built"}
                continue
            for cond in conditions:
                cell: dict = {"state": "error"}
                run_dir: Path | None = None
                try:
                    plan = make_plan(probe, backend, out_root / "suite-placeholder",
                                     PlanOptions(extra=probe_args))
                    for step in plan.steps:
                        step.argv[0] = str(exe)
                    opts = GridOptions(out_root=out_root, conditions=[cond], settle_s=settle_s,
                                       ready_timeout_s=ready_timeout_s,
                                       sample_interval_s=sample_interval_s,
                                       probe_args=probe_args)
                    run_dir = run_condition(cond, plan, opts, new_run_id(probe, backend, cond))
                    manifest = json.loads((run_dir / "manifest.json").read_text())
                    failed = sum(s.get("failed_checks", 0) for s in manifest.get("steps", []))
                    cell = {"state": manifest.get("status", "error"), "run_dir": str(run_dir),
                            "failed_checks": failed,
                            "headline": _headline(run_dir, probe, backend)}
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    cell = {"state": f"error: {type(e).__name__}: {e}"}
                cells[(probe, backend, cond)] = cell
    return cells


def _cell_text(cell: dict) -> str:
    state = cell.get("state", "error")
    if state == "not built":
        return "not built"
    extra = []
    if cell.get("failed_checks"):
        extra.append(f"{cell['failed_checks']} failed checks")
    if cell.get("headline"):
        extra.append(cell["headline"])
    return state + (f" ({', '.join(extra)})" if extra else "")


def render_suite_report(cells: dict[tuple[str, str], dict],
                        conditions: tuple[str, ...],
                        probes: tuple[str, ...],
                        backends: tuple[str, ...]) -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    lines = ["# Suite report", "",
             f"Generated by `./echolock suite` on {ts}; do not edit by hand.", "",
             "Every (probe, backend) cell whose binary exists, run under each load condition "
             "via the grid machinery (run/grid.py). `not built` is a state, not a failure.", ""]
    for cond in conditions:
        lines += [f"## condition: {cond}", "",
                  "| probe | " + " | ".join(backends) + " |",
                  "|---|" + "---|" * len(backends)]
        for probe in probes:
            row = [probe]
            for backend in backends:
                row.append(_cell_text(cells.get((probe, backend, cond), {"state": "not built"})))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    run_dirs = sorted({c["run_dir"] for c in cells.values() if c.get("run_dir")})
    lines += ["## run directories", ""]
    lines += [f"- `{d}`" for d in run_dirs] or ["- none"]
    lines.append("")
    return "\n".join(lines)


def run_suite(probes: tuple[str, ...] = PROBES,
              backends: tuple[str, ...] = BACKENDS,
              conditions: tuple[str, ...] = ("idle",),
              out_root: Path = Path("runs"),
              build_root: Path | None = None,
              probe_args: list[str] | None = None) -> Path:
    """Run the suite and write the combined report; returns the report path."""
    cells = suite_matrix(probes=probes, backends=backends, conditions=conditions,
                         out_root=out_root, build_root=build_root, probe_args=probe_args)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = out_root / f"suite-{ts}.md"
    out.write_text(render_suite_report(cells, conditions, probes, backends))
    return out
