"""The grid runner: one probe run per load condition, with the load generator started before the
run and stopped after it (milestone 4's second axis).

A condition names a load and its level: "idle" (no load), "cpu50" (CPU at 50% duty), "mem80"
(80% of RAM touched and held), "io500" (500 MB/s of file reads), "gpu50" (GPU at 50% duty via
gpuload). Each load is its own process speaking the record protocol in load/loadlib.py: it
writes load_start, then load_ready once it is at steady state, heartbeats, and load_end. The
runner waits for load_ready before starting the probe (a probe must not start against a
half-filled machine), stops the load afterwards, and copies its records into the run
directory as load.jsonl, so "50%" can be checked instead of assumed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from env.sampler import meminfo
from run.plans import Plan, PlanOptions, make_plan
from run.runner import Run, RunOptions, new_run_id


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out

ROOT = Path(__file__).resolve().parents[1]
LOADS = ROOT / "load"
BUILD_CUDA = ROOT / "build" / "cuda"


@dataclass
class GridOptions:
    out_root: Path = Path("runs")
    conditions: list[str] = field(default_factory=lambda: ["idle"])
    settle_s: float = 2.0        # after load_ready, before the probe
    ready_timeout_s: float = 300.0  # memfill can take a while to touch 90% of RAM
    sample_interval_s: float = 1.0
    pre_roll_s: float = 2.0      # sampler baseline before the first step
    post_roll_s: float = 3.0    # recovery samples after the last step
    probe_args: list[str] = field(default_factory=list)
    scratch_dir: Path | None = None


def load_argv(condition: str) -> list[str] | None:
    """A condition name to a load-generator command line; None means no load."""
    if condition == "idle":
        return None
    m = re.fullmatch(r"(cpu|mem|io|gpu)(\d+(?:\.\d+)?)", condition)
    if m is None:
        raise ValueError(f"unknown load condition {condition!r} "
                         "(have: idle, cpu50, mem80, io500, gpu50)")
    kind, level = m.group(1), float(m.group(2))
    if kind == "cpu":
        return [sys.executable, str(LOADS / "cpuload.py"), "--duty", str(level)]
    if kind == "mem":
        return [sys.executable, str(LOADS / "memfill.py"), "--fraction", str(level / 100)]
    if kind == "io":
        return [sys.executable, str(LOADS / "ioload.py"), "--mbps", str(level)]
    return [str(BUILD_CUDA / "gpuload"), "--duty", str(level)]


class LoadProcess:
    """A load generator subprocess: its stdout is drained into a record list as it runs."""

    def __init__(self, argv: list[str]):
        self.argv = argv
        self.records: list[dict] = []
        self.bad_lines = 0
        # The load watches this pid: if the runner dies, the load notices and exits on its own.
        self.proc = subprocess.Popen(argv + ["--parent-pid", str(os.getpid())],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, bufsize=1)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    self.records.append(rec)
                    continue
            except json.JSONDecodeError:
                pass
            self.bad_lines += 1

    def wait_ready(self, timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if any(r.get("type") in ("load_ready", "load_error") for r in self.records):
                return not any(r.get("type") == "load_error" for r in self.records)
            if self.proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self._thread.join(timeout=5)


def load_check(condition: str, run_dir: Path) -> dict | None:
    """Load-condition invariance (docs/crosscheck.md, later candidates): the requested level and
    the generator's own achieved record must agree, and the sampler must show at least the target
    (it also sees the probe's own work, so it bounds the ambient load from below only). Woven
    into the manifest so the report can flag a run whose load was not what its label says. pass
    is None when nothing is determinable; only determinable failures fail."""
    if condition == "idle":
        return None
    m = re.fullmatch(r"(cpu|mem|io|gpu)(\d+(?:\.\d+)?)", condition)
    kind, target = m.group(1), float(m.group(2))
    target = target / 100.0 if kind in ("cpu", "mem", "gpu") else target
    heartbeats = []
    lp = run_dir / "load.jsonl"
    if lp.exists():
        for line in lp.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "heartbeat":
                heartbeats.append(rec.get("achieved") or {})
    self_reported: float | None = None
    if heartbeats:
        key = {"cpu": "duty", "gpu": "duty", "io": "mbps", "mem": "resident_bytes"}.get(kind)
        vals = [h.get(key) for h in heartbeats if h.get(key) is not None]
        if vals:
            self_reported = sum(vals) / len(vals)
            if kind == "mem":
                total = (meminfo().get("MemTotal_kib") or 0) * 1024
                self_reported = self_reported / total if total else None
    # The sampler's independent view of the same load.
    sampler: float | None = None
    samples = load_jsonl(run_dir / "samples.jsonl")
    if kind == "cpu":
        vals = [s.get("host", {}).get("cpu_busy_frac") for s in samples]
        vals = [v for v in vals if v is not None]
        sampler = sum(vals) / len(vals) if vals else None
    elif kind == "gpu":
        vals = [g.get("util_gpu_pct") for s in samples for g in (s.get("gpu") or [])
                if isinstance(g.get("util_gpu_pct"), (int, float))]
        sampler = sum(vals) / len(vals) / 100.0 if vals else None
    elif kind == "mem":
        mem = [s.get("host", {}).get("mem") or {} for s in samples]
        avail = [m_.get("MemAvailable_kib") for m_ in mem if m_.get("MemAvailable_kib")]
        total = [m_.get("MemTotal_kib") for m_ in mem if m_.get("MemTotal_kib")]
        if avail and total and total[0]:
            sampler = 1.0 - min(avail) / total[0]
    def agrees(delivered: float | None) -> bool | None:
        if delivered is None:
            return None
        return abs(delivered - target) <= max(0.15 * target, 0.05)

    def at_least(delivered: float | None) -> bool | None:
        # The sampler sees the ambient load PLUS the probe's own work, so it can only confirm the
        # load existed (at least the target), never that it stopped exactly at the target.
        if delivered is None:
            return None
        return delivered >= target * 0.85
    checks = {"self_reported": agrees(self_reported), "sampler_at_least": at_least(sampler)}
    determinable = [v for v in checks.values() if v is not None]
    return {"requested": target, "self_reported": self_reported, "sampler": sampler,
            "pass": all(determinable) if determinable else None,
            "checks": checks}


def run_condition(condition: str, plan: Plan, opts: GridOptions,
                  run_id: str | None = None) -> Path:
    """One probe run under one load condition. The load starts, reports ready, the probe runs,
    the load stops, and its records land in the run directory as load.jsonl."""
    argv = load_argv(condition)
    run_id = run_id or new_run_id(plan.probe, plan.backend, condition)
    run = Run(plan, RunOptions(out_root=opts.out_root, label=condition,
                                load={"name": condition}, sample_interval_s=opts.sample_interval_s,
                                pre_roll_s=opts.pre_roll_s, post_roll_s=opts.post_roll_s),
              run_id)
    load: LoadProcess | None = None
    ready = True
    if argv is not None:
        load = LoadProcess(argv)
        ready = load.wait_ready(opts.ready_timeout_s)
        if not ready:
            print(f"[grid] load {condition!r} never became ready; running without it",
                  file=sys.stderr)
            load.stop()
            load = None
    try:
        if load is not None:
            time.sleep(opts.settle_s)
        run.execute()
    finally:
        if load is not None:
            load.stop()
        if run.dir.is_dir():
            with open(run.dir / "load.jsonl", "w") as f:
                for rec in load.records if load else []:
                    f.write(json.dumps(rec) + "\n")
                if argv is not None and not ready:
                    f.write(json.dumps({"v": 1, "t_ns": time.monotonic_ns(), "src": "load",
                                        "type": "load_error",
                                        "message": "load never reported ready"}) + "\n")
        mp = run.dir / "manifest.json"
        if mp.exists():
            manifest = json.loads(mp.read_text())
            manifest["load_check"] = load_check(condition, run.dir)
            mp.write_text(json.dumps(manifest, indent=1) + "\n")
    return run.dir


def run_grid(probe: str, backend: str, opts: GridOptions) -> list[Path]:
    """Every condition in sequence; the report reads the run directories side by side."""
    dirs: list[Path] = []
    for condition in opts.conditions:
        # The run directory is where the plan puts its scratch files, so it is fixed here and
        # passed to both make_plan and run_condition.
        run_id = new_run_id(probe, backend, condition)
        plan_opts = PlanOptions(extra=list(opts.probe_args), scratch_dir=opts.scratch_dir)
        plan = make_plan(probe, backend, opts.out_root / run_id, plan_opts)
        dirs.append(run_condition(condition, plan, opts, run_id=run_id))
        print(f"[grid] {probe} under {condition}: {dirs[-1]}", file=sys.stderr)
    return dirs
