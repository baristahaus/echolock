"""What one probe run executes: an ordered list of process invocations (steps).

A probe whose parts must not share a process gets one step per part: the CUDA device schedule is
fixed per process (sync), and a cold page-fault phase that hangs must be killable on its own
without losing the phases before it (file_page).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


@dataclass
class Step:
    name: str
    argv: list[str]
    timeout_s: float = 600.0
    # Steps whose failure makes the rest meaningless stop the run (e.g. file_page prep).
    required: bool = False
    # May the page cache be dropped before this step? False for steps that manage page residency
    # themselves (file_page's resident and evict phases): a drop behind their back would evict the
    # pages they are about to measure.
    drop_caches: bool = False


@dataclass
class Plan:
    probe: str
    backend: str
    steps: list[Step]
    cleanup: list[Path] = field(default_factory=list)


@dataclass
class PlanOptions:
    extra: list[str] = field(default_factory=list)  # passed to every step's probe
    scratch_dir: Path | None = None                # file_page: where the scratch file lives
    pin_cpus: list[int] = field(default_factory=list)  # sync: extra pinned blocking runs
    evict_file: str | None = None                  # file_page: evict by reading this file


PROBES = ("memory", "file_page", "sync")
BACKENDS = ("cuda",)


def binary(backend: str, probe: str) -> Path:
    return BUILD / backend / probe


def make_plan(probe: str, backend: str, run_dir: Path, opts: PlanOptions) -> Plan:
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} not implemented (have: {', '.join(BACKENDS)})")
    if probe not in PROBES:
        raise ValueError(f"unknown probe {probe!r} (have: {', '.join(PROBES)})")
    exe = str(binary(backend, probe))

    if probe == "memory":
        return Plan(probe, backend,
                    [Step("memory", [exe, *opts.extra], timeout_s=900, drop_caches=True)])

    if probe == "sync":
        steps = [Step(f"schedule={s}", [exe, "--schedule", s, *opts.extra], timeout_s=600,
                      drop_caches=True)
                 for s in ("blocking", "yield", "spin", "auto")]
        steps += [Step(f"schedule=blocking,cpu={c}",
                       ["taskset", "-c", str(c), exe, "--schedule", "blocking", *opts.extra],
                       timeout_s=600, drop_caches=True) for c in opts.pin_cpus]
        return Plan(probe, backend, steps)

    # file_page. The scratch file must be on a page-cache-backed filesystem (not tmpfs), or
    # eviction is impossible and the cold phases measure nothing.
    scratch = (opts.scratch_dir or run_dir) / "file_page.scratch"
    common = ["--file", str(scratch), *opts.extra]
    evict = ["--evict", opts.evict_file] if opts.evict_file else []
    steps = [Step("prep", [exe, "--phase", "prep", *common], required=True, drop_caches=True),
             Step("warm", [exe, "--phase", "warm", *common], required=True, drop_caches=True)]
    # The resident phases measure pages the warm step put in the cache: no drop before them.
    for phase in ("res-seq", "res-rows"):
        steps.append(Step(phase, [exe, "--phase", phase, *common]))
    # Evict tests its own eviction method; a drop right before it would measure a no-op.
    steps.append(Step("evict", [exe, "--phase", "evict", *common, *evict], timeout_s=900))
    # The cold phases want cold pages anyway, so a drop before them is a second chance at
    # eviction if the evict step's method silently failed.
    for phase in ("cold-serial", "cold-rows"):
        steps.append(Step(phase, [exe, "--phase", phase, *common], drop_caches=True))
    # res-rows-again re-reads the rows the cold phases faulted in: those pages must stay resident.
    steps.append(Step("res-rows-again", [exe, "--phase", "res-rows-again", *common]))
    return Plan(probe, backend, steps, cleanup=[scratch])
