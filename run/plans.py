"""What one probe run executes: an ordered list of process invocations (steps).

A probe whose parts must not share a process gets one step per part: the CUDA device schedule is
fixed per process (sync), and a cold page-fault phase that hangs must be killable on its own
without losing the phases before it (file_page).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
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


PROBES = ("memory", "file_page", "sync", "clock", "pointer_chase", "mma", "constants")
BACKENDS = ("cuda", "vulkan", "host")
# What each backend implements; make_plan validates the pair, so the CLI can list the union.
PROBES_BY_BACKEND = {
    "cuda": ("memory", "file_page", "sync", "clock", "pointer_chase", "mma"),
    "vulkan": ("memory", "file_page", "sync", "clock", "pointer_chase", "mma"),
    "host": ("constants",),
}


def binary(backend: str, probe: str) -> Path:
    if backend == "host":
        # The host backend is a Python script: it needs no build step and no GPU API.
        return ROOT / "backends" / "host" / f"{probe}.py"
    return BUILD / backend / probe


def make_plan(probe: str, backend: str, run_dir: Path, opts: PlanOptions) -> Plan:
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} not implemented (have: {', '.join(BACKENDS)})")
    have = PROBES_BY_BACKEND[backend]
    if probe not in have:
        raise ValueError(f"probe {probe!r} not implemented for backend {backend!r} "
                         f"(have: {', '.join(have)})")
    exe = str(binary(backend, probe))

    if probe == "constants":
        # Host-side Python: every source is best-effort and none touch the GPU API.
        return Plan(probe, backend, [Step("constants", [sys.executable, exe, *opts.extra],
                                          timeout_s=120)])

    if probe == "memory":
        return Plan(probe, backend,
                    [Step("memory", [exe, *opts.extra], timeout_s=900, drop_caches=True)])

    if probe == "clock":
        # One process: the idle measurement, the loaded trace and the load kernel share a context,
        # so the derived clock and the load it runs under are one observation.
        return Plan(probe, backend,
                    [Step("clock", [exe, *opts.extra], timeout_s=600, drop_caches=True)])

    if probe == "mma":
        return Plan(probe, backend,
                    [Step("mma", [exe, *opts.extra], timeout_s=600, drop_caches=True)])

    if probe == "pointer_chase":
        # The probe manages residency itself (resident walks then an evicted cold walk), so no
        # drop before it -- that would evict the pages its resident walks are about to measure.
        # The CUDA probe takes a scratch file it maps for the resident-vs-evicted walk; the
        # Vulkan probe measures buffers only (no file residency path), so no scratch file.
        if backend == "vulkan":
            return Plan(probe, backend,
                        [Step("pointer_chase", [exe, *opts.extra], timeout_s=900)])
        scratch = (opts.scratch_dir or run_dir) / "pointer_chase.scratch"
        common = ["--file", str(scratch), *opts.extra]
        return Plan(probe, backend,
                    [Step("pointer_chase", [exe, *common], timeout_s=900)], cleanup=[scratch])

    if probe == "sync":
        if backend == "vulkan":
            # The Vulkan probe has no SCHEDULE axis (no device-side schedule flag on a queue
            # submission): one step measures the one behavior.
            return Plan(probe, backend,
                        [Step("sync", [exe, *opts.extra], timeout_s=900, drop_caches=True)])
        steps = [Step(f"schedule={s}", [exe, "--schedule", s, *opts.extra], timeout_s=600,
                      drop_caches=True)
                 for s in ("blocking", "yield", "spin", "auto")]
        steps += [Step(f"schedule=blocking,cpu={c}",
                       ["taskset", "-c", str(c), exe, "--schedule", "blocking", *opts.extra],
                       timeout_s=600, drop_caches=True) for c in opts.pin_cpus]
        return Plan(probe, backend, steps)

    if backend == "vulkan":
        # The Vulkan file_page probe runs all phases in one process (it measures the import path
        # on anonymous memory; RADV rejects file-backed imports, so no scratch file is needed).
        return Plan(probe, backend,
                    [Step("file_page", [exe, *opts.extra], timeout_s=900, drop_caches=True)])

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
