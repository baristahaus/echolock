"""Runs one probe plan with the environment recorded around it.

A run directory holds:
  manifest.json   what ran, per-step exit status and timing, overall status, kernel-event counts
  env.json        static environment snapshot (env/capture.py), taken before the first step
  probe.jsonl     probe records, each tagged with run_id and step; plus runner step records
  samples.jsonl   1 s environment samples (env/sampler.py), from before the first step to
                  `post_roll_s` after the last
  kernel.jsonl    kernel log events (Xid, OOM, hangs) that appeared during the run
  stderr.log      probe stderr, prefixed by step

The load axis is recorded in the manifest ("load"); milestone 4's grid runner starts the load
generator around this same call.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from env.capture import capture
from env.kmsg import KernelLogWatcher
from env.sampler import Sampler

from .plans import Plan, Step

# Overall status, worst first. A run's status is the worst of its steps'.
STATUS_ORDER = ("error", "timeout", "verify_failed", "incomplete", "ok")


class RunInterrupted(Exception):
    """Raised by the SIGTERM/SIGHUP handler in place of the default immediate exit.

    Ctrl-C needs no handler: Python already raises KeyboardInterrupt on the main thread. Both then
    take the same path, which stops the step's process group and writes the manifest -- a step runs
    in its own session (`start_new_session`), so nothing else would keep it off the GPU once this
    process is gone.
    """


@dataclass
class StepResult:
    name: str
    argv: list[str]
    returncode: int | None
    status: str
    start_ns: int
    end_ns: int
    records: int = 0
    non_json_lines: int = 0
    failed_checks: int = 0
    killed: bool = False
    unkillable: bool = False
    interrupted: bool = False
    drop_caches: str | None = None


@dataclass
class RunOptions:
    out_root: Path = Path("runs")
    label: str = "idle"                 # the load condition this run was taken under
    load: dict = field(default_factory=lambda: {"name": "idle"})
    sample_interval_s: float = 1.0
    pre_roll_s: float = 2.0             # samples before the first step: the idle baseline
    post_roll_s: float = 3.0            # samples after the last step: recovery
    drop_caches: bool = False
    capture_env: bool = True


def drop_page_cache() -> str:
    """Drops clean page cache so a full cache from a previous step can't fail allocations."""
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return "ok"
    except OSError:
        pass
    p = subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                       capture_output=True, text=True)
    return "ok (sudo)" if p.returncode == 0 else "denied"


def step_status(returncode: int | None, timed_out: bool, saw_end: str | None) -> str:
    if timed_out:
        return "timeout"
    if returncode == 3 or saw_end == "verify_failed":
        return "verify_failed"
    if returncode != 0:
        return "error"
    return "ok" if saw_end == "ok" else "incomplete"


def worst(statuses: list[str]) -> str:
    if not statuses:
        return "incomplete"
    return min(statuses, key=STATUS_ORDER.index)


def new_run_id(probe: str, backend: str, label: str) -> str:
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{probe}-{backend}-{label}"


def interrupt_exit_code(exc: BaseException) -> int:
    """The usual 128+signal convention, so a caller can tell Ctrl-C from `kill`."""
    return 130 if isinstance(exc, KeyboardInterrupt) else 143


class Run:
    def __init__(self, plan: Plan, opts: RunOptions, run_id: str | None = None):
        self.plan = plan
        self.opts = opts
        self.run_id = run_id or new_run_id(plan.probe, plan.backend, opts.label)
        self.dir = opts.out_root / self.run_id
        self._lock = threading.Lock()
        self._probe_out = None                     # probe.jsonl, while the run holds it open
        self._results: list[StepResult] = []   # filled by _run_step, so an interrupt still records
        self._step_running: str | None = None
        self.interrupted: str | None = None    # set when a signal ended the run

    def execute(self) -> dict:
        self.dir.mkdir(parents=True, exist_ok=False)
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic_ns()
        self._results = []
        watcher: KernelLogWatcher | None = None
        sampler: Sampler | None = None
        try:
            if self.opts.capture_env:            # inside the try: a broken collector still leaves
                (self.dir / "env.json").write_text(json.dumps(capture(), indent=1))
            with self._terminal_signals(), \
                 open(self.dir / "probe.jsonl", "w") as probe_out, \
                 open(self.dir / "samples.jsonl", "w") as samples_out, \
                 open(self.dir / "kernel.jsonl", "w") as kernel_out, \
                 open(self.dir / "stderr.log", "w") as stderr_out:
                self._probe_out = probe_out
                watcher = KernelLogWatcher(out=kernel_out).start()
                sampler = Sampler(samples_out, self.opts.sample_interval_s).start()
                try:
                    time.sleep(self.opts.pre_roll_s)
                    for step in self.plan.steps:
                        res = self._run_step(step, stderr_out)
                        if step.required and res.status != "ok":
                            self._runner_record("abort", step=step.name,
                                                reason=f"required step ended {res.status}")
                            break
                    time.sleep(self.opts.post_roll_s)
                except BaseException as e:                    # Ctrl-C, SIGTERM, MemoryError, ...
                    self.interrupted = type(e).__name__ if not isinstance(e, RunInterrupted) \
                        else str(e)
                    if self._probe_out is not None and not self._probe_out.closed:
                        self._runner_record("abort", step=self._step_running,
                                            reason=f"run interrupted ({self.interrupted})")
                    raise
                finally:
                    sampler.stop()
                    watcher.stop()
                    for path in self.plan.cleanup:
                        try:
                            path.unlink()
                        except OSError:
                            pass
        finally:
            # Always true, however the run ended: a run directory left without a manifest is a run
            # directory no report can read.
            manifest = self._manifest(t0, started_utc, watcher, sampler)
        return manifest

    def _manifest(self, t0: int, started_utc: str, watcher: KernelLogWatcher | None,
                  sampler: Sampler | None) -> dict:
        """Writes manifest.json from whatever has happened so far, and returns it."""
        kernel = watcher.summary() if watcher else {"available": False,
                                                   "error": "watcher never started",
                                                   "during_run": {}, "before_run": {},
                                                   "before_run_last": [], "xids_during_run": []}
        statuses = [r.status for r in self._results]
        if self.interrupted:
            statuses.append("incomplete")       # the remaining steps did not run
        elif len(self._results) < len(self.plan.steps):
            statuses.append("incomplete")
        manifest = {
            "v": 1,
            "run_id": self.run_id,
            "probe": self.plan.probe,
            "backend": self.plan.backend,
            "label": self.opts.label,
            "load": self.opts.load,
            "status": worst(statuses),
            "started_utc": started_utc,
            "t_start_ns": t0,
            "t_end_ns": time.monotonic_ns(),
            "sample_interval_s": self.opts.sample_interval_s,
            "samples": sampler.count if sampler else 0,
            "steps": [asdict(r) for r in self._results],
            "planned_steps": [s.name for s in self.plan.steps],
            "kernel_log": kernel,
        }
        notes = []
        # A GPU fault during the run makes every number from it suspect.
        if kernel["during_run"].get("xid") or kernel["during_run"].get("sxid"):
            notes.append("Xid logged during the run")
        if self.interrupted:
            notes.append(f"interrupted by {self.interrupted}; "
                         f"{len(self._results)} of {len(self.plan.steps)} steps ran")
        if notes:
            manifest["status_note"] = "; ".join(notes)
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return manifest

    @contextmanager
    def _terminal_signals(self):
        """Turns `kill <pid>` and a hangup into RunInterrupted, so steps are stopped and the
        manifest is written. Handlers are installed only where the disposition is still the
        default, and restored on the way out."""
        def handler(signum, _frame):
            raise RunInterrupted(signal.Signals(signum).name)

        saved = {}
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                if signal.getsignal(sig) is signal.SIG_DFL:
                    signal.signal(sig, handler)
                    saved[sig] = signal.SIG_DFL
            except (OSError, ValueError):        # signals only work on the main thread
                pass
        try:
            yield
        finally:
            for sig, old in saved.items():
                try:
                    signal.signal(sig, old)
                except (OSError, ValueError):
                    pass

    def _runner_record(self, type_: str, **fields) -> None:
        rec = {"v": 1, "t_ns": time.monotonic_ns(), "src": "runner", "type": type_,
               "run_id": self.run_id, **fields}
        with self._lock:
            self._probe_out.write(json.dumps(rec) + "\n")
            self._probe_out.flush()

    def _run_step(self, step: Step, stderr_out) -> StepResult:
        dropped = drop_page_cache() if self.opts.drop_caches else None
        self._runner_record("step_start", step=step.name, argv=step.argv, drop_caches=dropped)
        start = time.monotonic_ns()
        res = StepResult(step.name, step.argv, None, "error", start, start, drop_caches=dropped)
        self._results.append(res)         # registered now, so an interrupt still reports this step
        self._step_running = step.name
        try:
            proc = subprocess.Popen(step.argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1, start_new_session=True)
        except OSError as e:
            res.end_ns = time.monotonic_ns()
            self._runner_record("step_end", step=step.name, status="error",
                                message=f"cannot start: {e}")
            return res

        saw_end: list[str | None] = [None]

        def pump_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        raise ValueError
                except ValueError:
                    res.non_json_lines += 1
                    rec = {"v": 1, "t_ns": time.monotonic_ns(), "src": "runner",
                           "type": "stdout_text", "text": line}
                rec["run_id"] = self.run_id
                rec["step"] = step.name
                res.records += 1
                if rec.get("type") == "check" and rec.get("pass") is False:
                    res.failed_checks += 1
                if rec.get("type") == "probe_end":
                    saw_end[0] = rec.get("status")
                with self._lock:
                    self._probe_out.write(json.dumps(rec) + "\n")
                    self._probe_out.flush()

        def pump_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                with self._lock:
                    stderr_out.write(f"[{step.name}] {line}")
                    stderr_out.flush()

        threads = [threading.Thread(target=pump_stdout, daemon=True),
                   threading.Thread(target=pump_stderr, daemon=True)]
        for t in threads:
            t.start()
        timed_out = False
        interrupted = False
        step_exc: BaseException | None = None
        try:
            proc.wait(timeout=step.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            res.killed = True
            self._kill(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Stuck in the driver (uninterruptible sleep): record it and move on.
                res.unkillable = True
        except BaseException as e:                    # KeyboardInterrupt, RunInterrupted
            # The step is in its own session, so the signal that reached us did not reach it: it
            # would go on holding the GPU after this process is gone. Stop its process group first.
            interrupted, step_exc = True, e
            res.killed = True
            res.interrupted = True
            self._kill(proc)
        for t in threads:
            t.join(timeout=5)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()
        res.end_ns = time.monotonic_ns()
        res.returncode = proc.returncode
        res.status = "error" if interrupted else step_status(proc.returncode, timed_out, saw_end[0])
        self._runner_record("step_end", step=step.name, status=res.status,
                            returncode=proc.returncode, duration_s=(res.end_ns - start) / 1e9,
                            timed_out=timed_out, unkillable=res.unkillable,
                            interrupted=interrupted)
        if interrupted:
            raise step_exc                        # re-raised here, outside the handler, because the
            # teardown above has to run whatever the interrupt was
        return res

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                continue
