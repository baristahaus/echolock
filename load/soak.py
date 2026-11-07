"""Thermal soak: heat every source the machine offers until the temperatures stop moving,
so a grid condition means "hot and at steady state" rather than "getting hot while we
measure".

Sources:
- CPU: every core at 100 % duty, run by cpuload.py as a child process. The child owns the
  threads and its /proc-measured duty; this process is the supervisor and relays the duty
  into its own records.
- GPU: build/cuda/gpuload --duty 100, started only when the binary exists AND the NVIDIA
  driver is present (/proc/driver/nvidia). Otherwise the gpu section is skipped and the
  evidence is in load_start's intended.

load_ready fires after the longer of the minimum soak (--soak-min, minutes, default 5) and
a temperature plateau: the rise over the last --plateau-window s (default 60) has stayed
under --plateau-rise C (default 1), sampled every --temp-int s (default 2). Temperature is
the max of what env/amdgpu.py reads (amdgpu hwmon temps) and the CPU thermal zones
(env/sampler.py); a machine with no temperature source at all has nothing to plateau, so
the minimum soak alone gates ready. Heartbeats every ~5 s carry achieved
{cpu_duty, gpu_duty or "skipped", temp_c}.

Usage: soak.py [--soak-min 5] [--parent-pid PID] [--deadline-min 120]
               [--plateau-window 60] [--plateau-rise 1] [--temp-int 2] [--heartbeat-int 5]
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadlib import (Stop, emit, install_stop, make_parent_check, parent_pid_of,
                     parse_args)  # noqa: E402

from env import amdgpu, sampler  # noqa: E402  (loadlib put the repo root on sys.path)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPULOAD = os.path.join(ROOT, "build", "cuda", "gpuload")
NVIDIA = "/proc/driver/nvidia"

USAGE = ("usage: soak.py [--soak-min 5] [--parent-pid PID] [--deadline-min 120] "
         "[--plateau-window 60] [--plateau-rise 1] [--temp-int 2] [--heartbeat-int 5]")


class Child:
    """A load-generator child: its stdout is drained, and the latest achieved duty is
    remembered for the parent's records. The child watches this process's pid, so it
    cannot outlive us even if we die hard."""

    def __init__(self, name: str, argv: list[str]):
        self.name = name
        self.duty: float | None = None
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("type") in ("load_ready", "heartbeat", "load_end"):
                duty = (rec.get("achieved") or {}).get("duty")
                if isinstance(duty, (int, float)):
                    self.duty = duty

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self._thread.join(timeout=5)


def plan_gpu() -> tuple[float | str, str | None]:
    """The gpu section of the soak, with the evidence for skipping it. Only a machine that
    has both the NVIDIA driver and a built gpuload can heat a GPU here; this box has
    AMD cards, so the honest result is a skip with the reason, not a pretend duty."""
    missing = []
    if not os.path.isdir(NVIDIA):
        missing.append(f"no NVIDIA GPU: {NVIDIA} absent")
    if not os.path.isfile(GPULOAD):
        missing.append(f"gpuload not built: {GPULOAD} missing")
    if missing:
        return "skipped", "; ".join(missing)
    return 1.0, None


def current_temp_c() -> float | None:
    """The hottest temperature we can see: amdgpu hwmon temps (every card) and the CPU
    thermal zones. Absent sources are ignored, not zero; a machine with no source at all
    reports None."""
    temps: list[float] = []
    try:
        for dev in amdgpu.cards():
            temps += [v for k, v in amdgpu.hwmon(dev / "device").items() if k.startswith("temp_")]
    except Exception:
        pass
    try:
        temps += list(sampler.thermal_zones().values())
    except Exception:
        pass
    return max(temps) if temps else None


def main() -> None:
    args = parse_args(sys.argv[1:], USAGE)
    soak_min = float(args.get("soak-min", 5))
    deadline_min = float(args.get("deadline-min", 120))
    window = float(args.get("plateau-window", 60))
    rise_c = float(args.get("plateau-rise", 1))
    temp_int = float(args.get("temp-int", 2))
    beat_int = float(args.get("heartbeat-int", 5))
    for name, v in (("soak-min", soak_min), ("deadline-min", deadline_min),
                    ("plateau-window", window), ("plateau-rise", rise_c),
                    ("temp-int", temp_int), ("heartbeat-int", beat_int)):
        if v <= 0:
            print(USAGE + f"\n--{name} must be positive", file=sys.stderr)
            raise SystemExit(2)

    gpu_duty, gpu_note = plan_gpu()
    cpu = Child("cpu", [sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpuload.py"),
                        "--duty", "100", "--parent-pid", str(os.getpid())])
    gpu = Child("gpu", [GPULOAD, "--duty", "100", "--parent-pid", str(os.getpid())]) \
        if gpu_duty == 1.0 else None

    emit("load_start", load="soak",
         intended={"cpu_duty": 1.0, "gpu_duty": gpu_duty, "gpu_note": gpu_note,
                   "soak_min": soak_min,
                   "plateau": {"window_s": window, "rise_c": rise_c}},
         args=sys.argv[1:], deadline_min=deadline_min)
    deadline = install_stop(deadline_min)
    check_parent = make_parent_check(parent_pid_of(args))

    t0 = time.monotonic()
    ready_at_min = t0 + soak_min * 60.0
    # Probed before the heat is on: a machine that reports a temperature has one to
    # plateau; one that reports none has nothing to plateau, so the minimum soak alone
    # gates ready.
    last_temp = current_temp_c()
    need_plateau = last_temp is not None
    plateau_at: float | None = None      # when the rise first stayed under the limit
    samples: deque[tuple[float, float]] = deque()   # (monotonic t, temp C)
    last_temp_t = t0
    last_beat_t = t0 - beat_int          # the first heartbeat comes right away
    ready_emitted = False

    def achieved(temp: float | None) -> dict:
        return {"cpu_duty": cpu.duty,
                "gpu_duty": gpu.duty if gpu is not None else "skipped",
                "temp_c": temp}

    try:
        while True:
            now = time.monotonic()
            check_parent()
            if not cpu.alive():
                emit("load_error",
                     message=f"cpu heat source died (exit {cpu.proc.returncode})")
                raise Stop()
            if now > deadline:
                raise Stop()

            if now - last_temp_t >= temp_int:
                last_temp_t = now
                temp = current_temp_c()
                if temp is not None:
                    last_temp = temp
                    if plateau_at is None:
                        samples.append((now, temp))
                        # Keep a span of at least `window` (plus one sample spacing, so the
                        # window is actually covered): the rise over it must stay under the
                        # limit for the plateau to count.
                        while samples and now - samples[0][0] > window + temp_int:
                            samples.popleft()
                        if now - samples[0][0] >= window:
                            temps = [t for _, t in samples]
                            if max(temps) - min(temps) < rise_c:
                                plateau_at = now

            ready = now >= ready_at_min and (not need_plateau or plateau_at is not None)
            if ready and not ready_emitted:
                ready_emitted = True
                emit("load_ready", achieved=achieved(last_temp),
                     detail={"soaked_s": round(now - t0, 1)})

            if now - last_beat_t >= beat_int:
                last_beat_t = now
                emit("heartbeat", achieved=achieved(last_temp))

            time.sleep(0.2)
    except Stop:
        pass
    finally:
        cpu.stop()
        if gpu is not None:
            gpu.stop()
        emit("load_end", achieved=achieved(current_temp_c()),
             detail={"soaked_s": round(time.monotonic() - t0, 1)})


if __name__ == "__main__":
    main()
