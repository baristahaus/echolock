"""1 s environment sampler: GPU clocks, temperature, power and clock-event reasons; host CPU
frequencies, temperatures, memory state and memory pressure.

Each tick writes one JSONL record (src "sampler", type "sample") stamped with CLOCK_MONOTONIC,
the clock the probes and the kernel log use. GPU values come from NVML when it loads, otherwise
from DRM/hwmon sysfs (AMD, Intel, Arm GPUs), so the sampler does not assume NVIDIA.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from pathlib import Path
from typing import IO

from . import amdgpu
from .nvml import Nvml

MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Dirty", "Writeback",
                "SwapTotal", "SwapFree", "AnonPages", "Shmem", "Unevictable", "HugePages_Free")


def read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def read_int(path: str | Path) -> int | None:
    t = read_text(path)
    try:
        return int(t) if t is not None else None
    except ValueError:
        return None


def meminfo() -> dict:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in MEMINFO_KEYS:
                parts = rest.split()
                if parts:
                    out[key + ("_kib" if len(parts) > 1 else "")] = int(parts[0])
    except OSError:
        pass
    return out


def pressure(resource: str) -> dict | None:
    text = read_text(f"/proc/pressure/{resource}")
    if text is None:
        return None
    out = {}
    for line in text.splitlines():
        kind, *fields = line.split()
        out[kind] = {k: float(v) for k, v in (f.split("=") for f in fields)}
    return out


def cpu_freqs_khz() -> list[int | None]:
    cpus = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"),
                  key=lambda p: int(p.rsplit("cpu", 1)[1]))
    return [read_int(f"{c}/cpufreq/scaling_cur_freq") for c in cpus]


def thermal_zones() -> dict[str, float]:
    out = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        milli = read_int(f"{zone}/temp")
        if milli is not None:
            out[f"{Path(zone).name}:{read_text(f'{zone}/type') or '?'}"] = milli / 1000.0
    return out


def hwmon() -> dict[str, float]:
    """Every temp/power/freq/energy hwmon input, scaled to C, W, MHz, J."""
    out = {}
    scale = {"temp": 1e-3, "power": 1e-6, "freq": 1e-6, "energy": 1e-6, "in": 1e-3, "curr": 1e-3}
    for mon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = read_text(f"{mon}/name") or Path(mon).name
        for f in sorted(glob.glob(f"{mon}/*_input") + glob.glob(f"{mon}/power*_average")):
            base = Path(f).name.rsplit("_", 1)[0]
            kind = base.rstrip("0123456789")
            if kind not in scale:
                continue
            v = read_int(f)
            if v is None:
                continue
            label = read_text(f"{mon}/{base}_label") or base
            suffix = "_avg" if f.endswith("_average") else ""
            out[f"{name}/{label}{suffix}"] = v * scale[kind]
    return out


def drm_gpus() -> list[dict]:
    """Vendor-neutral GPU state from DRM sysfs (amdgpu exposes the most; others give ids)."""
    out = []
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]")):
        dev = f"{card}/device"
        entry = {"card": Path(card).name, "vendor": read_text(f"{dev}/vendor"),
                 "device": read_text(f"{dev}/device")}
        driver = os.path.realpath(f"{dev}/driver") if os.path.exists(f"{dev}/driver") else None
        entry["driver"] = Path(driver).name if driver else None
        busy = read_int(f"{dev}/gpu_busy_percent")
        if busy is not None:
            entry["util_gpu_pct"] = busy
        for f in ("pp_dpm_sclk", "pp_dpm_mclk"):
            text = read_text(f"{dev}/{f}")
            if text:
                active = [line for line in text.splitlines() if line.endswith("*")]
                entry[f] = active[0] if active else None
        cur = read_int(f"{card}/gt_cur_freq_mhz") or read_int(f"{card}/gt/gt0/rps_cur_freq_mhz")
        if cur is not None:
            entry["gt_cur_freq_mhz"] = cur
        out.append(entry)
    return out


class Sampler:
    """Samples every `interval` seconds on a background thread until stopped."""

    def __init__(self, out: IO[str], interval: float = 1.0, nvml: Nvml | None = None):
        self.out = out
        self.interval = interval
        self.nvml = nvml if nvml is not None else Nvml()
        self.count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> dict:
        t0 = time.monotonic_ns()
        rec: dict = {"v": 1, "t_ns": t0, "src": "sampler", "type": "sample"}
        rec["gpu"], rec["gpu_source"] = self.gpu_state()
        rec["host"] = {
            "cpu_khz": cpu_freqs_khz(),
            "thermal_c": thermal_zones(),
            "hwmon": hwmon(),
            "mem": meminfo(),
            "psi_memory": pressure("memory"),
            "psi_cpu": pressure("cpu"),
            "loadavg": read_text("/proc/loadavg"),
        }
        rec["sample_cost_us"] = (time.monotonic_ns() - t0) / 1e3
        return rec

    def gpu_state(self) -> tuple[list[dict], str]:
        """GPU values from the first source that has any, in order of trust: NVML when it loads,
        then the amdgpu driver's own sysfs files (an AMD GPU needs no vendor library), then
        whatever DRM exposes. `gpu_source` says which one produced the numbers."""
        if self.nvml.available:
            return ([dict(index=i, **self.nvml.sample(h))
                     for i, h in enumerate(self.nvml.handles)], "nvml")
        cards = amdgpu.sample_all()
        if cards:
            return cards, "amdgpu_sysfs"
        return drm_gpus(), "drm_sysfs"

    def start(self) -> "Sampler":
        self._thread = threading.Thread(target=self._loop, name="sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        # Ticks stay on a fixed grid so a slow sample doesn't drift the series.
        next_t = time.monotonic()
        while not self._stop.is_set():
            self.out.write(json.dumps(self.sample()) + "\n")
            self.out.flush()
            self.count += 1
            next_t += self.interval
            self._stop.wait(max(0.0, next_t - time.monotonic()))


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 runs until Ctrl-C")
    args = ap.parse_args()
    s = Sampler(sys.stdout, args.interval).start()
    try:
        if args.duration:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    s.stop()


if __name__ == "__main__":
    main()
