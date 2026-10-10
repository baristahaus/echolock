"""Static environment snapshot recorded alongside every run.

Versions (kernel, driver, CUDA, firmware), the declared device limits, and host settings that
change what a probe measures (governors, THP, swap). Every collector is best-effort: a failure
becomes an entry under "errors" and never stops the run.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from . import amdgpu
from .nvml import Nvml
from .sampler import drm_gpus, meminfo, read_text

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str], timeout: float = 20) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def os_release() -> dict:
    out = {}
    text = read_text("/etc/os-release") or ""
    for line in text.splitlines():
        k, _, v = line.partition("=")
        if k in ("PRETTY_NAME", "ID", "VERSION_ID"):
            out[k.lower()] = v.strip('"')
    return out


def host() -> dict:
    u = platform.uname()
    cpus = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"),
                  key=lambda p: int(p.rsplit("cpu", 1)[1]))
    per_cpu = []
    for c in cpus:
        per_cpu.append({
            "cpu": int(c.rsplit("cpu", 1)[1]),
            "max_khz": read_text(f"{c}/cpufreq/cpuinfo_max_freq"),
            "governor": read_text(f"{c}/cpufreq/scaling_governor"),
            "part": None,
        })
    # Arm core types: MIDR part numbers per CPU, from /proc/cpuinfo.
    cpuinfo = read_text("/proc/cpuinfo") or ""
    idx = -1
    model_name = None
    for line in cpuinfo.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "processor":
            idx = int(v)
        elif k == "CPU part" and 0 <= idx < len(per_cpu):
            per_cpu[idx]["part"] = v
        elif k == "model name" and model_name is None:
            model_name = v
    idle_states = []
    for state in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cpuidle/state*")):
        idle_states.append({"name": read_text(f"{state}/name"),
                            "latency_us": read_text(f"{state}/latency"),
                            "disabled": read_text(f"{state}/disable")})
    if model_name is None:
        # Arm: no `model name` in cpuinfo, but the part numbers are already in per_cpu.
        model_name = arm_cpu_model([c["part"] for c in per_cpu])
    return {
        "hostname": u.node,
        "kernel": u.release,
        "kernel_version": u.version,
        "machine": u.machine,
        "os": os_release(),
        "cmdline": read_text("/proc/cmdline"),
        "cpu_model": model_name,
        "cpus": per_cpu,
        "cpuidle_governor": read_text("/sys/devices/system/cpu/cpuidle/current_governor_ro"),
        "cpuidle_cpu0": idle_states,
        "memory": meminfo(),
        "thp": read_text("/sys/kernel/mm/transparent_hugepage/enabled"),
        "swappiness": read_text("/proc/sys/vm/swappiness"),
        "overcommit_memory": read_text("/proc/sys/vm/overcommit_memory"),
        "dmesg_restrict": read_text("/proc/sys/kernel/dmesg_restrict"),
    }


#/ Arm MIDR part numbers -> core names, enough to name the cores this project runs on. Used only
# to summarise what is already captured per CPU (host.cpus[].part) into host.cpu_model, which
# /proc/cpuinfo leaves absent on Arm: it has no `model name` line there. Two of these are verified
# on hardware (0xd87/0xd85 on GB10, against lscpu); the rest are from Arm's TRM part-number tables
# and are named only so a log reader is not left with a bare hex code. An unknown part keeps its
# hex, because a wrong name is worse than a raw number.
ARM_PART_NAMES = {
    "0xd87": "Cortex-X925",   # verified on GB10 with lscpu
    "0xd85": "Cortex-A725",   # verified on GB10 with lscpu
    "0xd49": "Neoverse N2",
    "0xd4f": "Neoverse V2",
    "0xd0c": "Neoverse N1",
    "0xd4d": "Cortex-A710",
    "0xb45": "Cortex-A510",
    "0xd0d": "Cortex-A77",
    "0xd05": "Cortex-A55",
}


def arm_cpu_model(parts) -> str | None:
    """Summarise Arm MIDR part numbers into one model string, e.g. 'Cortex-X925 x10 + Cortex-A725 x10'.

    Returns None when there is nothing to name, so the caller leaves cpu_model unset rather than
    inventing a name. Counts are kept because an asymmetric host (GB10: 10 X925 + 10 A725) reads
    very differently from a symmetric one, and a single name would hide that.
    """
    counts: dict[str, int] = {}
    for part in parts:
        if not part or not part.strip():
            continue
        name = ARM_PART_NAMES.get(part.strip().lower()) or f"part {part.strip()}"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return " + ".join(f"{name} x{n}" if n > 1 else name for name, n in ordered)


def firmware() -> dict:
    dmi = {k: read_text(f"/sys/class/dmi/id/{k}")
           for k in ("sys_vendor", "product_name", "product_version", "board_name",
                     "bios_vendor", "bios_version", "bios_date")}
    return {"dmi": dmi, "devicetree_model": (read_text("/proc/device-tree/model") or "").strip("\x00")
            or None}


def nvidia_smi() -> dict | None:
    """Driver, CUDA, VBIOS and GSP firmware versions plus declared clocks from nvidia-smi -q -x."""
    text = run_cmd(["nvidia-smi", "-q", "-x"])
    if text is None:
        return None
    root = ET.fromstring(text)

    def t(node: ET.Element | None, path: str) -> str | None:
        if node is None:
            return None
        e = node.find(path)
        return e.text.strip() if e is not None and e.text else None

    def kids(node: ET.Element, path: str) -> list[ET.Element]:
        e = node.find(path)
        return list(e) if e is not None else []

    gpus = []
    for g in root.findall("gpu"):
        gpus.append({
            "pci_bus_id": g.get("id"),
            "product_name": t(g, "product_name"),
            "architecture": t(g, "product_architecture"),
            "vbios_version": t(g, "vbios_version"),
            "gsp_firmware_version": t(g, "gsp_firmware_version"),
            "pci_device_id": t(g, "pci/pci_device_id"),
            "persistence_mode": t(g, "persistence_mode"),
            "addressing_mode": t(g, "addressing_mode"),
            "compute_mode": t(g, "compute_mode"),
            "performance_state": t(g, "performance_state"),
            "clocks": {c.tag: c.text for c in kids(g, "clocks")},
            "max_clocks": {c.tag: c.text for c in kids(g, "max_clocks")},
            "applications_clocks": {c.tag: c.text for c in kids(g, "applications_clocks")},
            "power": {c.tag: c.text for c in kids(g, "gpu_power_readings")
                      if c.text and c.text.strip()},
            "temperature": {c.tag: c.text for c in kids(g, "temperature")},
            "fb_memory": {c.tag: c.text for c in kids(g, "fb_memory_usage")},
        })
    return {"driver_version": t(root, "driver_version"),
            "cuda_version": t(root, "cuda_version"), "gpus": gpus}


def nvidia() -> dict:
    out: dict = {"proc_version": read_text("/proc/driver/nvidia/version")}
    nv = Nvml()
    out["nvml"] = {"available": nv.available, "error": nv.error,
                   "driver_version": nv.driver_version() if nv.available else None,
                   "devices": [nv.describe(h) for h in nv.handles] if nv.available else []}
    out["smi"] = nvidia_smi()
    params = read_text("/proc/driver/nvidia/params")
    if params:
        keep = ("EnableGpuFirmware", "NVreg_", "RegistryDwords", "DynamicPowerManagement",
                "EnableStreamMemOPs", "PreserveVideoMemoryAllocations")
        out["module_params"] = {k.strip(): v.strip() for k, _, v in
                                (line.partition(":") for line in params.splitlines())
                                if any(s in k for s in keep)}
    return out


def toolchain() -> dict:
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    nvcc_out = run_cmd([nvcc, "--version"]) if os.path.exists(nvcc) else None
    # What the probes were built with (backends/cuda/Makefile writes it), so a run says which
    # arch its binaries target even when the toolchain has changed since.
    stamp_path = ROOT / "build" / "cuda" / "build.json"
    build = None
    if stamp_path.is_file():
        try:
            build = json.loads(stamp_path.read_text())
        except ValueError:
            build = {"error": "build/cuda/build.json is not valid JSON"}
    return {
        "nvcc": nvcc_out.strip().splitlines()[-1] if nvcc_out else None,
        "vulkaninfo": (run_cmd(["vulkaninfo", "--summary"]) or "").strip()[:4000] or None,
        "python": platform.python_version(),
        "build": build,
    }


def echolock_revision() -> dict:
    rev = run_cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = run_cmd(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {"commit": rev.strip() if rev else None,
            "dirty": bool(dirty.strip()) if dirty is not None else None}


COLLECTORS: dict[str, Callable[[], object]] = {
    "host": host,
    "firmware": firmware,
    "drm": drm_gpus,
    "nvidia": nvidia,
    "amd": amdgpu.describe_all,
    "toolchain": toolchain,
    "echolock": echolock_revision,
}


def capture(skip: tuple[str, ...] = ()) -> dict:
    snap: dict = {"v": 1, "t_ns": time.monotonic_ns(),
                  "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "errors": {}}
    for name, fn in COLLECTORS.items():
        if name in skip:
            continue
        try:
            snap[name] = fn()
        except Exception as e:  # noqa: BLE001 - one broken collector must not lose the rest
            snap["errors"][name] = f"{type(e).__name__}: {e}"
    return snap


def main() -> None:
    print(json.dumps(capture(), indent=2))


if __name__ == "__main__":
    main()
